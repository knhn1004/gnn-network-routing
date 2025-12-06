"""Evaluation script with normalized target values for MPNN shortest path prediction.

This script evaluates the model and computes metrics in both normalized and original scales.
This helps test if normalizing targets would improve generalization across different weight scales.

The model is evaluated normally (with unnormalized targets as it was trained), but metrics
are computed in both:
1. Original scale: Model predictions vs original targets (baseline)
2. Normalized scale: Normalized predictions vs normalized targets (shows if normalization helps)

Usage:
    uv run python -m gnn_routing.evaluate_normalized \
        --checkpoint checkpoints/best_model.pt \
        --topology_zoo_dir data/topology_zoo/graphml \
        --normalization_method weight_range \
        --output_dir results

Normalization methods:
    - weight_range: Normalize by (max_weight - min_weight)
    - weight_mean: Normalize by mean weight
    - weight_max: Normalize by max weight
    - log: Log-scale normalization (log10)
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm
import time
import json
from typing import List, Tuple, Dict
import wandb
from datetime import datetime
from dotenv import load_dotenv
import os

from gnn_routing.data import (
    SyntheticGraphGenerator,
    preprocess_graph,
    compute_shortest_paths,
    load_topology_zoo_networks,
)
from gnn_routing.models import create_model


def get_device():
    """Get the best available device: CUDA > MPS > CPU.

    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def normalize_target_by_weight_scale(
    G: nx.Graph, shortest_path_dist: float, normalization_method: str = "weight_range"
) -> Tuple[float, Dict]:
    """Normalize target value based on graph's weight scale.

    Args:
        G: NetworkX graph
        shortest_path_dist: Original shortest path distance
        normalization_method: Method to use for normalization
            - "weight_range": Normalize by (max - min) weight
            - "weight_mean": Normalize by mean weight
            - "weight_max": Normalize by max weight
            - "log": Log-scale normalization

    Returns:
        Tuple of (normalized_target, normalization_info_dict)
    """
    # Get edge weights
    edge_weights = []
    for u, v in G.edges():
        weight = G[u][v].get("weight", 1.0)
        edge_weights.append(weight)

    edge_weights = np.array(edge_weights, dtype=np.float32)

    if len(edge_weights) == 0:
        # No edges, return original
        return shortest_path_dist, {"scale": 1.0, "method": normalization_method}

    weight_min = edge_weights.min()
    weight_max = edge_weights.max()
    weight_mean = edge_weights.mean()
    weight_range = weight_max - weight_min

    if normalization_method == "weight_range":
        if weight_range > 0:
            scale = weight_range
        else:
            scale = weight_max if weight_max > 0 else 1.0
        normalized = shortest_path_dist / scale

    elif normalization_method == "weight_mean":
        scale = weight_mean if weight_mean > 0 else 1.0
        normalized = shortest_path_dist / scale

    elif normalization_method == "weight_max":
        scale = weight_max if weight_max > 0 else 1.0
        normalized = shortest_path_dist / scale

    elif normalization_method == "log":
        normalized = np.log10(shortest_path_dist + 1)
        scale = 1.0  # Log doesn't use scale

    else:
        raise ValueError(f"Unknown normalization method: {normalization_method}")

    norm_info = {
        "scale": float(scale),
        "method": normalization_method,
        "weight_min": float(weight_min),
        "weight_max": float(weight_max),
        "weight_mean": float(weight_mean),
        "weight_range": float(weight_range),
    }

    return float(normalized), norm_info


def preprocess_graph_normalized(
    G: nx.Graph,
    source: int = None,
    target: int = None,
    shortest_path_dist: float = None,
    normalization_method: str = "weight_range",
) -> Tuple:
    """Preprocess graph with normalized target values.

    Args:
        G: NetworkX graph
        source: Source node index
        target: Target node index
        shortest_path_dist: Original shortest path distance
        normalization_method: Method for normalizing targets

    Returns:
        Tuple of (PyG Data object, metadata dict, normalization_info dict)
    """
    # Normalize target if provided
    normalized_target = None
    norm_info = None
    if shortest_path_dist is not None:
        normalized_target, norm_info = normalize_target_by_weight_scale(
            G, shortest_path_dist, normalization_method
        )

    # Use standard preprocessing (which normalizes edge features)
    data, metadata = preprocess_graph(
        G, source=source, target=target, shortest_path_dist=normalized_target
    )

    return data, metadata, norm_info


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(predictions - targets))


def compute_mape(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error."""
    mask = targets != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((predictions[mask] - targets[mask]) / targets[mask])) * 100


def evaluate_model_normalized(
    model: nn.Module,
    graphs: List[nx.Graph],
    n_pairs_per_graph: int = 20,
    device: torch.device = None,
    normalization_method: str = "weight_range",
) -> Tuple[Dict, Dict]:
    """Evaluate model with normalized targets.

    Args:
        model: Trained MPNN model
        graphs: List of NetworkX graphs
        n_pairs_per_graph: Number of pairs per graph
        device: PyTorch device
        normalization_method: Method for normalizing targets

    Returns:
        Tuple of (normalized_metrics_dict, original_scale_metrics_dict)
    """
    if device is None:
        device = get_device()

    model.eval()
    predictions_normalized = []
    targets_normalized = []
    predictions_original = []
    targets_original = []
    inference_times = []
    normalization_scales = []

    with torch.no_grad():
        for G in tqdm(graphs, desc="Evaluating", unit="graph"):
            if not nx.is_connected(G):
                continue

            shortest_paths = compute_shortest_paths(G)
            n = len(G)

            # Sample pairs
            pairs = []
            attempts = 0
            max_attempts = n_pairs_per_graph * 10

            while len(pairs) < n_pairs_per_graph and attempts < max_attempts:
                source = np.random.randint(0, n)
                target = np.random.randint(0, n)

                if source != target and (source, target) in shortest_paths:
                    pairs.append((source, target))
                attempts += 1

            # Evaluate on each pair
            for source, target in pairs:
                shortest_dist_original = shortest_paths[(source, target)]

                # Preprocess normally (model expects unnormalized targets during training)
                # But compute normalization info for comparison
                _, _, norm_info = preprocess_graph_normalized(
                    G,
                    source=source,
                    target=target,
                    shortest_path_dist=shortest_dist_original,
                    normalization_method=normalization_method,
                )

                # Preprocess for model (use standard preprocessing with unnormalized target)
                data, _ = preprocess_graph(
                    G,
                    source=source,
                    target=target,
                    shortest_path_dist=shortest_dist_original,
                )
                data = data.to(device)

                # Inference (model outputs in its trained scale - unnormalized)
                start_time = time.perf_counter()
                output = model(data)
                inference_time = time.perf_counter() - start_time

                pred_original = output.cpu().item()

                # Compute normalized versions for comparison
                if normalization_method == "log":
                    pred_normalized = np.log10(pred_original + 1)
                    target_normalized = np.log10(shortest_dist_original + 1)
                else:
                    # Normalize by the graph's weight scale
                    if norm_info["scale"] > 0:
                        pred_normalized = pred_original / norm_info["scale"]
                        target_normalized = shortest_dist_original / norm_info["scale"]
                    else:
                        pred_normalized = pred_original
                        target_normalized = shortest_dist_original

                # Store both normalized and original
                predictions_normalized.append(pred_normalized)
                targets_normalized.append(target_normalized)
                predictions_original.append(pred_original)
                targets_original.append(shortest_dist_original)
                normalization_scales.append(norm_info["scale"])
                inference_times.append(inference_time)

    # Convert to numpy
    predictions_normalized = np.array(predictions_normalized)
    targets_normalized = np.array(targets_normalized)
    predictions_original = np.array(predictions_original)
    targets_original = np.array(targets_original)

    # Compute metrics in normalized space
    mae_normalized = compute_mae(predictions_normalized, targets_normalized)
    mape_normalized = compute_mape(predictions_normalized, targets_normalized)

    # Compute metrics in original scale
    mae_original = compute_mae(predictions_original, targets_original)
    mape_original = compute_mape(predictions_original, targets_original)

    avg_inference_time = np.mean(inference_times)

    normalized_metrics = {
        "mae": mae_normalized,
        "mape": mape_normalized,
        "avg_inference_time": avg_inference_time,
        "num_samples": len(predictions_normalized),
    }

    original_metrics = {
        "mae": mae_original,
        "mape": mape_original,
        "avg_inference_time": avg_inference_time,
        "num_samples": len(predictions_original),
    }

    return normalized_metrics, original_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MPNN model with normalized targets"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--n_test_graphs", type=int, default=50, help="Number of test graphs"
    )
    parser.add_argument(
        "--n_pairs_per_graph",
        type=int,
        default=20,
        help="Number of (source, target) pairs per graph",
    )
    parser.add_argument(
        "--node_range",
        type=int,
        nargs=2,
        default=[100, 500],
        help="Range of node counts [min, max]",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--topology_zoo_dir",
        type=str,
        default=None,
        help="Directory containing Topology Zoo GraphML files",
    )
    parser.add_argument(
        "--max_topology_zoo_networks",
        type=int,
        default=None,
        help="Maximum number of Topology Zoo networks to evaluate",
    )
    parser.add_argument(
        "--normalization_method",
        type=str,
        default="weight_range",
        choices=["weight_range", "weight_mean", "weight_max", "log"],
        help="Method for normalizing targets",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B logging mode",
    )

    args = parser.parse_args()

    load_dotenv()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize wandb
    wandb_project = args.wandb_project or os.getenv(
        "WANDB_PROJECT", "gnn-network-routing-eval-normalized"
    )
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_run_name = f"{wandb_project}_{dt_str}"

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )

    device = get_device()
    print(f"Using device: {device}")
    print(f"Normalization method: {args.normalization_method}")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint.get("args", {})

    # Determine model type from checkpoint or args
    model_type = model_args.get("model_type", "mpnn")
    num_heads = model_args.get("num_heads", 4)
    use_layer_norm = model_args.get("use_layer_norm", False)

    model = create_model(
        model_type=model_type,
        node_feature_dim=4,
        edge_feature_dim=1,
        hidden_dim=model_args.get("hidden_dim", 64),
        num_layers=model_args.get("num_layers", 3),
        output_dim=1,
        dropout=0.1,
        num_heads=num_heads,
        use_layer_norm=use_layer_norm,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    results = {}

    # Evaluate on synthetic graphs
    print("\nGenerating test graphs...")
    test_generator = SyntheticGraphGenerator(seed=args.seed + 100)
    test_graphs = test_generator.generate_dataset(
        n_graphs=args.n_test_graphs, node_range=tuple(args.node_range)
    )

    print("\nEvaluating on synthetic graphs...")
    norm_metrics, orig_metrics = evaluate_model_normalized(
        model,
        test_graphs,
        args.n_pairs_per_graph,
        device,
        args.normalization_method,
    )

    print(f"\nSynthetic Graphs Results:")
    print(f"  Normalized Scale:")
    print(f"    MAE: {norm_metrics['mae']:.6f}")
    print(f"    MAPE: {norm_metrics['mape']:.2f}%")
    print(f"  Original Scale:")
    print(f"    MAE: {orig_metrics['mae']:.6f}")
    print(f"    MAPE: {orig_metrics['mape']:.2f}%")

    results["synthetic"] = {
        "normalized": norm_metrics,
        "original": orig_metrics,
    }

    # Evaluate on Topology Zoo if provided
    if args.topology_zoo_dir:
        graphml_dir = Path(args.topology_zoo_dir)
        if graphml_dir.exists():
            print(f"\nLoading Topology Zoo networks from {graphml_dir}...")
            topology_zoo_networks = load_topology_zoo_networks(
                graphml_dir,
                min_nodes=50,
                max_nodes=1000,
                max_networks=args.max_topology_zoo_networks,
            )

            if len(topology_zoo_networks) > 0:
                print(f"Loaded {len(topology_zoo_networks)} Topology Zoo networks")
                tz_graphs = [g for g, _ in topology_zoo_networks]

                print("\nEvaluating on Topology Zoo networks...")
                tz_norm_metrics, tz_orig_metrics = evaluate_model_normalized(
                    model,
                    tz_graphs,
                    args.n_pairs_per_graph,
                    device,
                    args.normalization_method,
                )

                print(f"\nTopology Zoo Results:")
                print(f"  Normalized Scale:")
                print(f"    MAE: {tz_norm_metrics['mae']:.6f}")
                print(f"    MAPE: {tz_norm_metrics['mape']:.2f}%")
                print(f"  Original Scale:")
                print(f"    MAE: {tz_orig_metrics['mae']:.6f}")
                print(f"    MAPE: {tz_orig_metrics['mape']:.2f}%")

                results["topology_zoo"] = {
                    "normalized": tz_norm_metrics,
                    "original": tz_orig_metrics,
                }

    # Save results
    results_file = (
        output_dir / f"evaluation_results_normalized_{args.normalization_method}.json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Log to wandb
    wandb.summary.update(
        {
            "eval/synthetic/normalized/mae": norm_metrics["mae"],
            "eval/synthetic/normalized/mape": norm_metrics["mape"],
            "eval/synthetic/original/mae": orig_metrics["mae"],
            "eval/synthetic/original/mape": orig_metrics["mape"],
        }
    )

    if "topology_zoo" in results:
        wandb.summary.update(
            {
                "eval/topology_zoo/normalized/mae": tz_norm_metrics["mae"],
                "eval/topology_zoo/normalized/mape": tz_norm_metrics["mape"],
                "eval/topology_zoo/original/mae": tz_orig_metrics["mae"],
                "eval/topology_zoo/original/mape": tz_orig_metrics["mape"],
            }
        )

    wandb.finish()


if __name__ == "__main__":
    main()
