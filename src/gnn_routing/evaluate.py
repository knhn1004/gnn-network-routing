"""Evaluation script for MPNN shortest path prediction."""

import argparse
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm
import time
import json
from typing import List, Tuple

from gnn_routing.data import (
    SyntheticGraphGenerator,
    preprocess_graph,
    compute_shortest_paths,
    load_topology_zoo_networks,
)
from gnn_routing.models import MPNN


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


def dijkstra_shortest_path(G: nx.Graph, source: int, target: int) -> float:
    """Compute shortest path using Dijkstra's algorithm.

    Args:
        G: NetworkX graph with edge weights
        source: Source node index
        target: Target node index

    Returns:
        Shortest path distance
    """
    try:
        path_length = nx.shortest_path_length(
            G, source=source, target=target, weight="weight"
        )
        return path_length
    except nx.NetworkXNoPath:
        return float("inf")


def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Absolute Error.

    Args:
        predictions: Predicted values
        targets: Ground truth values

    Returns:
        MAE value
    """
    return np.mean(np.abs(predictions - targets))


def compute_mape(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error.

    Args:
        predictions: Predicted values
        targets: Ground truth values

    Returns:
        MAPE value (as percentage)
    """
    # Avoid division by zero
    mask = targets != 0
    if not np.any(mask):
        return 0.0

    return np.mean(np.abs((predictions[mask] - targets[mask]) / targets[mask])) * 100


def evaluate_model(
    model: nn.Module,
    graphs: List[nx.Graph],
    n_pairs_per_graph: int = 20,
    device: torch.device = None,
) -> Tuple[float, float, float, List[float]]:
    """Evaluate model on graphs.

    Args:
        model: Trained MPNN model
        graphs: List of NetworkX graphs to evaluate on
        n_pairs_per_graph: Number of (source, target) pairs per graph
        device: PyTorch device

    Returns:
        Tuple of (MAE, MAPE, avg_inference_time, list of inference times)
    """
    if device is None:
        device = get_device()

    model.eval()
    predictions = []
    targets = []
    inference_times = []

    with torch.no_grad():
        for G in tqdm(graphs, desc="Evaluating", unit="graph"):
            if not nx.is_connected(G):
                continue

            # Compute shortest paths for this graph
            shortest_paths = compute_shortest_paths(G)
            n = len(G)

            # Sample random (source, target) pairs
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
                shortest_dist = shortest_paths[(source, target)]

                # Preprocess graph
                data, _ = preprocess_graph(
                    G, source=source, target=target, shortest_path_dist=shortest_dist
                )
                data = data.to(device)

                # Measure inference time
                start_time = time.perf_counter()
                output = model(data)
                inference_time = time.perf_counter() - start_time

                pred = output.cpu().item()
                predictions.append(pred)
                targets.append(shortest_dist)
                inference_times.append(inference_time)

    predictions = np.array(predictions)
    targets = np.array(targets)

    mae = compute_mae(predictions, targets)
    mape = compute_mape(predictions, targets)
    avg_inference_time = np.mean(inference_times)

    return mae, mape, avg_inference_time, inference_times


def benchmark_dijkstra(
    graphs: List[nx.Graph], n_pairs_per_graph: int = 20
) -> Tuple[float, List[float]]:
    """Benchmark Dijkstra's algorithm.

    Args:
        graphs: List of NetworkX graphs
        n_pairs_per_graph: Number of (source, target) pairs per graph

    Returns:
        Tuple of (average inference time, list of inference times)
    """
    inference_times = []

    for G in tqdm(graphs, desc="Benchmarking Dijkstra", unit="graph"):
        if not nx.is_connected(G):
            continue

        shortest_paths = compute_shortest_paths(G)
        n = len(G)

        # Sample random pairs
        pairs = []
        attempts = 0
        max_attempts = n_pairs_per_graph * 10

        while len(pairs) < n_pairs_per_graph and attempts < max_attempts:
            source = np.random.randint(0, n)
            target = np.random.randint(0, n)

            if source != target and (source, target) in shortest_paths:
                pairs.append((source, target))

            attempts += 1

        # Measure Dijkstra time for each pair
        for source, target in pairs:
            start_time = time.perf_counter()
            dijkstra_shortest_path(G, source, target)
            inference_time = time.perf_counter() - start_time
            inference_times.append(inference_time)

    avg_inference_time = np.mean(inference_times)
    return avg_inference_time, inference_times


def evaluate_model_per_network(
    model: nn.Module,
    graphs: List[nx.Graph],
    graph_metadata: List[dict],
    n_pairs_per_graph: int = 20,
    device: torch.device = None,
) -> List[dict]:
    """Evaluate model on graphs with per-network breakdown.

    Args:
        model: Trained MPNN model
        graphs: List of NetworkX graphs to evaluate on
        graph_metadata: List of metadata dictionaries for each graph
        n_pairs_per_graph: Number of (source, target) pairs per graph
        device: PyTorch device

    Returns:
        List of dictionaries with per-network results
    """
    if device is None:
        device = get_device()

    model.eval()
    per_network_results = []

    with torch.no_grad():
        for G, metadata in tqdm(
            zip(graphs, graph_metadata),
            desc="Evaluating per network",
            total=len(graphs),
            unit="graph",
        ):
            if not nx.is_connected(G):
                continue

            predictions = []
            targets = []
            inference_times = []

            # Compute shortest paths for this graph
            shortest_paths = compute_shortest_paths(G)
            n = len(G)

            # Sample random (source, target) pairs
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
                shortest_dist = shortest_paths[(source, target)]

                # Preprocess graph
                data, _ = preprocess_graph(
                    G, source=source, target=target, shortest_path_dist=shortest_dist
                )
                data = data.to(device)

                # Measure inference time
                start_time = time.perf_counter()
                output = model(data)
                inference_time = time.perf_counter() - start_time

                pred = output.cpu().item()
                predictions.append(pred)
                targets.append(shortest_dist)
                inference_times.append(inference_time)

            if len(predictions) > 0:
                predictions = np.array(predictions)
                targets = np.array(targets)

                mae = compute_mae(predictions, targets)
                mape = compute_mape(predictions, targets)
                avg_inference_time = np.mean(inference_times)

                result = {
                    **metadata,
                    "mae": float(mae),
                    "mape": float(mape),
                    "avg_inference_time": float(avg_inference_time),
                    "num_samples": len(predictions),
                }
                per_network_results.append(result)

    return per_network_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MPNN model")
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
        "--benchmark_dijkstra",
        action="store_true",
        help="Benchmark against Dijkstra algorithm",
    )
    parser.add_argument(
        "--topology_zoo_dir",
        type=str,
        default=None,
        help="Directory containing Topology Zoo GraphML files (e.g., data/topology_zoo/graphml)",
    )
    parser.add_argument(
        "--max_topology_zoo_networks",
        type=int,
        default=None,
        help="Maximum number of Topology Zoo networks to evaluate (None for all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device (CUDA > MPS > CPU)
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint.get("args", {})

    model = MPNN(
        node_feature_dim=4,
        edge_feature_dim=1,
        hidden_dim=model_args.get("hidden_dim", 64),
        num_layers=model_args.get("num_layers", 3),
        output_dim=1,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation MAE: {checkpoint.get('val_mae', 'N/A')}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate test graphs (in-distribution)
    print("\nGenerating in-distribution test graphs...")
    test_generator = SyntheticGraphGenerator(seed=args.seed + 100)
    test_graphs_id = test_generator.generate_dataset(
        n_graphs=args.n_test_graphs, node_range=tuple(args.node_range)
    )

    # Evaluate on in-distribution graphs
    print("\nEvaluating on in-distribution graphs...")
    mae_id, mape_id, avg_time_id, inference_times_id = evaluate_model(
        model, test_graphs_id, args.n_pairs_per_graph, device
    )

    print(f"\nIn-Distribution Results:")
    print(f"  MAE: {mae_id:.6f}")
    print(f"  MAPE: {mape_id:.2f}%")
    print(f"  Avg Inference Time: {avg_time_id*1000:.4f} ms")

    # Evaluate on out-of-distribution graphs (different sizes)
    print("\nGenerating out-of-distribution test graphs...")
    ood_generator = SyntheticGraphGenerator(seed=args.seed + 200)
    # Use different node range for OOD
    ood_node_range = (args.node_range[0] + 100, args.node_range[1] + 200)
    test_graphs_ood = ood_generator.generate_dataset(
        n_graphs=args.n_test_graphs, node_range=ood_node_range
    )

    print("\nEvaluating on out-of-distribution graphs...")
    mae_ood, mape_ood, avg_time_ood, inference_times_ood = evaluate_model(
        model, test_graphs_ood, args.n_pairs_per_graph, device
    )

    print(f"\nOut-of-Distribution Results:")
    print(f"  MAE: {mae_ood:.6f}")
    print(f"  MAPE: {mape_ood:.2f}%")
    print(f"  Avg Inference Time: {avg_time_ood*1000:.4f} ms")

    # Generalization gap
    generalization_gap_mape = mape_ood - mape_id
    print(f"\nGeneralization Gap (MAPE): {generalization_gap_mape:.2f}%")

    # Benchmark Dijkstra if requested
    dijkstra_results = {}
    if args.benchmark_dijkstra:
        print("\nBenchmarking Dijkstra's algorithm...")
        avg_time_dijkstra_id, inference_times_dijkstra_id = benchmark_dijkstra(
            test_graphs_id, args.n_pairs_per_graph
        )
        avg_time_dijkstra_ood, inference_times_dijkstra_ood = benchmark_dijkstra(
            test_graphs_ood, args.n_pairs_per_graph
        )

        print(f"\nDijkstra Results (ID):")
        print(f"  Avg Inference Time: {avg_time_dijkstra_id*1000:.4f} ms")
        print(f"\nDijkstra Results (OOD):")
        print(f"  Avg Inference Time: {avg_time_dijkstra_ood*1000:.4f} ms")

        speedup_id = avg_time_dijkstra_id / avg_time_id
        speedup_ood = avg_time_dijkstra_ood / avg_time_ood

        print(f"\nSpeed Comparison:")
        print(
            f"  GNN is {speedup_id:.2f}x {'faster' if speedup_id > 1 else 'slower'} than Dijkstra (ID)"
        )
        print(
            f"  GNN is {speedup_ood:.2f}x {'faster' if speedup_ood > 1 else 'slower'} than Dijkstra (OOD)"
        )

        dijkstra_results = {
            "avg_time_id": avg_time_dijkstra_id,
            "avg_time_ood": avg_time_dijkstra_ood,
            "speedup_id": speedup_id,
            "speedup_ood": speedup_ood,
        }

    # Evaluate on Topology Zoo networks if provided
    topology_zoo_results = {}
    if args.topology_zoo_dir:
        graphml_dir = Path(args.topology_zoo_dir)
        if not graphml_dir.exists():
            print(f"\nWarning: Topology Zoo directory not found: {graphml_dir}")
        else:
            print(f"\nLoading Topology Zoo networks from {graphml_dir}...")
            topology_zoo_networks = load_topology_zoo_networks(
                graphml_dir,
                min_nodes=50,
                max_nodes=1000,
                max_networks=args.max_topology_zoo_networks,
            )

            if len(topology_zoo_networks) == 0:
                print("No valid Topology Zoo networks found.")
            else:
                print(f"Loaded {len(topology_zoo_networks)} Topology Zoo networks")

                # Extract graphs and metadata
                tz_graphs = [g for g, _ in topology_zoo_networks]
                tz_metadata = [m for _, m in topology_zoo_networks]

                # Evaluate overall
                print("\nEvaluating on Topology Zoo networks...")
                mae_tz, mape_tz, avg_time_tz, inference_times_tz = evaluate_model(
                    model, tz_graphs, args.n_pairs_per_graph, device
                )

                print(f"\nTopology Zoo Results:")
                print(f"  MAE: {mae_tz:.6f}")
                print(f"  MAPE: {mape_tz:.2f}%")
                print(f"  Avg Inference Time: {avg_time_tz*1000:.4f} ms")

                # Per-network breakdown
                print("\nComputing per-network breakdown...")
                per_network_results = evaluate_model_per_network(
                    model, tz_graphs, tz_metadata, args.n_pairs_per_graph, device
                )

                # Benchmark Dijkstra on Topology Zoo if requested
                if args.benchmark_dijkstra:
                    print("\nBenchmarking Dijkstra on Topology Zoo...")
                    avg_time_dijkstra_tz, inference_times_dijkstra_tz = (
                        benchmark_dijkstra(tz_graphs, args.n_pairs_per_graph)
                    )
                    print(f"\nDijkstra Results (Topology Zoo):")
                    print(f"  Avg Inference Time: {avg_time_dijkstra_tz*1000:.4f} ms")

                    speedup_tz = avg_time_dijkstra_tz / avg_time_tz
                    print(f"\nSpeed Comparison (Topology Zoo):")
                    print(
                        f"  GNN is {speedup_tz:.2f}x {'faster' if speedup_tz > 1 else 'slower'} than Dijkstra"
                    )

                    dijkstra_results["avg_time_tz"] = avg_time_dijkstra_tz
                    dijkstra_results["speedup_tz"] = speedup_tz

                # Comparison with synthetic
                print(f"\nComparison with Synthetic Graphs:")
                print(f"  MAPE Gap (TZ vs ID): {mape_tz - mape_id:.2f}%")
                print(f"  MAPE Gap (TZ vs OOD): {mape_tz - mape_ood:.2f}%")

                topology_zoo_results = {
                    "overall": {
                        "mae": mae_tz,
                        "mape": mape_tz,
                        "avg_inference_time": avg_time_tz,
                        "num_samples": len(inference_times_tz),
                        "num_networks": len(tz_graphs),
                    },
                    "per_network": per_network_results,
                    "comparison": {
                        "mape_gap_vs_id": mape_tz - mape_id,
                        "mape_gap_vs_ood": mape_tz - mape_ood,
                    },
                }

    # Save results
    results = {
        "in_distribution": {
            "mae": mae_id,
            "mape": mape_id,
            "avg_inference_time": avg_time_id,
            "num_samples": len(inference_times_id),
        },
        "out_of_distribution": {
            "mae": mae_ood,
            "mape": mape_ood,
            "avg_inference_time": avg_time_ood,
            "num_samples": len(inference_times_ood),
        },
        "generalization": {"mape_gap": generalization_gap_mape},
        "dijkstra": dijkstra_results,
        "topology_zoo": topology_zoo_results,
    }

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
