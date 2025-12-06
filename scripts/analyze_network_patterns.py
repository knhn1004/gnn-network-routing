"""Analyze patterns in Topology Zoo networks to understand performance differences."""

import json
from pathlib import Path
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import List, Dict


def compute_graph_metrics(G: nx.Graph) -> Dict:
    """Compute detailed graph metrics.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary of graph metrics
    """
    metrics = {}

    n = len(G)
    if n == 0:
        return metrics

    # Basic metrics
    metrics["num_nodes"] = n
    metrics["num_edges"] = G.number_of_edges()
    metrics["density"] = G.number_of_edges() / (n * (n - 1) / 2.0) if n > 1 else 0.0

    # Clustering coefficient
    try:
        metrics["avg_clustering"] = nx.average_clustering(G)
    except:
        metrics["avg_clustering"] = 0.0

    # Average shortest path length and diameter (approximate for large graphs)
    if n <= 100:
        try:
            metrics["avg_path_length"] = nx.average_shortest_path_length(
                G, weight="weight"
            )
            metrics["diameter"] = nx.diameter(G)
        except:
            metrics["avg_path_length"] = 0.0
            metrics["diameter"] = 0.0
    else:
        # Approximate for large graphs
        sample_size = min(50, n // 2)
        sample_nodes = np.random.choice(list(G.nodes()), sample_size, replace=False)
        subgraph = G.subgraph(sample_nodes)
        try:
            if nx.is_connected(subgraph):
                metrics["avg_path_length"] = nx.average_shortest_path_length(
                    subgraph, weight="weight"
                )
                metrics["diameter"] = nx.diameter(subgraph)
            else:
                metrics["avg_path_length"] = 0.0
                metrics["diameter"] = 0.0
        except:
            metrics["avg_path_length"] = 0.0
            metrics["diameter"] = 0.0

    # Modularity (community structure)
    try:
        if n <= 200:
            communities = nx.community.greedy_modularity_communities(G)
            metrics["modularity"] = nx.community.modularity(G, communities)
        else:
            # Approximate for large graphs
            metrics["modularity"] = 0.0  # Skip for large graphs (expensive)
    except:
        metrics["modularity"] = 0.0

    # Edge weight statistics
    edge_weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    if edge_weights:
        metrics["weight_mean"] = np.mean(edge_weights)
        metrics["weight_std"] = np.std(edge_weights)
        metrics["weight_min"] = np.min(edge_weights)
        metrics["weight_max"] = np.max(edge_weights)
        metrics["weight_range"] = metrics["weight_max"] - metrics["weight_min"]
    else:
        metrics["weight_mean"] = 0.0
        metrics["weight_std"] = 0.0
        metrics["weight_min"] = 0.0
        metrics["weight_max"] = 0.0
        metrics["weight_range"] = 0.0

    return metrics


def analyze_network_patterns(
    results_file: str = "results/evaluation_results.json", topology_zoo_dir: str = None
):
    """Analyze patterns in well vs poorly performing networks.

    Args:
        results_file: Path to evaluation results JSON file
        topology_zoo_dir: Optional path to Topology Zoo GraphML directory for detailed metrics
    """

    # Load evaluation results
    results_path = Path(results_file)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_file}")
        return

    with open(results_path) as f:
        results = json.load(f)

    networks = results["topology_zoo"]["per_network"]

    # Sort by MAPE
    networks_sorted = sorted(networks, key=lambda x: x["mape"])

    # Get top 10 best and worst
    best_10 = networks_sorted[:10]
    worst_10 = networks_sorted[-10:]

    # Compute detailed graph metrics if topology_zoo_dir is provided
    if topology_zoo_dir:
        print("\nComputing detailed graph metrics...")
        from gnn_routing.data.topology_zoo import load_topology_zoo_networks

        graphml_dir = Path(topology_zoo_dir)
        if graphml_dir.exists():
            # Load graphs and compute metrics
            loaded_networks = load_topology_zoo_networks(
                graphml_dir, min_nodes=50, max_nodes=1000
            )
            graph_dict = {
                name: G for G, meta in loaded_networks for name in [meta["name"]]
            }

            # Add metrics to network results
            for network in networks:
                network_name = network.get("name", "")
                if network_name in graph_dict:
                    G = graph_dict[network_name]
                    metrics = compute_graph_metrics(G)
                    network.update(metrics)

    print("=" * 80)
    print("PATTERN ANALYSIS: Best vs Worst Performing Networks")
    print("=" * 80)
    print("\nTraining Data Characteristics:")
    print("  Node Range: 100-500")
    print(
        "  Graph Types: Erdős–Rényi, Barabási–Albert, Watts–Strogatz, Random Geometric, Powerlaw Cluster"
    )
    print("  Edge Weights: 0.1-10.0 (uniform random)")

    # Calculate statistics
    def calc_stats(network_list, label):
        nodes = [n["num_nodes"] for n in network_list]
        edges = [n["num_edges"] for n in network_list]
        densities = [n["num_edges"] / n["num_nodes"] for n in network_list]
        maes = [n["mae"] for n in network_list]
        mapes = [n["mape"] for n in network_list]

        stats = {
            "label": label,
            "nodes_mean": np.mean(nodes),
            "nodes_std": np.std(nodes),
            "nodes_min": np.min(nodes),
            "nodes_max": np.max(nodes),
            "edges_mean": np.mean(edges),
            "edges_std": np.std(edges),
            "density_mean": np.mean(densities),
            "density_std": np.std(densities),
            "density_min": np.min(densities),
            "density_max": np.max(densities),
            "mae_mean": np.mean(maes),
            "mape_mean": np.mean(mapes),
            "mape_std": np.std(mapes),
        }

        # Add detailed metrics if available
        if "avg_clustering" in network_list[0]:
            stats["avg_clustering_mean"] = np.mean(
                [n.get("avg_clustering", 0.0) for n in network_list]
            )
            stats["avg_path_length_mean"] = np.mean(
                [n.get("avg_path_length", 0.0) for n in network_list]
            )
            stats["diameter_mean"] = np.mean(
                [n.get("diameter", 0.0) for n in network_list]
            )
            stats["modularity_mean"] = np.mean(
                [n.get("modularity", 0.0) for n in network_list]
            )
            stats["weight_mean_mean"] = np.mean(
                [n.get("weight_mean", 0.0) for n in network_list]
            )
            stats["weight_std_mean"] = np.mean(
                [n.get("weight_std", 0.0) for n in network_list]
            )
            stats["weight_range_mean"] = np.mean(
                [n.get("weight_range", 0.0) for n in network_list]
            )

        return stats

    best_stats = calc_stats(best_10, "Best 10")
    worst_stats = calc_stats(worst_10, "Worst 10")
    all_stats = calc_stats(networks, "All Networks")

    print("\n" + "=" * 80)
    print("NETWORK SIZE ANALYSIS")
    print("=" * 80)
    print(f"\n{'Metric':<30} {'Best 10':<20} {'Worst 10':<20} {'All Networks':<20}")
    print("-" * 90)
    print(
        f"{'Avg Nodes':<30} {best_stats['nodes_mean']:<20.1f} {worst_stats['nodes_mean']:<20.1f} {all_stats['nodes_mean']:<20.1f}"
    )
    print(
        f"{'Node Range':<30} {best_stats['nodes_min']:.0f}-{best_stats['nodes_max']:<17.0f} {worst_stats['nodes_min']:.0f}-{worst_stats['nodes_max']:<17.0f} {all_stats['nodes_min']:.0f}-{all_stats['nodes_max']:<17.0f}"
    )
    print(
        f"{'Avg Edges':<30} {best_stats['edges_mean']:<20.1f} {worst_stats['edges_mean']:<20.1f} {all_stats['edges_mean']:<20.1f}"
    )

    print("\n" + "=" * 80)
    print("GRAPH DENSITY ANALYSIS")
    print("=" * 80)
    print(f"\n{'Metric':<30} {'Best 10':<20} {'Worst 10':<20} {'All Networks':<20}")
    print("-" * 90)
    print(
        f"{'Avg Density (E/N)':<30} {best_stats['density_mean']:<20.2f} {worst_stats['density_mean']:<20.2f} {all_stats['density_mean']:<20.2f}"
    )
    print(
        f"{'Density Range':<30} {best_stats['density_min']:.2f}-{best_stats['density_max']:<17.2f} {worst_stats['density_min']:.2f}-{worst_stats['density_max']:<17.2f} {all_stats['density_min']:.2f}-{all_stats['density_max']:<17.2f}"
    )

    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    print(f"\n{'Metric':<30} {'Best 10':<20} {'Worst 10':<20} {'All Networks':<20}")
    print("-" * 90)
    print(
        f"{'Avg MAE':<30} {best_stats['mae_mean']:<20.2f} {worst_stats['mae_mean']:<20.2f} {all_stats['mae_mean']:<20.2f}"
    )
    print(
        f"{'Avg MAPE (%)':<30} {best_stats['mape_mean']:<20.2f} {worst_stats['mape_mean']:<20.2f} {all_stats['mape_mean']:<20.2f}"
    )

    # Analyze size distribution
    print("\n" + "=" * 80)
    print("SIZE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    def count_in_range(network_list, min_nodes, max_nodes):
        return sum(1 for n in network_list if min_nodes <= n["num_nodes"] <= max_nodes)

    training_range = (100, 500)

    best_in_range = count_in_range(best_10, *training_range)
    worst_in_range = count_in_range(worst_10, *training_range)
    all_in_range = count_in_range(networks, *training_range)

    print(
        f"\nNetworks in training range ({training_range[0]}-{training_range[1]} nodes):"
    )
    print(f"  Best 10:  {best_in_range}/10 ({best_in_range*10}%)")
    print(f"  Worst 10: {worst_in_range}/10 ({worst_in_range*10}%)")
    print(
        f"  All:      {all_in_range}/{len(networks)} ({all_in_range/len(networks)*100:.1f}%)"
    )

    # Networks smaller than training range
    best_small = count_in_range(best_10, 0, 99)
    worst_small = count_in_range(worst_10, 0, 99)

    print(f"\nNetworks smaller than training range (<100 nodes):")
    print(f"  Best 10:  {best_small}/10 ({best_small*10}%)")
    print(f"  Worst 10: {worst_small}/10 ({worst_small*10}%)")

    # Analyze density patterns
    print("\n" + "=" * 80)
    print("DENSITY PATTERNS")
    print("=" * 80)

    # Low density (< 1.5 edges/node) vs high density
    def count_by_density(network_list, threshold):
        low = sum(
            1 for n in network_list if n["num_edges"] / n["num_nodes"] < threshold
        )
        high = len(network_list) - low
        return low, high

    best_low, best_high = count_by_density(best_10, 1.5)
    worst_low, worst_high = count_by_density(worst_10, 1.5)

    print(f"\nNetworks with density < 1.5 (sparse):")
    print(f"  Best 10:  {best_low}/10 ({best_low*10}%)")
    print(f"  Worst 10: {worst_low}/10 ({worst_low*10}%)")

    print(f"\nNetworks with density >= 1.5 (dense):")
    print(f"  Best 10:  {best_high}/10 ({best_high*10}%)")
    print(f"  Worst 10: {worst_high}/10 ({worst_high*10}%)")

    # Detailed breakdown of best and worst
    print("\n" + "=" * 80)
    print("BEST PERFORMING NETWORKS (Top 10)")
    print("=" * 80)
    print(
        f"{'Network':<20} {'Nodes':<8} {'Edges':<8} {'Density':<10} {'MAPE (%)':<12} {'MAE':<8}"
    )
    print("-" * 80)
    for n in best_10:
        density = n["num_edges"] / n["num_nodes"]
        print(
            f"{n['name']:<20} {n['num_nodes']:<8} {n['num_edges']:<8} {density:<10.2f} {n['mape']:<12.2f} {n['mae']:<8.2f}"
        )

    print("\n" + "=" * 80)
    print("WORST PERFORMING NETWORKS (Bottom 10)")
    print("=" * 80)
    print(
        f"{'Network':<20} {'Nodes':<8} {'Edges':<8} {'Density':<10} {'MAPE (%)':<12} {'MAE':<8}"
    )
    print("-" * 80)
    for n in worst_10:
        density = n["num_edges"] / n["num_nodes"]
        print(
            f"{n['name']:<20} {n['num_nodes']:<8} {n['num_edges']:<8} {density:<10.2f} {n['mape']:<12.2f} {n['mae']:<8.2f}"
        )

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    insights = []

    if best_stats["nodes_mean"] < worst_stats["nodes_mean"]:
        insights.append(
            f"✓ Best performers are SMALLER on average ({best_stats['nodes_mean']:.1f} vs {worst_stats['nodes_mean']:.1f} nodes)"
        )
    else:
        insights.append(
            f"✗ Best performers are LARGER on average ({best_stats['nodes_mean']:.1f} vs {worst_stats['nodes_mean']:.1f} nodes)"
        )

    if best_stats["density_mean"] < worst_stats["density_mean"]:
        insights.append(
            f"✓ Best performers are SPARSER on average ({best_stats['density_mean']:.2f} vs {worst_stats['density_mean']:.2f} edges/node)"
        )
    else:
        insights.append(
            f"✗ Best performers are DENSER on average ({best_stats['density_mean']:.2f} vs {worst_stats['density_mean']:.2f} edges/node)"
        )

    if best_in_range > worst_in_range:
        insights.append(
            f"✓ More best performers are in training range ({best_in_range}/10 vs {worst_in_range}/10)"
        )
    else:
        insights.append(
            f"✗ Fewer best performers are in training range ({best_in_range}/10 vs {worst_in_range}/10)"
        )

    if best_small > worst_small:
        insights.append(
            f"✓ More best performers are SMALLER than training range ({best_small}/10 vs {worst_small}/10)"
        )
    else:
        insights.append(
            f"✗ Fewer best performers are smaller than training range ({best_small}/10 vs {worst_small}/10)"
        )

    for insight in insights:
        print(f"  {insight}")

    # Print detailed metrics if available
    if "avg_clustering" in networks[0] if networks else False:
        print("\n" + "=" * 80)
        print("DETAILED GRAPH METRICS")
        print("=" * 80)
        print(f"\n{'Metric':<30} {'Best 10':<20} {'Worst 10':<20} {'All Networks':<20}")
        print("-" * 90)
        print(
            f"{'Avg Clustering':<30} {best_stats.get('avg_clustering_mean', 0.0):<20.3f} {worst_stats.get('avg_clustering_mean', 0.0):<20.3f} {all_stats.get('avg_clustering_mean', 0.0):<20.3f}"
        )
        print(
            f"{'Avg Path Length':<30} {best_stats.get('avg_path_length_mean', 0.0):<20.3f} {worst_stats.get('avg_path_length_mean', 0.0):<20.3f} {all_stats.get('avg_path_length_mean', 0.0):<20.3f}"
        )
        print(
            f"{'Avg Diameter':<30} {best_stats.get('diameter_mean', 0.0):<20.3f} {worst_stats.get('diameter_mean', 0.0):<20.3f} {all_stats.get('diameter_mean', 0.0):<20.3f}"
        )
        print(
            f"{'Avg Modularity':<30} {best_stats.get('modularity_mean', 0.0):<20.3f} {worst_stats.get('modularity_mean', 0.0):<20.3f} {all_stats.get('modularity_mean', 0.0):<20.3f}"
        )
        print(
            f"{'Avg Weight Mean':<30} {best_stats.get('weight_mean_mean', 0.0):<20.3f} {worst_stats.get('weight_mean_mean', 0.0):<20.3f} {all_stats.get('weight_mean_mean', 0.0):<20.3f}"
        )
        print(
            f"{'Avg Weight Std':<30} {best_stats.get('weight_std_mean', 0.0):<20.3f} {worst_stats.get('weight_std_mean', 0.0):<20.3f} {all_stats.get('weight_std_mean', 0.0):<20.3f}"
        )
        print(
            f"{'Avg Weight Range':<30} {best_stats.get('weight_range_mean', 0.0):<20.3f} {worst_stats.get('weight_range_mean', 0.0):<20.3f} {all_stats.get('weight_range_mean', 0.0):<20.3f}"
        )

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("\nThe model was trained on synthetic graphs with:")
    print("  - Node range: 100-500")
    print("  - Various graph topologies (ER, BA, WS, etc.)")
    print("  - Random edge weights: 0.1-10.0")
    print("\nKey findings:")
    print(
        f"  1. Best performers average {best_stats['nodes_mean']:.1f} nodes vs worst {worst_stats['nodes_mean']:.1f} nodes"
    )
    print(
        f"  2. Best performers average {best_stats['density_mean']:.2f} edges/node vs worst {worst_stats['density_mean']:.2f} edges/node"
    )
    print(
        f"  3. {best_in_range}/10 best networks are in training range vs {worst_in_range}/10 worst networks"
    )
    print(
        f"  4. {best_small}/10 best networks are smaller than training range vs {worst_small}/10 worst networks"
    )

    if best_stats["nodes_mean"] < 100:
        print(
            "\n⚠️  Most best performers are SMALLER than training range - suggests model"
        )
        print("   may have learned patterns that work better on smaller graphs!")

    if best_stats["density_mean"] < worst_stats["density_mean"]:
        print("\n⚠️  Best performers are SPARSER - suggests model generalizes better to")
        print("   sparse network topologies similar to synthetic graphs.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze network patterns in Topology Zoo results"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="results/evaluation_results.json",
        help="Path to evaluation results JSON file",
    )
    parser.add_argument(
        "--topology_zoo_dir",
        type=str,
        default=None,
        help="Path to Topology Zoo GraphML directory for detailed metrics",
    )
    args = parser.parse_args()
    analyze_network_patterns(args.results_file, args.topology_zoo_dir)
