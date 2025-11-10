"""Analyze patterns in Topology Zoo networks to understand performance differences."""

import json
from pathlib import Path
import numpy as np
from collections import defaultdict


def analyze_network_patterns():
    """Analyze patterns in well vs poorly performing networks."""

    # Load evaluation results
    results_file = Path("results/evaluation_results.json")
    with open(results_file) as f:
        results = json.load(f)

    networks = results["topology_zoo"]["per_network"]

    # Sort by MAPE
    networks_sorted = sorted(networks, key=lambda x: x["mape"])

    # Get top 10 best and worst
    best_10 = networks_sorted[:10]
    worst_10 = networks_sorted[-10:]

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

        return {
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
    analyze_network_patterns()
