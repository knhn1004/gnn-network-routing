#!/usr/bin/env python3
"""Exploratory Data Analysis on Topology Zoo networks - compute centrality and graph statistics."""

import json
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Dict
import pandas as pd
import argparse
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn_routing.data.topology_zoo import load_topology_zoo_networks


def compute_centrality_metrics(G: nx.Graph) -> Dict:
    """Compute centrality metrics for a graph.

    Args:
        G: NetworkX graph

    Returns:
        Dictionary of centrality statistics
    """
    metrics = {}
    n = len(G)

    if n == 0:
        return metrics

    # Basic graph properties
    metrics["num_nodes"] = n
    metrics["num_edges"] = G.number_of_edges()
    metrics["density"] = G.number_of_edges() / (n * (n - 1) / 2.0) if n > 1 else 0.0

    # Degree centrality
    degree_centrality = nx.degree_centrality(G)
    degree_values = list(degree_centrality.values())
    metrics["degree_centrality_mean"] = np.mean(degree_values)
    metrics["degree_centrality_std"] = np.std(degree_values)
    metrics["degree_centrality_max"] = np.max(degree_values)

    # Betweenness centrality (sample for large graphs)
    if n <= 200:
        betweenness = nx.betweenness_centrality(G, weight="weight")
        betweenness_values = list(betweenness.values())
        metrics["betweenness_centrality_mean"] = np.mean(betweenness_values)
        metrics["betweenness_centrality_std"] = np.std(betweenness_values)
        metrics["betweenness_centrality_max"] = np.max(betweenness_values)
    else:
        # Sample nodes for large graphs
        sample_size = min(100, n)
        sample_nodes = np.random.choice(list(G.nodes()), sample_size, replace=False)
        betweenness = nx.betweenness_centrality(G, k=sample_size, weight="weight")
        betweenness_values = [betweenness.get(node, 0) for node in G.nodes()]
        metrics["betweenness_centrality_mean"] = np.mean(betweenness_values)
        metrics["betweenness_centrality_std"] = np.std(betweenness_values)
        metrics["betweenness_centrality_max"] = np.max(betweenness_values)

    # Closeness centrality (sample for large graphs)
    if n <= 200:
        try:
            closeness = nx.closeness_centrality(G, distance="weight")
            closeness_values = list(closeness.values())
            metrics["closeness_centrality_mean"] = np.mean(closeness_values)
            metrics["closeness_centrality_std"] = np.std(closeness_values)
            metrics["closeness_centrality_max"] = np.max(closeness_values)
        except:
            metrics["closeness_centrality_mean"] = 0.0
            metrics["closeness_centrality_std"] = 0.0
            metrics["closeness_centrality_max"] = 0.0
    else:
        metrics["closeness_centrality_mean"] = 0.0
        metrics["closeness_centrality_std"] = 0.0
        metrics["closeness_centrality_max"] = 0.0

    # Eigenvector centrality
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=100, weight="weight")
        eigenvector_values = list(eigenvector.values())
        metrics["eigenvector_centrality_mean"] = np.mean(eigenvector_values)
        metrics["eigenvector_centrality_std"] = np.std(eigenvector_values)
        metrics["eigenvector_centrality_max"] = np.max(eigenvector_values)
    except:
        metrics["eigenvector_centrality_mean"] = 0.0
        metrics["eigenvector_centrality_std"] = 0.0
        metrics["eigenvector_centrality_max"] = 0.0

    # PageRank
    try:
        pagerank = nx.pagerank(G, weight="weight")
        pagerank_values = list(pagerank.values())
        metrics["pagerank_mean"] = np.mean(pagerank_values)
        metrics["pagerank_std"] = np.std(pagerank_values)
        metrics["pagerank_max"] = np.max(pagerank_values)
    except:
        metrics["pagerank_mean"] = 0.0
        metrics["pagerank_std"] = 0.0
        metrics["pagerank_max"] = 0.0

    # Clustering coefficient
    try:
        clustering = nx.clustering(G, weight="weight")
        clustering_values = list(clustering.values())
        metrics["clustering_coefficient_mean"] = np.mean(clustering_values)
        metrics["clustering_coefficient_std"] = np.std(clustering_values)
    except:
        metrics["clustering_coefficient_mean"] = 0.0
        metrics["clustering_coefficient_std"] = 0.0

    # Average clustering
    try:
        metrics["avg_clustering"] = nx.average_clustering(G, weight="weight")
    except:
        metrics["avg_clustering"] = 0.0

    # Path length metrics (approximate for large graphs)
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
        # Sample for large graphs
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
            metrics["modularity"] = 0.0
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
        metrics["weight_cv"] = (
            metrics["weight_std"] / metrics["weight_mean"]
            if metrics["weight_mean"] > 0
            else 0.0
        )
        metrics["weight_median"] = np.median(edge_weights)
    else:
        metrics["weight_mean"] = 0.0
        metrics["weight_std"] = 0.0
        metrics["weight_min"] = 0.0
        metrics["weight_max"] = 0.0
        metrics["weight_range"] = 0.0
        metrics["weight_cv"] = 0.0
        metrics["weight_median"] = 0.0

    return metrics


def main():
    parser = argparse.ArgumentParser(description="EDA on Topology Zoo networks")
    parser.add_argument(
        "--topology_zoo_dir",
        type=str,
        default="data/topology_zoo/graphml",
        help="Directory containing Topology Zoo GraphML files",
    )
    parser.add_argument(
        "--min_nodes",
        type=int,
        default=50,
        help="Minimum number of nodes",
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=1000,
        help="Maximum number of nodes",
    )
    parser.add_argument(
        "--max_networks",
        type=int,
        default=None,
        help="Maximum number of networks to analyze (None for all)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/topology_zoo_eda_stats.json",
        help="Output JSON file for detailed stats",
    )
    parser.add_argument(
        "--output_table",
        type=str,
        default="results/topology_zoo_eda_table.md",
        help="Output markdown table file",
    )

    args = parser.parse_args()

    # Load Topology Zoo networks
    graphml_dir = Path(args.topology_zoo_dir)
    if not graphml_dir.exists():
        print(f"Error: Topology Zoo directory not found: {graphml_dir}")
        return

    print(f"Loading Topology Zoo networks from {graphml_dir}...")
    topology_zoo_networks = load_topology_zoo_networks(
        graphml_dir,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        max_networks=args.max_networks,
    )

    if len(topology_zoo_networks) == 0:
        print("No valid Topology Zoo networks found.")
        return

    print(f"Loaded {len(topology_zoo_networks)} networks")
    print("Computing metrics...")

    # Compute metrics for each network
    all_metrics = []
    for G, metadata in topology_zoo_networks:
        metrics = compute_centrality_metrics(G)
        metrics["name"] = metadata["name"]
        all_metrics.append(metrics)

    # Create DataFrame for easy statistics
    df = pd.DataFrame(all_metrics)

    # Compute summary statistics
    summary_stats = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col != "name":
            summary_stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
                "q25": float(df[col].quantile(0.25)),
                "q75": float(df[col].quantile(0.75)),
            }

    # Save detailed results
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(exist_ok=True)

    results = {
        "summary_statistics": summary_stats,
        "per_network": all_metrics,
        "num_networks": len(all_metrics),
    }

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved detailed results to: {args.output_file}")

    # Create markdown table
    table_lines = ["# Topology Zoo EDA Statistics\n"]
    table_lines.append(f"**Total Networks Analyzed:** {len(all_metrics)}\n")
    table_lines.append("## Summary Statistics\n")
    table_lines.append("| Metric | Mean | Std | Min | Median | Max | Q25 | Q75 |")
    table_lines.append("|--------|------|-----|-----|--------|-----|-----|-----|")

    # Sort metrics by category
    metric_categories = {
        "Basic": ["num_nodes", "num_edges", "density"],
        "Centrality - Degree": [
            "degree_centrality_mean",
            "degree_centrality_std",
            "degree_centrality_max",
        ],
        "Centrality - Betweenness": [
            "betweenness_centrality_mean",
            "betweenness_centrality_std",
            "betweenness_centrality_max",
        ],
        "Centrality - Closeness": [
            "closeness_centrality_mean",
            "closeness_centrality_std",
            "closeness_centrality_max",
        ],
        "Centrality - Eigenvector": [
            "eigenvector_centrality_mean",
            "eigenvector_centrality_std",
            "eigenvector_centrality_max",
        ],
        "Centrality - PageRank": ["pagerank_mean", "pagerank_std", "pagerank_max"],
        "Clustering": [
            "clustering_coefficient_mean",
            "clustering_coefficient_std",
            "avg_clustering",
        ],
        "Path Metrics": ["avg_path_length", "diameter"],
        "Community": ["modularity"],
        "Edge Weights": [
            "weight_mean",
            "weight_std",
            "weight_min",
            "weight_max",
            "weight_range",
            "weight_cv",
            "weight_median",
        ],
    }

    for category, metrics in metric_categories.items():
        table_lines.append(f"\n### {category}\n")
        for metric in metrics:
            if metric in summary_stats:
                stats = summary_stats[metric]
                table_lines.append(
                    f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                    f"{stats['min']:.4f} | {stats['median']:.4f} | {stats['max']:.4f} | "
                    f"{stats['q25']:.4f} | {stats['q75']:.4f} |"
                )

    # Add remaining metrics not in categories
    remaining = [
        col
        for col in numeric_cols
        if col not in [m for metrics in metric_categories.values() for m in metrics]
    ]
    if remaining:
        table_lines.append(f"\n### Other Metrics\n")
        for metric in remaining:
            if metric in summary_stats:
                stats = summary_stats[metric]
                table_lines.append(
                    f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                    f"{stats['min']:.4f} | {stats['median']:.4f} | {stats['max']:.4f} | "
                    f"{stats['q25']:.4f} | {stats['q75']:.4f} |"
                )

    with open(args.output_table, "w") as f:
        f.write("\n".join(table_lines))

    print(f"Saved statistics table to: {args.output_table}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("TOPOLOGY ZOO EDA SUMMARY")
    print("=" * 80)
    print(f"\nTotal Networks: {len(all_metrics)}")
    print(f"\nBasic Statistics:")
    print(
        f"  Nodes: {summary_stats['num_nodes']['mean']:.1f} ± {summary_stats['num_nodes']['std']:.1f} "
        f"(range: {summary_stats['num_nodes']['min']:.0f} - {summary_stats['num_nodes']['max']:.0f})"
    )
    print(
        f"  Edges: {summary_stats['num_edges']['mean']:.1f} ± {summary_stats['num_edges']['std']:.1f} "
        f"(range: {summary_stats['num_edges']['min']:.0f} - {summary_stats['num_edges']['max']:.0f})"
    )
    print(
        f"  Density: {summary_stats['density']['mean']:.4f} ± {summary_stats['density']['std']:.4f}"
    )

    print(f"\nCentrality Statistics:")
    print(
        f"  Degree Centrality (mean): {summary_stats['degree_centrality_mean']['mean']:.4f} ± {summary_stats['degree_centrality_mean']['std']:.4f}"
    )
    print(
        f"  Betweenness Centrality (mean): {summary_stats['betweenness_centrality_mean']['mean']:.4f} ± {summary_stats['betweenness_centrality_mean']['std']:.4f}"
    )
    print(
        f"  Closeness Centrality (mean): {summary_stats['closeness_centrality_mean']['mean']:.4f} ± {summary_stats['closeness_centrality_mean']['std']:.4f}"
    )
    print(
        f"  Eigenvector Centrality (mean): {summary_stats['eigenvector_centrality_mean']['mean']:.4f} ± {summary_stats['eigenvector_centrality_mean']['std']:.4f}"
    )
    print(
        f"  PageRank (mean): {summary_stats['pagerank_mean']['mean']:.6f} ± {summary_stats['pagerank_mean']['std']:.6f}"
    )

    print(f"\nClustering:")
    print(
        f"  Average Clustering: {summary_stats['avg_clustering']['mean']:.4f} ± {summary_stats['avg_clustering']['std']:.4f}"
    )

    print(f"\nEdge Weights:")
    print(
        f"  Mean Weight: {summary_stats['weight_mean']['mean']:.2e} ± {summary_stats['weight_mean']['std']:.2e}"
    )
    print(
        f"  Weight CV: {summary_stats['weight_cv']['mean']:.4f} ± {summary_stats['weight_cv']['std']:.4f}"
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
