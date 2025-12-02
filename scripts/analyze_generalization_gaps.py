"""Deep dive analysis into which network properties correlate with generalization success."""

import json
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import pearsonr, spearmanr
import argparse


def compute_correlation_analysis(networks: List[Dict]) -> Dict:
    """Compute correlations between network properties and performance.
    
    Args:
        networks: List of network dictionaries with metrics and performance
        
    Returns:
        Dictionary of correlation results
    """
    correlations = {}
    
    # Extract metrics
    metrics_to_analyze = [
        'num_nodes', 'num_edges', 'mape', 'mae',
        'avg_clustering', 'avg_path_length', 'diameter', 'modularity',
        'weight_mean', 'weight_std', 'weight_range'
    ]
    
    # Filter networks that have all required metrics
    valid_networks = [
        n for n in networks
        if all(key in n for key in ['mape', 'num_nodes', 'num_edges'])
    ]
    
    if len(valid_networks) < 10:
        print("Warning: Not enough networks with complete metrics for correlation analysis")
        return correlations
    
    # Compute correlations with MAPE (lower is better)
    for metric in metrics_to_analyze:
        if metric in ['mape', 'mae']:
            continue  # Skip performance metrics themselves
        
        values = [n.get(metric, None) for n in valid_networks]
        mapes = [n.get('mape', None) for n in valid_networks]
        
        # Filter out None values
        valid_pairs = [(v, m) for v, m in zip(values, mapes) if v is not None and m is not None]
        
        if len(valid_pairs) >= 10:
            metric_vals, mape_vals = zip(*valid_pairs)
            try:
                pearson_r, pearson_p = pearsonr(metric_vals, mape_vals)
                spearman_r, spearman_p = spearmanr(metric_vals, mape_vals)
                
                correlations[metric] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'n_samples': len(valid_pairs),
                }
            except:
                pass
    
    return correlations


def analyze_generalization_gaps(
    results_file: str = "results/evaluation_results.json",
    topology_zoo_dir: str = None,
    output_file: str = "results/generalization_gap_analysis.json"
):
    """Analyze which network properties correlate with generalization success.
    
    Args:
        results_file: Path to evaluation results JSON file
        topology_zoo_dir: Optional path to Topology Zoo GraphML directory
        output_file: Path to save detailed analysis results
    """
    # Load evaluation results
    results_path = Path(results_file)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_file}")
        return
    
    with open(results_path) as f:
        results = json.load(f)

    networks = results["topology_zoo"]["per_network"]
    
    print("=" * 80)
    print("GENERALIZATION GAP ANALYSIS")
    print("=" * 80)
    
    # Compute detailed graph metrics if topology_zoo_dir is provided
    if topology_zoo_dir:
        print("\nComputing detailed graph metrics...")
        from gnn_routing.data.topology_zoo import load_topology_zoo_networks
        
        def compute_graph_metrics(G: nx.Graph) -> Dict:
            """Compute detailed graph metrics."""
            metrics = {}
            n = len(G)
            if n == 0:
                return metrics
            
            metrics['num_nodes'] = n
            metrics['num_edges'] = G.number_of_edges()
            metrics['density'] = G.number_of_edges() / (n * (n - 1) / 2.0) if n > 1 else 0.0
            
            try:
                metrics['avg_clustering'] = nx.average_clustering(G)
            except:
                metrics['avg_clustering'] = 0.0
            
            if n <= 100:
                try:
                    metrics['avg_path_length'] = nx.average_shortest_path_length(G, weight='weight')
                    metrics['diameter'] = nx.diameter(G)
                except:
                    metrics['avg_path_length'] = 0.0
                    metrics['diameter'] = 0.0
            else:
                sample_size = min(50, n // 2)
                sample_nodes = np.random.choice(list(G.nodes()), sample_size, replace=False)
                subgraph = G.subgraph(sample_nodes)
                try:
                    if nx.is_connected(subgraph):
                        metrics['avg_path_length'] = nx.average_shortest_path_length(subgraph, weight='weight')
                        metrics['diameter'] = nx.diameter(subgraph)
                    else:
                        metrics['avg_path_length'] = 0.0
                        metrics['diameter'] = 0.0
                except:
                    metrics['avg_path_length'] = 0.0
                    metrics['diameter'] = 0.0
            
            try:
                if n <= 200:
                    communities = nx.community.greedy_modularity_communities(G)
                    metrics['modularity'] = nx.community.modularity(G, communities)
                else:
                    metrics['modularity'] = 0.0
            except:
                metrics['modularity'] = 0.0
            
            edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
            if edge_weights:
                metrics['weight_mean'] = np.mean(edge_weights)
                metrics['weight_std'] = np.std(edge_weights)
                metrics['weight_min'] = np.min(edge_weights)
                metrics['weight_max'] = np.max(edge_weights)
                metrics['weight_range'] = metrics['weight_max'] - metrics['weight_min']
            else:
                metrics['weight_mean'] = 0.0
                metrics['weight_std'] = 0.0
                metrics['weight_min'] = 0.0
                metrics['weight_max'] = 0.0
                metrics['weight_range'] = 0.0
            
            return metrics
        
        graphml_dir = Path(topology_zoo_dir)
        if graphml_dir.exists():
            loaded_networks = load_topology_zoo_networks(graphml_dir, min_nodes=50, max_nodes=1000)
            graph_dict = {meta['name']: G for G, meta in loaded_networks}
            
            # Add metrics to network results
            for network in networks:
                network_name = network.get('name', '')
                if network_name in graph_dict:
                    G = graph_dict[network_name]
                    metrics = compute_graph_metrics(G)
                    network.update(metrics)
    
    # Sort by MAPE
    networks_sorted = sorted(networks, key=lambda x: x["mape"])
    
    # Define performance groups
    n_total = len(networks_sorted)
    top_quartile = networks_sorted[:n_total // 4]  # Best 25%
    bottom_quartile = networks_sorted[-n_total // 4:]  # Worst 25%
    
    print(f"\nAnalyzing {n_total} networks")
    print(f"  Top quartile (best): {len(top_quartile)} networks")
    print(f"  Bottom quartile (worst): {len(bottom_quartile)} networks")
    
    # Compute correlations
    print("\nComputing correlations between network properties and performance...")
    correlations = compute_correlation_analysis(networks)
    
    # Print correlation results
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS (with MAPE - lower is better)")
    print("=" * 80)
    print(f"\n{'Metric':<25} {'Pearson r':<15} {'Pearson p':<15} {'Spearman r':<15} {'N':<10}")
    print("-" * 80)
    
    significant_correlations = []
    for metric, corr_data in sorted(correlations.items(), key=lambda x: abs(x[1]['pearson_r']), reverse=True):
        pearson_r = corr_data['pearson_r']
        pearson_p = corr_data['pearson_p']
        spearman_r = corr_data['spearman_r']
        n = corr_data['n_samples']
        
        significance = "***" if pearson_p < 0.001 else "**" if pearson_p < 0.01 else "*" if pearson_p < 0.05 else ""
        
        print(f"{metric:<25} {pearson_r:>10.3f}{significance:<5} {pearson_p:>10.3e} {spearman_r:>10.3f} {n:>10}")
        
        if abs(pearson_r) > 0.3 and pearson_p < 0.05:
            significant_correlations.append({
                'metric': metric,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'interpretation': 'positive' if pearson_r > 0 else 'negative'
            })
    
    # Compare top vs bottom quartile
    print("\n" + "=" * 80)
    print("TOP QUARTILE vs BOTTOM QUARTILE COMPARISON")
    print("=" * 80)
    
    def get_stat(network_list, key, func=np.mean):
        values = [n.get(key, None) for n in network_list]
        valid_values = [v for v in values if v is not None]
        return func(valid_values) if valid_values else None
    
    comparison_metrics = [
        'num_nodes', 'num_edges', 'avg_clustering', 'avg_path_length',
        'diameter', 'modularity', 'weight_mean', 'weight_std', 'weight_range'
    ]
    
    print(f"\n{'Metric':<25} {'Top Quartile':<20} {'Bottom Quartile':<20} {'Difference':<15}")
    print("-" * 80)
    
    for metric in comparison_metrics:
        top_val = get_stat(top_quartile, metric)
        bottom_val = get_stat(bottom_quartile, metric)
        
        if top_val is not None and bottom_val is not None:
            diff = top_val - bottom_val
            diff_pct = (diff / bottom_val * 100) if bottom_val != 0 else 0
            print(f"{metric:<25} {top_val:>15.3f} {bottom_val:>15.3f} {diff:>12.3f} ({diff_pct:>6.1f}%)")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    if significant_correlations:
        print("\nSignificant correlations (|r| > 0.3, p < 0.05):")
        for corr in significant_correlations:
            direction = "increases" if corr['pearson_r'] > 0 else "decreases"
            print(f"  • {corr['metric']}: {direction} MAPE (r={corr['pearson_r']:.3f}, p={corr['pearson_p']:.3e})")
    else:
        print("\nNo strong correlations found (|r| > 0.3, p < 0.05)")
    
    # Save detailed results
    analysis_results = {
        'correlations': correlations,
        'significant_correlations': significant_correlations,
        'top_quartile_stats': {
            metric: {
                'mean': get_stat(top_quartile, metric, np.mean),
                'std': get_stat(top_quartile, metric, np.std),
                'median': get_stat(top_quartile, metric, np.median),
            }
            for metric in comparison_metrics
        },
        'bottom_quartile_stats': {
            metric: {
                'mean': get_stat(bottom_quartile, metric, np.mean),
                'std': get_stat(bottom_quartile, metric, np.std),
                'median': get_stat(bottom_quartile, metric, np.median),
            }
            for metric in comparison_metrics
        },
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nDetailed analysis saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze generalization gaps in Topology Zoo results")
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
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/generalization_gap_analysis.json",
        help="Path to save detailed analysis results",
    )
    args = parser.parse_args()
    analyze_generalization_gaps(args.results_file, args.topology_zoo_dir, args.output_file)

