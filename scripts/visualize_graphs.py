#!/usr/bin/env python3
"""Visualize train and test graphs using NetworkX and matplotlib."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn_routing.data.generator import SyntheticGraphGenerator


def visualize_graph(G, ax, title, pos=None):
    """Visualize a single graph on the given axes.

    Args:
        G: NetworkX graph
        ax: Matplotlib axes
        title: Title for the plot
        pos: Optional node positions (will compute if None)
    """
    if pos is None:
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)

    # Get edge weights for coloring
    edges = G.edges()
    weights = [G[u][v].get("weight", 1.0) for u, v in edges]

    # Normalize weights for color mapping
    if weights:
        min_weight = min(weights)
        max_weight = max(weights)
        if max_weight > min_weight:
            normalized_weights = [
                (w - min_weight) / (max_weight - min_weight) for w in weights
            ]
        else:
            normalized_weights = [0.5] * len(weights)
    else:
        normalized_weights = [0.5] * len(edges)

    # Draw edges with color based on weight
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color=normalized_weights,
        edge_cmap=plt.cm.viridis,
        width=1.5,
        alpha=0.6,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color="lightblue",
        node_size=30,
        alpha=0.9,
    )

    # Add labels for smaller graphs
    if len(G.nodes()) <= 50:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")

    # Add graph info as text
    info_text = f"Nodes: {len(G.nodes())}\nEdges: {G.number_of_edges()}"
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def main():
    """Generate and visualize train and test graphs."""
    # Set random seed for reproducibility
    seed = 42

    # Default parameters matching training script
    node_range = (100, 500)
    weight_range = (0.1, 10.0)

    print("Generating 5 training graphs...")
    train_generator = SyntheticGraphGenerator(seed=seed)
    train_graphs = train_generator.generate_dataset(
        n_graphs=5,
        node_range=node_range,
        weight_range=weight_range,
        use_augmentation=False,
    )

    print("Generating 5 test graphs...")
    test_generator = SyntheticGraphGenerator(seed=seed + 100)
    test_graphs = test_generator.generate_dataset(
        n_graphs=5,
        node_range=node_range,
        weight_range=weight_range,
        use_augmentation=False,
    )

    # Create figure with subplots: 2 rows (train, test) x 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Train and Test Graph Visualizations", fontsize=16, fontweight="bold")

    # Visualize train graphs (top row)
    for i, G in enumerate(train_graphs):
        ax = axes[0, i]
        visualize_graph(G, ax, f"Train Graph {i+1}")

    # Visualize test graphs (bottom row)
    for i, G in enumerate(test_graphs):
        ax = axes[1, i]
        visualize_graph(G, ax, f"Test Graph {i+1}")

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "graph_visualizations.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path}")

    # Also show the plot
    plt.show()


if __name__ == "__main__":
    main()
