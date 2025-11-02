# GNN Network Routing

Learning shortest path heuristics with Graph Neural Networks for cross-topology generalization.

## Overview

This project explores whether a Graph Neural Network (GNN) can approximate shortest-path computation and generalize across different network topologies. By training on synthetic graphs and evaluating on real-world ISP networks, we aim to benchmark the GNN's ability to learn routing heuristics that transfer across network structures.

## Features

- **Synthetic Graph Generation**: Generate diverse network topologies using NetworkX (Erdős–Rényi, Barabási–Albert, Watts–Strogatz, etc.)
- **MPNN Architecture**: Message Passing Neural Network implemented with PyTorch Geometric
- **Training Pipeline**: End-to-end training with early stopping and checkpointing
- **Evaluation**: Comprehensive evaluation with MAE/MAPE metrics and speed benchmarking against Dijkstra's algorithm

## Installation

This project uses `uv` for dependency management:

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

## Usage

### Training

Train the MPNN model on synthetic graphs:

```bash
uv run python -m gnn_routing.train \
    --n_train_graphs 100 \
    --n_val_graphs 20 \
    --n_pairs_per_graph 10 \
    --node_range 100 500 \
    --hidden_dim 64 \
    --num_layers 3 \
    --batch_size 32 \
    --lr 0.001 \
    --epochs 100 \
    --patience 10 \
    --checkpoint_dir checkpoints
```

## Usage

### Training

Train the MPNN model on synthetic graphs:

```bash
uv run python -m gnn_routing.train \
    --n_train_graphs 100 \
    --n_val_graphs 20 \
    --n_pairs_per_graph 10 \
    --node_range 100 500 \
    --hidden_dim 64 \
    --num_layers 3 \
    --batch_size 32 \
    --lr 0.001 \
    --epochs 100 \
    --patience 10 \
    --checkpoint_dir checkpoints
```

### Evaluation

Evaluate a trained model on synthetic graphs:

```bash
uv run python -m gnn_routing.evaluate \
    --checkpoint checkpoints/best_model.pt \
    --n_test_graphs 50 \
    --n_pairs_per_graph 20 \
    --benchmark_dijkstra \
    --output_dir results
```

### Internet Topology Zoo Evaluation

1. Download Topology Zoo networks:

```bash
uv run python scripts/download_topology_zoo.py --output_dir data/topology_zoo
```

2. Evaluate on Topology Zoo networks:

```bash
uv run python -m gnn_routing.evaluate \
    --checkpoint checkpoints/best_model.pt \
    --topology_zoo_dir data/topology_zoo/graphml \
    --max_topology_zoo_networks 50 \
    --benchmark_dijkstra \
    --output_dir results
```

This will evaluate the model on real-world ISP networks from the [Internet Topology Zoo](https://topology-zoo.org/dataset.html) and compare performance with synthetic graphs.

## Project Structure

```
gnn-network-routing/
├── pyproject.toml          # uv project configuration
├── README.md               # This file
├── .gitignore             # Git ignore patterns
├── src/
│   └── gnn_routing/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── generator.py      # Synthetic graph generation
│       │   ├── preprocessing.py  # Data preprocessing and feature engineering
│       │   └── topology_zoo.py   # Internet Topology Zoo network loading
│       ├── models/
│       │   ├── __init__.py
│       │   └── mpnn.py          # MPNN architecture
│       ├── train.py              # Training script
│       └── evaluate.py           # Evaluation script
├── scripts/
│   └── download_topology_zoo.py  # Download Topology Zoo networks
├── data/
│   └── topology_zoo/            # Downloaded Topology Zoo networks
└── notebooks/
    └── exploration.ipynb         # Optional: for data exploration
```

## Model Architecture

The MPNN (Message Passing Neural Network) consists of:

- **Node Feature Encoder**: Linear layers processing node features (degree, betweenness centrality, source/target indicators)
- **Edge Feature Encoder**: Linear layers processing edge features (normalized edge weights)
- **Message Passing Layers**: Graph convolutional layers (GCNConv) for message passing
- **Output Layer**: Predicts shortest path distance from source and target node embeddings

## Data Format

- **Node Features**: [normalized_degree, normalized_betweenness, is_source, is_target]
- **Edge Features**: [normalized_edge_weight]
- **Target**: Shortest path distance (computed using Floyd-Warshall)

## Metrics

- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Inference Time**: Per-query latency
- **Generalization Gap**: Difference between OOD and ID MAPE

## License

MIT

