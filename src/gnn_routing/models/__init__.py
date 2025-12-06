"""GNN model architectures."""

from typing import Union
from .mpnn import MPNN
from .gat import GAT

__all__ = ["MPNN", "GAT", "create_model"]


def create_model(
    model_type: str,
    node_feature_dim: int = 4,
    edge_feature_dim: int = 1,
    hidden_dim: int = 64,
    num_layers: int = 3,
    output_dim: int = 1,
    dropout: float = 0.1,
    num_heads: int = 4,
    use_layer_norm: bool = True,
) -> Union[MPNN, GAT]:
    """Factory function to create a model by name.

    Args:
        model_type: Type of model - 'mpnn' or 'gat'
        node_feature_dim: Dimension of node features
        edge_feature_dim: Dimension of edge features
        hidden_dim: Hidden dimension for GNN layers
        num_layers: Number of message passing layers
        output_dim: Output dimension (1 for shortest path distance)
        dropout: Dropout rate
        num_heads: Number of attention heads (for GAT only)
        use_layer_norm: Whether to use layer normalization (for GAT only)

    Returns:
        Model instance
    """
    import torch.nn as nn

    if model_type.lower() == "mpnn":
        return MPNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
        )
    elif model_type.lower() == "gat":
        return GAT(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            output_dim=output_dim,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'mpnn' or 'gat'.")
