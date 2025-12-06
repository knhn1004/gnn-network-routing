"""Graph Attention Network (GAT) for shortest path prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from typing import Optional


class GAT(nn.Module):
    """Graph Attention Network for shortest path distance prediction.

    Architecture:
    - Node feature encoder (Linear layers)
    - Edge feature encoder (Linear layers)
    - Multiple GAT layers with multi-head attention
    - Output layer for shortest path distance prediction
    """

    def __init__(
        self,
        node_feature_dim: int = 4,
        edge_feature_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        output_dim: int = 1,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        """Initialize GAT model.

        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of message passing layers
            num_heads: Number of attention heads
            output_dim: Output dimension (1 for shortest path distance)
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super(GAT, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # GAT layers with multi-head attention
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        for i in range(num_layers):
            if i == 0:
                # First layer: input is hidden_dim, output is hidden_dim
                self.convs.append(
                    GATConv(
                        hidden_dim,
                        hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True,
                    )
                )
            elif i == num_layers - 1:
                # Last layer: use average instead of concatenation for final representation
                self.convs.append(
                    GATConv(
                        hidden_dim,
                        hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=False,  # Average heads instead of concatenate
                    )
                )
            else:
                # Middle layers
                self.convs.append(
                    GATConv(
                        hidden_dim,
                        hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True,
                    )
                )

            if use_layer_norm:
                # Layer norm after each GAT layer
                if i == num_layers - 1:
                    # Last layer output dimension is hidden_dim (not concatenated)
                    self.layer_norms.append(nn.LayerNorm(hidden_dim))
                else:
                    # Other layers output dimension is hidden_dim (concatenated heads)
                    self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Output layer: predict shortest path distance
        # We'll use source and target node embeddings to predict distance
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through GAT.

        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, node_feature_dim]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_feature_dim]
                - y: Ground truth shortest path distance (optional)

        Returns:
            Predicted shortest path distance [batch_size, 1]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Encode node features
        x = self.node_encoder(x)

        # Encode edge features (GATConv can use edge_attr)
        edge_attr_encoded = None
        if edge_attr is not None and edge_attr.size(1) > 0:
            edge_attr_encoded = self.edge_encoder(edge_attr)

        # Message passing through GAT layers
        for i, conv in enumerate(self.convs):
            # GATConv supports edge_attr parameter
            if edge_attr_encoded is not None:
                x_new = conv(x, edge_index, edge_attr=edge_attr_encoded)
            else:
                x_new = conv(x, edge_index)

            # Apply layer normalization if enabled
            if self.use_layer_norm and self.layer_norms is not None:
                x_new = self.layer_norms[i](x_new)

            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            # Residual connection (if dimensions match)
            if x.size(1) == x_new.size(1):
                x = x + x_new
            else:
                x = x_new

        # Extract source and target node embeddings
        # Find source and target nodes from node features (one-hot encoding)
        source_mask = data.x[:, 2] == 1.0  # Source indicator is 3rd feature
        target_mask = data.x[:, 3] == 1.0  # Target indicator is 4th feature

        source_idx = torch.where(source_mask)[0]
        target_idx = torch.where(target_mask)[0]

        if len(source_idx) == 0 or len(target_idx) == 0:
            # Fallback: use first and last nodes if source/target not found
            source_idx = torch.tensor([0], device=x.device)
            target_idx = torch.tensor([data.num_nodes - 1], device=x.device)

        source_emb = x[source_idx[0]]  # [hidden_dim]
        target_emb = x[target_idx[0]]  # [hidden_dim]

        # Concatenate source and target embeddings
        combined = torch.cat([source_emb, target_emb], dim=0)  # [hidden_dim * 2]

        # Predict shortest path distance
        output = self.output_layer(combined)  # [output_dim]

        return output
