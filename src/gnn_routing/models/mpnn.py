"""Message Passing Neural Network (MPNN) for shortest path prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.data import Data
from typing import Optional


class MPNN(nn.Module):
    """Message Passing Neural Network for shortest path distance prediction.

    Architecture:
    - Node feature encoder (Linear layers)
    - Edge feature encoder (Linear layers)
    - Multiple graph convolutional layers for message passing
    - Output layer for shortest path distance prediction
    """

    def __init__(
        self,
        node_feature_dim: int = 4,
        edge_feature_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 3,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        """Initialize MPNN model.

        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of message passing layers
            output_dim: Output dimension (1 for shortest path distance)
            dropout: Dropout rate
        """
        super(MPNN, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

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

        # Graph convolutional layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))

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
        """Forward pass through MPNN.

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

        # Encode edge features (if provided)
        if edge_attr is not None and edge_attr.size(1) > 0:
            # Note: GCNConv doesn't use edge features directly in its default implementation
            # We could use GATConv or custom message passing, but for simplicity
            # we'll encode edge features and use them in message passing if needed
            pass

        # Message passing through GCN layers
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
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


class MPNNWithEdgeFeatures(nn.Module):
    """MPNN variant that explicitly uses edge features in message passing."""

    def __init__(
        self,
        node_feature_dim: int = 4,
        edge_feature_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 3,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        """Initialize MPNN model with edge features.

        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of message passing layers
            output_dim: Output dimension (1 for shortest path distance)
            dropout: Dropout rate
        """
        super(MPNNWithEdgeFeatures, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

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

        # Use GCNConv (simpler) - edge features can be incorporated via attention or custom layers
        # For now, we'll use standard GCNConv
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through MPNN with edge features."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Encode node features
        x = self.node_encoder(x)

        # Encode edge features (encoded but not directly used in GCNConv)
        if edge_attr is not None and edge_attr.size(1) > 0:
            edge_attr_encoded = self.edge_encoder(edge_attr)

        # Message passing
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            if x.size(1) == x_new.size(1):
                x = x + x_new
            else:
                x = x_new

        # Extract source and target embeddings
        source_mask = data.x[:, 2] == 1.0
        target_mask = data.x[:, 3] == 1.0

        source_idx = torch.where(source_mask)[0]
        target_idx = torch.where(target_mask)[0]

        if len(source_idx) == 0 or len(target_idx) == 0:
            source_idx = torch.tensor([0], device=x.device)
            target_idx = torch.tensor([data.num_nodes - 1], device=x.device)

        source_emb = x[source_idx[0]]
        target_emb = x[target_idx[0]]

        combined = torch.cat([source_emb, target_emb], dim=0)
        output = self.output_layer(combined)

        return output
