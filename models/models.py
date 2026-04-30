from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, GCNConv, APPNP, GATConv
except ImportError as exc:
    raise ImportError("torch_geometric is required. Install it with: pip install torch-geometric") from exc


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = torch.relu(h)
        out = self.conv2(h, edge_index)
        return out


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class APPNPModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, K: int = 5, alpha: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.prop = APPNP(K=K, alpha=alpha)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.mlp(x)
        out = self.prop(h, edge_index)
        return out


class ResidualGAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads for GATConv concat output.")

        self.dropout = dropout
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.gat1 = GATConv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.gat2 = GATConv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)

        h1 = self.gat1(h, edge_index)
        h = self.norm1(h + h1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h2 = self.gat2(h, edge_index)
        h = self.norm2(h + h2)
        h = F.relu(h)

        return self.pred_head(h)


def get_model(
    name: str,
    in_dim: int,
    hidden_dim: int = 32,
    heads: int = 4,
    dropout: float = 0.0,
    K: int = 5,
    alpha: float = 0.2,
) -> nn.Module:
    key = name.lower()
    if key in ("graphsage", "graphsage_model"):
        return GraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim)
    if key in ("gcn", "gcn_model"):
        return GCN(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    if key in ("appnp", "appnpmodel"):
        return APPNPModel(in_dim=in_dim, hidden_dim=hidden_dim, K=K, alpha=alpha)
    if key in ("residualgat", "residual_gat"):
        return ResidualGAT(in_dim=in_dim, hidden_dim=hidden_dim, heads=heads, dropout=dropout)
    raise ValueError(f"Unknown model: {name}")
