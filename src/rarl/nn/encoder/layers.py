"""Graphormer attention layers: centrality encoding, spatial encoding, MHA."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor, LongTensor
from torch_geometric.utils import to_dense_batch


class CentralityEncoding(nn.Module):
    """Learnable node embedding indexed by node degree (Ying et al., 2021)."""

    def __init__(self, max_degree: int, node_dim: int):
        super().__init__()
        self.z = nn.Parameter(torch.randn(max_degree, node_dim))
        self.max_degree = max_degree

    def forward(self, degree: LongTensor) -> Tensor:
        return self.z[torch.clamp(degree, 0, self.max_degree - 1)]


class SpatialEncoding(nn.Module):
    """Learnable attention bias indexed by pairwise shortest-path length."""

    def __init__(self, max_path_distance: int):
        super().__init__()
        self.b = nn.Parameter(torch.randn(max_path_distance + 1))
        self.max_path_distance = max_path_distance

    def forward(self, path_lengths: Tensor) -> Tensor:
        """
        :param path_lengths: [B, N, N] integer path lengths.
        :return: Additive attention bias [B, N, N].
        """
        no_path = path_lengths == 0
        idx = torch.clamp(path_lengths - 1, 0, self.max_path_distance)
        bias = F.embedding(idx, self.b.unsqueeze(1)).squeeze(-1)
        bias[no_path] = 0.0
        return bias


class GraphormerAttentionHead(nn.Module):
    """Single attention head for Graphormer."""

    def __init__(self, dim_in: int, dim_qk: int, dim_v: int):
        super().__init__()
        self.dim_qk = dim_qk
        self.q = nn.Linear(dim_in, dim_qk)
        self.k = nn.Linear(dim_in, dim_qk)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(
        self,
        x: Tensor,
        b: Tensor,
        batch: Optional[LongTensor] = None,
        return_attn_logits: bool = False,
    ) -> Tensor:
        """
        :param x: Node features [B*N, x_dim].
        :param b: Spatial bias [B, N, N].
        :param batch: Batch vector [B*N].
        :param return_attn_logits: Return pre-softmax attention logits [B, N, N].
        :return: Updated node features [B*N, dim_v] or logits [B, N, N].
        """
        BxN = x.shape[0]
        if batch is None:
            batch = torch.zeros(BxN, dtype=torch.long, device=x.device)
        x_dense, mask = to_dense_batch(x, batch)
        B, N, _ = x_dense.shape
        assert torch.all(mask), "Unequal node counts across batch — not supported."

        Q = self.q(x_dense)          # [B, N, dim_qk]
        K = self.k(x_dense)          # [B, N, dim_qk]
        a = torch.matmul(Q, K.transpose(1, 2)) / self.dim_qk ** 0.5 + b  # [B, N, N]

        if return_attn_logits:
            return a

        V = self.v(x_dense)
        out = torch.matmul(torch.softmax(a, dim=-1), V)  # [B, N, dim_v]
        return out.view(B * N, -1)


class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_qk: int, dim_v: int):
        super().__init__()
        self.heads = nn.ModuleList([
            GraphormerAttentionHead(dim_in, dim_qk, dim_v) for _ in range(num_heads)
        ])
        self.linear = nn.Linear(num_heads * dim_v, dim_in)

    def forward(self, x: Tensor, b: Tensor, batch: Optional[LongTensor] = None) -> Tensor:
        return self.linear(torch.cat([h(x, b, batch) for h in self.heads], dim=-1))


class GraphormerEncoderLayer(nn.Module):
    """
    One Graphormer layer:  h' = MHA(LN(h)) + h,  h_out = FFN(LN(h')) + h'
    """

    def __init__(self, node_dim: int, n_heads: int, ff_dim: int):
        super().__init__()
        self.attention = GraphormerMultiHeadAttention(n_heads, node_dim, node_dim, node_dim)
        self.ln1 = nn.LayerNorm(node_dim)
        self.ln2 = nn.LayerNorm(node_dim)
        self.ff = nn.Sequential(
            nn.Linear(node_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, node_dim)
        )

    def forward(self, x: Tensor, b: Tensor, batch: Optional[LongTensor] = None) -> Tensor:
        x = self.attention(self.ln1(x), b, batch) + x
        return self.ff(self.ln2(x)) + x
