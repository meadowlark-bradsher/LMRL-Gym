"""The CD-DRL policy/value network (PyTorch, CPU-sized).

A shared MLP encodes the state features into ``h``; a per-row MLP scores each menu
row from ``[row_features | h]`` into one logit (softmax over rows = the policy, so
the head is permutation-invariant and handles the variable-length menu); a value
head reads ``h`` into a scalar V. Rows are padded to a common width per batch with
a boolean mask; padded logits are set to -inf before the softmax.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .features import ROW_DIM, STATE_DIM

NEG_INF = -1e9


def _mlp(sizes, out_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if out_act or i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class PolicyValueNet(nn.Module):
    def __init__(self, hidden: int = 128, state_dim: int = STATE_DIM, row_dim: int = ROW_DIM):
        super().__init__()
        self.encoder = _mlp([state_dim, hidden, hidden], out_act=True)
        self.row_scorer = _mlp([hidden + row_dim, hidden, 1])
        self.value_head = _mlp([hidden, hidden // 2, 1])

    def forward(self, state_feats: torch.Tensor, row_feats: torch.Tensor, row_mask: torch.Tensor):
        """``state_feats`` [B, STATE_DIM]; ``row_feats`` [B, R, ROW_DIM]; ``row_mask``
        [B, R] (True = real row). Returns (logits [B, R] with padded entries -inf,
        value [B])."""
        h = self.encoder(state_feats)                                  # [B, H]
        b, r, _ = row_feats.shape
        h_exp = h.unsqueeze(1).expand(b, r, h.shape[-1])               # [B, R, H]
        scorer_in = torch.cat([h_exp, row_feats], dim=-1)             # [B, R, H+ROW_DIM]
        logits = self.row_scorer(scorer_in).squeeze(-1)                # [B, R]
        logits = logits.masked_fill(~row_mask, NEG_INF)
        value = self.value_head(h).squeeze(-1)                         # [B]
        return logits, value

    @torch.no_grad()
    def choose_index(self, state_feat, row_feat, row_mask) -> int:
        """Greedy argmax row index for a single node (numpy/tensor inputs)."""
        sf = torch.as_tensor(state_feat, dtype=torch.float32).unsqueeze(0)
        rf = torch.as_tensor(row_feat, dtype=torch.float32).unsqueeze(0)
        mk = torch.as_tensor(row_mask, dtype=torch.bool).unsqueeze(0)
        logits, _ = self.forward(sf, rf, mk)
        return int(torch.argmax(logits[0]).item())
