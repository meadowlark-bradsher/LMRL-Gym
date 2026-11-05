"""R0 behavior-cloning trainer: fit the per-row policy to ``exact_ask``'s choice
and the value head to V*. Supervised only — no environment interaction.

Policy loss is cross-entropy to the row ``exact_ask`` picks; value loss is MSE to
the exact V*. A node counts as *optimal-matched* when the net's argmax row is in
the oracle's within-EPS optimal set (ties with exact_ask's tie-break count), which
is the decision-relevant metric reported alongside exact (same-row) match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import TensorDataset
from .net import NEG_INF, PolicyValueNet


@dataclass
class TrainConfig:
    hidden: int = 128
    epochs: int = 400
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    value_coef: float = 1.0
    temperature: float = 0.02   # distillation temp on Q*; small => concentrate on the optimal row(s)
    ce_coef: float = 0.25       # auxiliary hard CE to exact_ask's pick
    val_frac: float = 0.15
    seed: int = 0
    patience: int = 80          # early stop on val optimal-match


def _soft_targets(q_star: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    """softmax(Q*/T) over valid rows — mass on the within-EPS-optimal set, scaled by the Q gaps."""
    q = q_star.masked_fill(~mask, NEG_INF)
    return torch.softmax(q / temperature, dim=1)


def _split(n: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    g = np.random.default_rng(seed)
    perm = g.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    return perm[n_val:], perm[:n_val]


def _metrics(net: PolicyValueNet, S, Rf, M, best, optimal) -> Dict[str, float]:
    net.eval()
    with torch.no_grad():
        logits, value = net(S, Rf, M)
        pred = torch.argmax(logits, dim=1)
        exact = (pred == best).float().mean().item()
        opt = optimal[torch.arange(len(pred)), pred].float().mean().item()
    return {"exact_match": exact, "optimal_match": opt}


def train(ds: TensorDataset, cfg: TrainConfig = TrainConfig()) -> Tuple[PolicyValueNet, Dict]:
    torch.manual_seed(cfg.seed)
    S = torch.as_tensor(ds.state)
    Rf = torch.as_tensor(ds.rows)
    M = torch.as_tensor(ds.mask)
    best = torch.as_tensor(ds.best_idx)
    optimal = torch.as_tensor(ds.optimal)
    Q = torch.as_tensor(ds.q_star)
    v = torch.as_tensor(ds.v_star)

    tr, va = _split(len(best), cfg.val_frac, cfg.seed)
    tr_t, va_t = torch.as_tensor(tr), torch.as_tensor(va)
    net = PolicyValueNet(hidden=cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val, best_state, best_epoch, hist = -1.0, None, -1, []
    for epoch in range(cfg.epochs):
        net.train()
        perm = tr[torch.randperm(len(tr)).numpy()]
        for i in range(0, len(perm), cfg.batch_size):
            idx = torch.as_tensor(perm[i:i + cfg.batch_size])
            logits, value = net(S[idx], Rf[idx], M[idx])
            logp = F.log_softmax(logits, dim=1)
            soft = _soft_targets(Q[idx], M[idx], cfg.temperature)
            policy_loss = -(soft * logp).sum(dim=1).mean()            # distill from Q*
            loss = policy_loss + cfg.ce_coef * F.cross_entropy(logits, best[idx]) \
                + cfg.value_coef * F.mse_loss(value, v[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            m_tr = _metrics(net, S[tr_t], Rf[tr_t], M[tr_t], best[tr_t], optimal[tr_t])
            m_va = _metrics(net, S[va_t], Rf[va_t], M[va_t], best[va_t], optimal[va_t])
            hist.append({"epoch": epoch, "train": m_tr, "val": m_va})
            if m_va["optimal_match"] > best_val:
                best_val, best_epoch = m_va["optimal_match"], epoch
                best_state = {k: vv.detach().clone() for k, vv in net.state_dict().items()}
            elif epoch - best_epoch > cfg.patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    final = {
        "val_optimal_match": best_val, "best_epoch": best_epoch,
        "train": _metrics(net, S[tr_t], Rf[tr_t], M[tr_t], best[tr_t], optimal[tr_t]),
        "val": _metrics(net, S[va_t], Rf[va_t], M[va_t], best[va_t], optimal[va_t]),
        "full": _metrics(net, S, Rf, M, best, optimal),
        "n_nodes": int(len(best)), "n_train": int(len(tr)), "n_val": int(len(va)),
        "max_rows": ds.max_rows, "history": hist, "config": cfg.__dict__,
    }
    return net, final
