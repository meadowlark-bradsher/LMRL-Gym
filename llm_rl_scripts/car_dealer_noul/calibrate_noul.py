"""CD-NOUL-2 Phase 0 constants from the frozen Stage 1 matrices. No model calls; deterministic.

Two things, per answer decoder (``noul``, ``sonnet``):

* a **temperature** T for the read p_yes -> sigmoid(T * logit(p_yes)), fit by minimum log-loss of the
  truthful answer over the Stage 1 bank (pooled over regimes; the per-regime fits are reported so the
  registration can see how much T depends on how guarded the text is). 0.5 is a fixed point, so a
  deflection stays uninformative; a monotone map cannot change accuracy, only confidence.
* the **per-question expected read error** a_q = mean over the bank items of that question and regime of
  min(p', 1 - p') under the pooled T — the flip rate the Stage 2 planner's ASK branch uses for question q
  (``LiveConfig.answer_noise_by_q``), one 9-vector per decoder x regime; the raw (T = 1) vectors are
  reported beside them.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_noul.calibrate_noul --bank results/noul_bank_20261108

writes ``<bank>/calibration_noul2.json``. The registration pins this file's hash."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from llm_rl_scripts.car_dealer_binned.questions import BANK

from .decode import CAL_EPS, calibrated_p_yes

DECODERS = ("noul", "sonnet")
REGIMES = ("plain", "oblique")
QUESTION_IDS = tuple(q.id for q in BANK)
T_MIN, T_MAX = 0.05, 50.0
CONFIDENT = 0.9
MAX_NOISE = 0.49          # ``Question.likelihoods`` needs a flip rate strictly below 1/2


def load_rows(bank_dir: Path, decoder: str) -> List[dict]:
    rows = []
    with (Path(bank_dir) / f"matrix_{decoder}.jsonl").open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append({"id": r["id"], "p_yes": float(r["p_yes"]), "truthful": r["truthful"], "regime": r["regime"],
                             "question_id": r["question_id"]})
    return rows


def read(p_yes: float, T: float, b: float = 0.0) -> float:
    """Platt read sigmoid(T * logit(p) + b); b = 0 is the pure temperature (``decode.calibrated_p_yes``)."""
    if b == 0.0:
        return calibrated_p_yes(p_yes, T)
    p = min(max(float(p_yes), CAL_EPS), 1.0 - CAL_EPS)
    z = T * math.log(p / (1.0 - p)) + b
    return 1.0 / (1.0 + math.exp(-z))


def _p_truth(row: dict, T: float, b: float = 0.0) -> float:
    p = min(max(read(row["p_yes"], T, b), CAL_EPS), 1.0 - CAL_EPS)
    return p if row["truthful"] == "yes" else 1.0 - p


def nll_bits(rows: Iterable[dict], T: float, b: float = 0.0) -> float:
    rows = list(rows)
    return -sum(math.log2(_p_truth(r, T, b)) for r in rows) / len(rows)


def fit_temperature(rows: List[dict], lo: float = T_MIN, hi: float = T_MAX, iters: int = 80) -> float:
    """Golden-section search on log T; the temperature log-loss is convex in T."""
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll_bits(rows, math.exp(c)), nll_bits(rows, math.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll_bits(rows, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll_bits(rows, math.exp(d))
    return math.exp((a + b) / 2.0)


def fit_platt(rows: List[dict], rounds: int = 12) -> Tuple[float, float]:
    """(T, b) by alternating golden-section searches; the logistic log-loss is jointly convex in (T, b)."""
    def golden(f, a, b_, iters=60):
        g = (math.sqrt(5.0) - 1.0) / 2.0
        c, d = b_ - g * (b_ - a), a + g * (b_ - a)
        fc, fd = f(c), f(d)
        for _ in range(iters):
            if fc < fd:
                b_, d, fd = d, c, fc
                c = b_ - g * (b_ - a); fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b_ - a); fd = f(d)
        return (a + b_) / 2.0
    T, b = 1.0, 0.0
    for _ in range(rounds):
        T = math.exp(golden(lambda lt: nll_bits(rows, math.exp(lt), b), math.log(T_MIN), math.log(T_MAX)))
        b = golden(lambda bb: nll_bits(rows, T, bb), -4.0, 4.0)
    return T, b


def accuracy(rows: Iterable[dict]) -> float:
    rows = list(rows)
    hit = sum(1 for r in rows if (r["p_yes"] > 0.5) == (r["truthful"] == "yes") and r["p_yes"] != 0.5)
    return hit / len(rows)


def wrong_confident(rows: Iterable[dict], T: float, thr: float = CONFIDENT, b: float = 0.0) -> int:
    n = 0
    for r in rows:
        p = read(r["p_yes"], T, b)
        if (p > thr and r["truthful"] == "no") or (p < 1.0 - thr and r["truthful"] == "yes"):
            n += 1
    return n


def expected_error(rows: Iterable[dict], T: float, b: float = 0.0) -> float:
    rows = list(rows)
    return sum(min(p, 1.0 - p) for p in (read(r["p_yes"], T, b) for r in rows)) / len(rows)


def a_q_table(rows: List[dict], T: float, b: float = 0.0) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for rg in REGIMES:
        out[rg] = {}
        for q in QUESTION_IDS:
            sub = [r for r in rows if r["regime"] == rg and r["question_id"] == q]
            out[rg][q] = min(expected_error(sub, T, b), MAX_NOISE) if sub else None
    return out


def confident_wrong_ids(rows: Iterable[dict], T: float, thr: float = CONFIDENT, b: float = 0.0) -> List[str]:
    out = []
    for r in rows:
        p = read(r["p_yes"], T, b)
        if (p > thr and r["truthful"] == "no") or (p < 1.0 - thr and r["truthful"] == "yes"):
            out.append(r.get("id", ""))
    return out


def safe_temperature(rows: List[dict], lo: float = 1.0, hi: float = T_MAX, iters: int = 60) -> float:
    """The largest T in [lo, hi] with no confident-wrong read on ``rows`` (bisection; ``lo`` if none)."""
    if wrong_confident(rows, lo) > 0:
        return lo
    if wrong_confident(rows, hi) == 0:
        return hi
    a, b = lo, hi
    for _ in range(iters):
        m = (a + b) / 2.0
        if wrong_confident(rows, m) == 0:
            a = m
        else:
            b = m
    return a


def metrics_at(rows: List[dict], by_regime: Dict[str, List[dict]], T: float, b: float = 0.0) -> dict:
    return {"temperature": T, "bias": b,
            "logloss_bits": {"pooled": nll_bits(rows, T, b), **{rg: nll_bits(by_regime[rg], T, b) for rg in REGIMES}},
            "accuracy": {rg: sum(1 for r in by_regime[rg] if (read(r["p_yes"], T, b) > 0.5) == (r["truthful"] == "yes")) / len(by_regime[rg])
                         for rg in REGIMES},
            "wrong_confident": {rg: wrong_confident(by_regime[rg], T, b=b) for rg in REGIMES},
            "expected_error": {rg: expected_error(by_regime[rg], T, b) for rg in REGIMES},
            "a_q": a_q_table(rows, T, b)}


def as_noise_map(a_q: Dict[str, float], denominator: int = 10000) -> Tuple[Tuple[str, Fraction], ...]:
    """The ``LiveConfig.answer_noise_by_q`` tuple for one decoder x regime, as exact Fractions."""
    return tuple((q, Fraction(a_q[q]).limit_denominator(denominator)) for q in QUESTION_IDS)


def _sha16(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def calibration(bank_dir: Path, decoders: Tuple[str, ...] = DECODERS) -> dict:
    bank_dir = Path(bank_dir)
    out = {"schema": "calibration_noul2-v1", "eps": CAL_EPS, "confident": CONFIDENT, "max_noise": MAX_NOISE,
           "t_range": [T_MIN, T_MAX], "questions": list(QUESTION_IDS), "inputs": {}, "decoders": {}}
    for d in decoders:
        rows = load_rows(bank_dir, d)
        out["inputs"][f"matrix_{d}.jsonl"] = _sha16(bank_dir / f"matrix_{d}.jsonl")
        T = fit_temperature(rows)
        by_regime = {rg: [r for r in rows if r["regime"] == rg] for rg in REGIMES}
        rec = {
            "n": len(rows),
            "temperature_pooled": T,
            "temperature_by_regime": {rg: fit_temperature(by_regime[rg]) for rg in REGIMES},
            "logloss_bits": {
                "pooled": {"raw": nll_bits(rows, 1.0), "calibrated": nll_bits(rows, T)},
                **{rg: {"raw": nll_bits(by_regime[rg], 1.0), "calibrated": nll_bits(by_regime[rg], T)} for rg in REGIMES},
            },
            "accuracy": {rg: accuracy(by_regime[rg]) for rg in REGIMES},
            "wrong_confident": {rg: {"raw": wrong_confident(by_regime[rg], 1.0), "calibrated": wrong_confident(by_regime[rg], T)}
                                for rg in REGIMES},
            "expected_error": {rg: {"raw": expected_error(by_regime[rg], 1.0), "calibrated": expected_error(by_regime[rg], T)}
                               for rg in REGIMES},
            "a_q": a_q_table(rows, T),
            "a_q_raw": a_q_table(rows, 1.0),
        }
        # candidate temperatures for the registration to choose among: the pooled fit, the oblique-only
        # fit, the raw decoder, and the largest T that creates no confident-wrong read on the oblique bank
        t_safe = safe_temperature(by_regime["oblique"])
        cands = {"raw": (1.0, 0.0), "pooled": (T, 0.0), "oblique_fit": (rec["temperature_by_regime"]["oblique"], 0.0),
                 "safe": (t_safe, 0.0), "platt": fit_platt(rows), "platt_oblique": fit_platt(by_regime["oblique"])}
        rec["candidates"] = {name: metrics_at(rows, by_regime, t, b) for name, (t, b) in cands.items()}
        rec["confident_wrong_ids_pooled_oblique"] = confident_wrong_ids(by_regime["oblique"], T)
        out["decoders"][d] = rec
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank", default="results/noul_bank_20261108")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cal = calibration(Path(args.bank))
    out = Path(args.out) if args.out else Path(args.bank) / "calibration_noul2.json"
    with out.open("w") as f:
        json.dump(cal, f, indent=1)
    for d, rec in cal["decoders"].items():
        ll = rec["logloss_bits"]
        for name, c in rec["candidates"].items():
            print(f"   {d:7s} {name:13s} T {c['temperature']:7.3f} b {c['bias']:+.3f}  acc oblique {c['accuracy']['oblique']:.3f}  logloss plain {c['logloss_bits']['plain']:.3f} oblique {c['logloss_bits']['oblique']:.3f}"
                  f"  wrong_confident oblique {c['wrong_confident']['oblique']:3d}  E[err] plain {c['expected_error']['plain']:.3f} oblique {c['expected_error']['oblique']:.3f}")
        print(f"{d:7s} T_pooled {rec['temperature_pooled']:.3f}  T_by_regime " +
              " ".join(f"{rg} {t:.3f}" for rg, t in rec["temperature_by_regime"].items()) +
              f"  logloss pooled {ll['pooled']['raw']:.3f} -> {ll['pooled']['calibrated']:.3f} bits" +
              f"  wrong_confident oblique {rec['wrong_confident']['oblique']['raw']} -> {rec['wrong_confident']['oblique']['calibrated']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
