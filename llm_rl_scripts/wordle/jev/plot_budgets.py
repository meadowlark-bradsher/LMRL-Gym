"""Per-budget figure for the Wordle Jev benchmark (supports the "flat across
budgets" claim, report section 3.3).

Three small multiples vs the guess budget, all from the SAME cluster bootstrap
(by target word) that analyze_wordle.py uses for the reported numbers -- so the
figure regenerates exactly like the table:

  (a) solve rate  -- Jev vs the exact optimum; the quantity that SEPARATES as the
      budget tightens.
  (b) match@1     -- fraction of Jev turns that pick an exact-EIG-optimal word;
      FLAT across budgets.
  (c) mean regret -- bits of expected information Jev leaves on the table; FLAT.

Each budget's point comes from a labelled result dir (printed and stamped on the
figure). The 3-guess budget also carries the census run (236 words x 5 rollouts)
as a separate, more-precise marker.

    conda run --no-capture-output -n LLM_RL python -u \
        -m llm_rl_scripts.wordle.jev.plot_budgets
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from llm_rl_scripts.wordle.jev import analyze_wordle as aw  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGDIR = HERE / "figures"
BOOT = 2000
BOOT_SEED = 0

# Okabe-Ito, colorblind-safe. Two entities, two hues: identity follows the
# entity, not the run. The census point is still *Jev* (a more precise estimate
# of the same quantity), so it shares Jev's gold and is set apart by its marker
# (a dark-edged diamond), not a third color.
C_JEV = "#E69F00"   # gold (Jev)
C_OPT = "#0072B2"   # blue (optimal)
INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"

# (budget, result dir, json filename, kind). Order = plot order on x.
RUNS = [
    (3, "live_200_t3_20261025", "results.json", "sampled"),
    (4, "live_200_t4_20261025", "results.json", "sampled"),
    (6, "live_200_20261025", "results.json", "sampled"),
]
CENSUS = (3, "live_200_t3_census_20261025", "summary.json")


def by_target_from_resultsjson(path: Path) -> Dict[str, List[dict]]:
    """Reshape eval_jev's --out JSON (jev_games + optimal_games, aligned by
    index) into the per-target record dict analyze_wordle expects, so the same
    bootstrap runs on it."""
    d = json.loads(path.read_text())
    jev, opt = d["jev_games"], d["optimal_games"]
    assert len(jev) == len(opt), "jev/optimal game lists misaligned"
    by_target: Dict[str, List[dict]] = defaultdict(list)
    for jg, og in zip(jev, opt):
        assert jg["target"] == og["target"], "paired games disagree on target"
        by_target[jg["target"]].append({
            "jev_solved": jg["solved"], "jev_guesses": jg["guesses"],
            "opt_solved": og["solved"], "opt_guesses": og["guesses"],
            "jev_turns": [t for t in jg.get("turns", []) if "match" in t],
        })
    return dict(by_target)


def estimate(path: Path) -> Dict[str, object]:
    bt = by_target_from_resultsjson(path)
    pe = aw.point_estimates(bt)
    ci = aw.bootstrap(bt, BOOT, BOOT_SEED)
    return {"pe": pe, "ci": ci,
            "n_targets": len(bt),
            "n_games": sum(len(v) for v in bt.values())}


def yerr(pe, ci, key):
    v, (lo, hi) = pe[key], ci[key]
    return [[v - lo], [hi - v]]


def main():
    rows = []
    for budget, d, fname, kind in RUNS:
        est = estimate(RESULTS / d / fname)
        est.update(budget=budget, dir=d, kind=kind)
        rows.append(est)
        print("budget={} [{}] n_targets={} n_games={}  jev_solve={:.3f} "
              "opt_solve={:.3f} match@1={:.3f} regret={:.3f}".format(
                  budget, d, est["n_targets"], est["n_games"],
                  est["pe"]["jev_solve"], est["pe"]["opt_solve"],
                  est["pe"]["match_at_1"], est["pe"]["mean_regret_bits"]))
    cb, cdir, cfname = CENSUS
    census = estimate(RESULTS / cdir / cfname)
    print("budget={} [{}] CENSUS n_targets={} n_games={}  jev_solve={:.3f} "
          "opt_solve={:.3f} match@1={:.3f} regret={:.3f}".format(
              cb, cdir, census["n_targets"], census["n_games"],
              census["pe"]["jev_solve"], census["pe"]["opt_solve"],
              census["pe"]["match_at_1"], census["pe"]["mean_regret_bits"]))

    budgets = [r["budget"] for r in rows]
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(12.5, 4.2))

    # (a) solve rate: Jev vs optimal -- separates.
    axA.errorbar(budgets, [r["pe"]["opt_solve"] for r in rows],
                 yerr=_stack([yerr(r["pe"], r["ci"], "opt_solve") for r in rows]),
                 color=C_OPT, marker="o", lw=2, capsize=3, label="optimal (exact EIG)")
    axA.errorbar(budgets, [r["pe"]["jev_solve"] for r in rows],
                 yerr=_stack([yerr(r["pe"], r["ci"], "jev_solve") for r in rows]),
                 color=C_JEV, marker="o", lw=2, capsize=3, label="Jev (Choice)")
    axA.errorbar([cb], [census["pe"]["jev_solve"]],
                 yerr=yerr(census["pe"], census["ci"], "jev_solve"),
                 color=C_JEV, marker="D", ms=7, capsize=3, ls="none",
                 markeredgecolor=INK, markeredgewidth=1.2,
                 label="Jev, census (236×5)", zorder=5)
    axA.set_title("(a) solve rate  — separates", fontsize=11, color=INK)
    axA.set_ylim(0, 1.02)
    axA.legend(frameon=False, fontsize=8.5, loc="lower right")
    # direct labels at the tightest budget
    axA.annotate("optimal", (budgets[0], rows[0]["pe"]["opt_solve"]),
                 textcoords="offset points", xytext=(6, 6), color=C_OPT, fontsize=8.5)
    axA.annotate("Jev", (budgets[0], rows[0]["pe"]["jev_solve"]),
                 textcoords="offset points", xytext=(6, -12), color=C_JEV, fontsize=8.5)

    # (b) match@1 -- flat.
    axB.errorbar(budgets, [r["pe"]["match_at_1"] for r in rows],
                 yerr=_stack([yerr(r["pe"], r["ci"], "match_at_1") for r in rows]),
                 color=C_JEV, marker="o", lw=2, capsize=3)
    axB.errorbar([cb], [census["pe"]["match_at_1"]],
                 yerr=yerr(census["pe"], census["ci"], "match_at_1"),
                 color=C_JEV, marker="D", ms=7, capsize=3, ls="none",
                 markeredgecolor=INK, markeredgewidth=1.2)
    axB.set_title("(b) match@1 (Jev turns)  — flat", fontsize=11, color=INK)
    axB.set_ylim(0, 1.02)

    # (c) mean regret -- flat.
    axC.errorbar(budgets, [r["pe"]["mean_regret_bits"] for r in rows],
                 yerr=_stack([yerr(r["pe"], r["ci"], "mean_regret_bits") for r in rows]),
                 color=C_JEV, marker="o", lw=2, capsize=3)
    axC.errorbar([cb], [census["pe"]["mean_regret_bits"]],
                 yerr=yerr(census["pe"], census["ci"], "mean_regret_bits"),
                 color=C_JEV, marker="D", ms=7, capsize=3, ls="none",
                 markeredgecolor=INK, markeredgewidth=1.2)
    axC.set_title("(c) mean regret  — flat", fontsize=11, color=INK)
    axC.set_ylim(0, 1.5)
    axC.set_ylabel("bits", fontsize=9, color=MUTED)

    for ax in (axA, axB, axC):
        ax.set_xlabel("guess budget (tighter →)", fontsize=9, color=MUTED)
        ax.set_xticks(budgets)
        ax.invert_xaxis()  # tighter budgets (where it bites) on the right
        ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)

    fig.suptitle("Wordle × Jev: selection is flat across budgets; only its cost "
                 "(solve rate) changes", fontsize=12.5, color=INK, y=0.99)

    prov = ("plot_budgets.py · analyze_wordle.bootstrap (cluster by target, "
            "B={}, seed {}) · jev-latest, wordle_official_200.txt · "
            "b3={}, b4={}, b6={}, b3★(census)={}").format(
                BOOT, BOOT_SEED, RUNS[0][1], RUNS[1][1], RUNS[2][1], CENSUS[1])
    fig.text(0.5, 0.005, prov, ha="center", fontsize=6.5, color=MUTED)

    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    png, pdf = FIGDIR / "budget_sweep.png", FIGDIR / "budget_sweep.pdf"
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    print("\nwrote {}\nwrote {}".format(png, pdf))


def _stack(pairs):
    """Combine per-point [[lo],[hi]] asymmetric errors into a 2xN array."""
    lo = [p[0][0] for p in pairs]
    hi = [p[1][0] for p in pairs]
    return [lo, hi]


if __name__ == "__main__":
    main()
