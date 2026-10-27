"""Render JEV_RESULTS.md from the frozen Jev-run analysis and verdict JSON.

No number is typed by hand. Sources: ``results/binned_b_jev_r2_easy_<date>/``
and ``results/binned_b_jev_r6_easy_<date>/`` (``analysis.json``, ``verdicts.json``,
``summary.json``) plus the frozen scripted ``random`` arm for the sanity gate.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.render_jev_note \
        --primary results/binned_b_jev_r2_easy_20261026 \
        --secondary results/binned_b_jev_r6_easy_20261026 \
        --out llm_rl_scripts/car_dealer_binned/JEV_RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ARMS = ["exact", "greedy_eig", "myopic", "random", "jev_plain", "jev_eig"]


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def ci(d):
    if d is None:
        return "—"
    return f"{d['delta']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"


def rho(d):
    if d is None:
        return "—"
    if d.get("ci95") is None:
        return f"{d['rho']:+.3f} (no CI)"
    return f"{d['rho']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] (n={d['n']})"


def load(d):
    return (
        json.load(open(os.path.join(d, "analysis.json"))),
        json.load(open(os.path.join(d, "verdicts.json"))),
        json.load(open(os.path.join(d, "summary.json"))),
    )


def section(w, title, d, a, v, s, scripted_random):
    w(f"## {title}")
    w("")
    w(f"Source: `{d}` — `analysis.json` (`analyze_condition_b.py`, B = {a['bootstrap']['B']}, seed {a['bootstrap']['seed']}), `verdicts.json` (`verdicts_b.py`). "
      f"Config: rounds = {s['config']['rounds']}, regime {s['config']['regime']} (r = {s['config']['reservation']}), K = {s['repeats']} for stochastic arms, "
      f"{s['episodes']} episodes, {s['error_stubs']} error stubs, option order shuffled = {s['jev_shuffle_options']}, model `jev-latest` via `{s['jev_backend']}`.")
    w("")
    w("### Verdicts (rules fixed in `PREREGISTRATION_B.md`)")
    w("")
    w("| rule | quantity | prediction | estimate [95 % CI] | verdict |")
    w("|---|---|---|---|---|")
    for k, r in v["rules"].items():
        if "verdict" in r:
            w(f"| **{k}** | {r['quantity']} | {r['prediction']} | {ci(r['estimate'])} | **{r['verdict']}** |")
        else:
            for arm, pa in r["per_arm"].items():
                w(f"| **{k}** ({arm}) | {r['quantity']} | {r['prediction']} | large-gap ρ {rho(pa['large_gap'])}; small-gap ρ {rho(pa['small_gap'])}; overall ρ {rho(pa['overall'])} | **{pa['pattern']}** |")
    w("")
    w("### Arms")
    w("")
    w("| arm | episodes | U | gain G | agreement | V* − U | Σ decision regret | zero-regret share | EIG regret / decision (bits) | questions | offers | final H (bits) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        x = a["by_arm"].get(arm)
        if not x:
            continue
        w(f"| `{arm}` | {x['episodes']} | {f4(x['mean_learner_utility'])} | {f4(x['mean_gain'])} | {f3(x['agreement_rate'])} | {f4(x['mean_optimality_gap'])} | "
          f"{f4(x['mean_total_decision_regret'])} | {f3(x['zero_regret_decision_share'])} | {f4(x['mean_eig_regret_per_decision_bits'])} | {x['mean_questions_asked']:.2f} | {x['mean_offers_made']:.2f} | {f3(x['mean_final_entropy_bits'])} |")
    w("")
    w("### Jev-specific")
    w("")
    for arm in ("jev_plain", "jev_eig"):
        j = a["by_arm"].get(arm, {}).get("jev")
        if not j:
            continue
        w(f"- `{arm}`: well-formed {f3(j['well_formed_share'])}; mean choice entropy {f3(j['mean_choice_entropy_bits'])} bits; mean confidence {f3(j['mean_confidence'])}; "
          f"chosen position counts {j['chosen_position_counts']}; chosen kind by turn {j['chosen_kind_by_t']}; "
          f"offer tags {a['by_arm'][arm]['chosen_tag_counts']}.")
        for name, t in j["by_gap_tercile"].items():
            w(f"  - {name}: n = {t['n']}, gap ∈ [{t['gap_range'][0]:.4f}, {t['gap_range'][1]:.4f}], ρ = {rho(t['spearman_entropy_vs_regret'])}, mean entropy {f3(t['mean_entropy_bits'])}, mean regret {f4(t['mean_regret'])}")
    w("")
    w("### Pairwise (cell-paired cluster bootstrap)")
    w("")
    w("| pair | Δ utility | Δ Σ decision regret | Δ EIG regret / decision | Δ questions | Δ agreement |")
    w("|---|---|---|---|---|---|")
    for key in ("jev_plain - random", "jev_eig - random", "jev_eig - jev_plain", "jev_plain - exact", "jev_eig - exact", "jev_plain - greedy_eig", "jev_plain - myopic"):
        pr = a["pairs"].get(key)
        if pr:
            w(f"| {key} | {ci(pr['learner_utility'])} | {ci(pr['total_decision_regret'])} | {ci(pr['eig_regret_per_decision_bits'])} | {ci(pr['questions_asked'])} | {ci(pr['agreement'])} |")
    w("")
    w("### Sanity gates")
    w("")
    g = v["gates"]
    w(f"- Well-formed answers: jev_plain {g['well_formed_jev_plain']['value']}, jev_eig {g['well_formed_jev_eig']['value']} → {'PASS' if g['well_formed_jev_plain']['pass'] and g['well_formed_jev_eig']['pass'] else 'FAIL'}.")
    rr = a["by_arm"]["random"]["mean_learner_utility"]
    sr = scripted_random
    half = (sr["ci95"][1] - sr["ci95"][0]) / 2
    w(f"- In-run `random` mean utility {f4(rr)} vs frozen scripted `random` {f4(sr['mean'])}: difference {rr - sr['mean']:+.4f}, scripted cell-bootstrap half-width {half:.4f} → {'PASS' if abs(rr - sr['mean']) <= half else 'FAIL'}.")
    for arm in ("jev_plain", "jev_eig"):
        pr = a["pairs"].get(f"{arm} - exact", {}).get("learner_utility")
        if pr and abs(pr["delta"]) < 1e-9 and abs(pr["ci95"][0]) < 1e-9 and abs(pr["ci95"][1]) < 1e-9:
            w(f"- `{arm}` − `exact` utility is exactly 0 in every cell: the arm matched the exact policy's realised utility everywhere (zero-regret share {f3(a['by_arm'][arm]['zero_regret_decision_share'])}); the registered S5 rule (CI < 0) therefore reads \"not supported\", which here means *equal to optimal*, not worse.")
    w(f"- Error stubs: {s['error_stubs']}. Deterministic arms in this run reproduce the frozen scripted values by construction (same code, zero noise).")
    w("")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--primary", required=True)
    p.add_argument("--secondary", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    L = []
    w = L.append
    w("# Binned Car Dealer, Condition B — Jev results (pre-registered)")
    w("")
    w("_Rendered by `render_jev_note.py` from frozen JSON; do not edit numbers by hand._")
    w("")
    w("Hypotheses, rules, seeds and gates were fixed in [`PREREGISTRATION_B.md`](PREREGISTRATION_B.md) before any live call; verdicts were applied by `verdicts_b.py` and frozen before being read. "
      "Scripted baselines: [`SCRIPTED_RESULTS.md`](SCRIPTED_RESULTS.md). Design: [`DESIGN_B.md`](DESIGN_B.md). "
      "One deviation from the registered command lines: the deterministic arms (`exact`, `greedy_eig`, `myopic`) were included in each run so every pair is in one file; they are identical to the frozen scripted arms by construction (zero noise, one play per cell). "
      "`jev-latest` is stochastic server-side; `--seed` fixes only the option shuffle and cell order. All Jev claims are **stochastic**; the deterministic arms are exact-on-logs.")
    w("")
    w("**Provenance note on S4.** The first analysis pass assigned Q-gap terciles by inclusive value bounds, which double-counted decisions whose gap tied at a boundary (most gaps are exactly 0). "
      "It was replaced by disjoint rank-based terciles before this note was written. The S4 verdict under the flawed binning was \"not supported\" for both arms at the primary horizon; the rule itself is unchanged.")
    w("")
    # headline, computed
    a1, v1, s1 = load(args.primary)
    a2, v2, s2 = load(args.secondary)
    w("## Headline (computed from the verdict files)")
    w("")
    for k in ("P1", "S1", "S2", "S3", "S5_jev_plain", "S5_jev_eig"):
        r = v1["rules"][k]
        w(f"- **{k}** ({r['quantity']}): {ci(r['estimate'])} → **{r['verdict']}**")
    for arm, pa in v1["rules"]["S4"]["per_arm"].items():
        w(f"- **S4** ({arm}): large-gap ρ {rho(pa['large_gap'])}, small-gap ρ {rho(pa['small_gap'])} → **{pa['pattern']}**")
        lg, sm = pa["large_gap"], pa["small_gap"]
        if lg and sm and lg.get("ci95") and sm.get("ci95"):
            if lg["ci95"][0] > 0 and sm["ci95"][1] < 0:
                w(f"  - Reading: the entropy–regret relation *reverses sign* with the gap for `{arm}`. When one option is clearly best, a diffuse choice goes with error; when the top options tie (gap = 0, i.e. several equally good actions), a diffuse choice goes with *lower* regret and the confident picks are the mistakes. The registered pattern asked for ρ ≈ 0 in the small-gap tercile, so it is \"not supported\" as written, but the document's underlying claim — that the relation changes with the gap — holds in a stronger form.")
    r = v2["rules"]["S6"]
    w(f"- **S6** ({r['quantity']}): {ci(r['estimate'])} → **{r['verdict']}**")
    for arm in ("jev_plain", "jev_eig"):
        w(f"- 6 rounds, `{arm}`: utility {f4(a2['by_arm'][arm]['mean_learner_utility'])} vs exact {f4(a2['by_arm']['exact']['mean_learner_utility'])}; zero-regret decisions {f3(a2['by_arm'][arm]['zero_regret_decision_share'])} of {a2['by_arm'][arm]['decisions']} (random: {f3(a2['by_arm']['random']['zero_regret_decision_share'])}).")
    w("")
    for title, d in (("Primary horizon — 2 rounds + final offer, easy", args.primary), ("Secondary horizon — 6 rounds + final offer, easy", args.secondary)):
        a, v, s = load(d)
        rounds = s["config"]["rounds"]
        sca = json.load(open(f"results/binned_b_scripted_r{rounds}_easy_20261026/analysis.json"))
        section(w, title, d, a, v, s, {"mean": sca["by_arm"]["random"]["mean_learner_utility"], "ci95": sca["vs_exact"]["random"]["learner_utility"]["ci95"]})
    w("## Regenerate the analysis (the live logs are not regenerable)")
    w("")
    w("```bash")
    for d, seed, hz in ((args.primary, 11, "primary"), (args.secondary, 12, "secondary")):
        w(f"PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.analyze_condition_b {d} --seed {seed} --B 5000 --json {d}/analysis.json")
        w(f"PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.verdicts_b {d}/analysis.json --horizon {hz}")
    w(f"PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_binned.render_jev_note --primary {args.primary} --secondary {args.secondary} --out llm_rl_scripts/car_dealer_binned/JEV_RESULTS.md")
    w("```")
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {args.out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
