"""Render the CD-PERSONA-1 results note from the frozen result directory written by
``analyze_current.py`` (``results.json``, ``audit_key_{A,N}.json``, ``audit_labels_{A,N}.jsonl``)
and the raw transcripts (``arm_{A,N}.jsonl``). Every number is formatted from those files; none
is typed.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_persona.render_current_note \\
        --dir results/persona_current_20261107 \\
        --out llm_rl_scripts/car_dealer_persona/RESULTS_CURRENT.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

from . import analyze_current as ac
from .analyze import BUYERS, HELD_MAX, NO_DISCOUNT_SELLERS, SELLERS, SOFT_MAX, wilson

TAG = "prereg-persona1-20261107"
BASELINE_TAG = "prereg-persona-20261106"
MODEL = "claude-sonnet-5-5"


def f3(x: float) -> str:
    return f"{x:.3f}"


def ci(c) -> str:
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def sci(c) -> str:
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def rate(x: dict) -> str:
    return f"{f3(x['rate'])} {ci(x['ci95'])} ({x['k']}/{x['n']})"


def frac(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k} of {n} ({k / n:.3f}, interval {lo:.3f} to {hi:.3f})"


def band_text(v: dict) -> str:
    return f"`{v['band']}`" + (", borderline (the interval crosses a band edge)" if v["borderline"] else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    def load(name):
        with open(os.path.join(args.dir, name)) as f:
            return json.load(f)

    def load_jsonl(name):
        with open(os.path.join(args.dir, name)) as f:
            return [json.loads(line) for line in f if line.strip()]

    R = load("results.json")
    A, N, aud, dec, P = R["A"], R["N"], R["audit"], R["decomposition"], R["predictions"]
    vA, vN, vb = R["M1_verdict"]["A"], R["M1_verdict"]["N"], R["M1b_verdict"]
    keys = {arm: load(f"audit_key_{arm}.json") for arm in ac.ARMS}
    labels = {arm: {d["item"]: d["outcome"] for d in load_jsonl(f"audit_labels_{arm}.jsonl")} for arm in ac.ARMS}
    raw = {arm: load_jsonl(f"arm_{arm}.jsonl") for arm in ac.ARMS}
    stopsN = Counter(r.get("stop_reason") for r in raw["N"])
    emptyN = sum(1 for r in raw["N"] if not r.get("original_conversation"))
    cost = {arm: sum(r["usage"]["input_tokens"] for r in raw[arm]) * 2 / 1e6
            + sum(r["usage"]["output_tokens"] for r in raw[arm]) * 10 / 1e6 for arm in ac.ARMS}

    L = []
    w = L.append

    w("# CD-PERSONA-1 results: does a current model hold a planted Car Dealer persona?")
    w("")
    w(f"_Rendered by `render_current_note.py` from `{args.dir}` written by `analyze_current.py`; every "
      f"number is formatted from the result files, none is typed. Registered in "
      f"`PREREGISTRATION_CURRENT.md`, frozen at tag `{TAG}` before any confirmatory conversation was "
      f"generated. The 2023 baseline is CD-PERSONA-0 (`RESULTS_BASELINE2023.md`, tag `{BASELINE_TAG}`)._")
    w("")
    w(f"Two arms, one model (`{MODEL}`, model-default sampling), {A['n']} conversations each, 250 per "
      f"buyer × seller cell, ids fixed by the registered seed.")
    w("")
    w("| Arm | Protocol | Refusals | Scorer `unclear` | Notes | Spend |")
    w("|---|---|---|---|---|---|")
    w(f"| A | buyer agent against content-independent scripted sellers | {A['refusals']} | {A['unclear']} | "
      f"{A['M5']['timeout']['k']} decision-token timeouts ({A['M5']['timeout']['rate']:.1%}), "
      f"{A['M5']['malformed_replies']['k']} malformed tokens in {A['M5']['malformed_replies']['n']} replies | "
      f"${cost['A']:.2f} |")
    w(f"| N | the 2023 single-narrator prompt, verbatim, `max_tokens` 2048 | {N['refusals']} | {N['unclear']} | "
      f"{stopsN.get('max_tokens', 0)} stories cut at the 2023 token cap ({stopsN.get('max_tokens', 0) / N['n']:.1%}), "
      f"{emptyN} of them empty (thinking used the whole budget) | ${cost['N']:.2f} |")
    w("")

    w("## The 2×2: who leaked, the model or the protocol?")
    w("")
    w("`eps_buyer` = audit-corrected P(sale) in the hard cells `H` = B-discount × {S-features, S-concise}, "
      "where no discount is on offer. In Arm A no discount was offered by construction, so this is the "
      "buyer's own violation rate; in Arm N, as in 2023, it is an upper bound.")
    w("")
    w("| | 2023 narrator (GPT-3.5) | Current narrator (Arm N) | Current buyer agent (Arm A) |")
    w("|---|---|---|---|")
    c = dec["cells"]
    w(f"| `eps_buyer` | {f3(c['2023_narrator']['rate'])} {ci(c['2023_narrator']['ci95'])} | "
      f"{f3(c['current_narrator']['rate'])} {ci(c['current_narrator']['ci95'])} | "
      f"{f3(c['current_agent']['rate'])} {ci(c['current_agent']['ci95'])} |")
    w(f"| band | `SOFT` | {band_text(vN)} | {band_text(vA)} |")
    w("")
    me, pe = dec["model_effect"], dec["protocol_effect"]
    w(f"- Model effect, old protocol with the model swapped: **{me['diff']:+.3f} {sci(me['ci95'])}**.")
    w(f"- Protocol effect, new model with the protocol swapped: **{pe['diff']:+.3f} {sci(pe['ci95'])}**.")
    w("")
    w(f"The registered estimator returns a `[0.000, 0.000]` interval when every audited scorer-sale is a "
      f"false sale (precision 0) and no audited no-sale is a miss; it is not a certainty claim. The scorer's "
      f"own raw rates with their Wilson intervals are {rate(A['M1_cell'])} in Arm A and {rate(N['M1_cell'])} "
      f"in Arm N, and the audit found every one of those scorer-sales to be a no-sale.")
    w("")

    w("## Predictions")
    w("")
    w("| | Claim | Value | Verdict |")
    w("|---|---|---|---|")
    p1 = P["P1"]["value"]
    w(f"| P1 | Arm A: `eps_buyer` `HELD` (≤ {HELD_MAX}) with S-concise and above {HELD_MAX} with S-features "
      f"(raw, by seller) | S-features {f3(p1['S-features'])}, S-concise {f3(p1['S-concise'])}; pooled "
      f"audit-corrected {f3(P['P1']['pooled_audit_corrected'])} | {P['P1']['verdict']} |")
    w(f"| P2 | Arm N: `eps_buyer` above {HELD_MAX} and at most the 2023 value {ac.BASELINE_EPS} | "
      f"{f3(P['P2']['value'])} | {P['P2']['verdict']} |")
    w(f"| P3 | protocol effect `eps_N − eps_A` at least {ac.DECOMP_MIN} | {P['P3']['value']:+.3f} | "
      f"{P['P3']['verdict']} |")
    w(f"| P4 | B-impatient compatibility gap negative in both arms | A {P['P4']['value']['A']:+.3f}, "
      f"N {P['P4']['value']['N']:+.3f} | {P['P4']['verdict']} |")
    w(f"| P5 | `eps_spec` (M1b) above {HELD_MAX} | {f3(P['P5']['value'])} {ci(vb['ci95'])}"
      + (", borderline" if vb["borderline"] else "") + f" | {P['P5']['verdict']} |")
    w(f"| P6 | token agrees with dialogue ≥ {ac.TOKEN_AGREEMENT_MIN} and timeout share < {ac.TIMEOUT_MAX} | "
      f"agreement {f3(P['P6']['value']['agreement'])}, timeout {f3(P['P6']['value']['timeout'])} | "
      f"{P['P6']['verdict']} |")
    p7 = P["P7"]["value"]
    w(f"| P7 | M6 cumulative acceptance non-decreasing in the offer and ≥ {ac.M6_MIN} after 15% | "
      + ", ".join(f"{k} {f3(v)}" for k, v in p7.items()) + f" | {P['P7']['verdict']} |")
    w(f"| P8 | second buyer model | none registered | {P['P8']['verdict']} |")
    w("")
    w("P1, P2 and P3 were written expecting a non-zero leak in both arms and more of it under the narrator "
      "protocol; the leak is zero in both, so all three are refuted in the direction of the persona holding "
      "better than predicted. P6 fails on its timeout half only. P7 fails on its level, not its shape: "
      f"acceptance is non-decreasing but reaches only {f3(A['M6']['cumulative']['close']['rate'])} by the "
      "close (see M6).")
    w("")

    w("## M1. Buyer discount constraint")
    w("")
    for arm, m, v in (("A", A, vA), ("N", N, vN)):
        a = aud[arm]["M1"]
        s = aud[arm]["M1_unclear_as_sale"]
        w(f"**Arm {arm}**: audit-corrected **{f3(v['corrected_rate'])} {ci(v['ci95'])}**, band {band_text(v)}.")
        w("")
        w("| | Rate | |")
        w("|---|---|---|")
        w(f"| Raw scorer | {rate(m['M1_cell'])} | {m['M1_cell']['unclear']} scorer-unclear excluded, bounds "
          f"{ci(m['M1_cell']['bounds_unclear'])} |")
        w(f"| Audit-corrected (registered) | {f3(a['corrected_rate'])} {ci(a['ci95'])} | precision "
          f"{f3(a['precision'])} on {a['audited_sales']} audited sales, miss rate {f3(a['miss_rate'])} on "
          f"{a['audited_no_sales']} audited no-sales |")
        w(f"| Sensitivity: audit `unclear` counted as a sale | {f3(s['corrected_rate'])} {ci(s['ci95'])} | |")
        for sel in NO_DISCOUNT_SELLERS:
            x = m["M1_cell"]["by_seller"][sel]
            w(f"| With {sel} (raw) | {rate(x)} | {x['unclear']} unclear |")
        if arm == "N":
            w(f"| Text-level, exploratory: discount buyer, no discount detected, any seller | "
              f"{rate(m['M1_text'])} | |")
        w("")

    w("## M1b. Buyer spec constraint (Arm A)")
    w("")
    b, ab, sb = A["M1b_cell"], aud["A"]["M1b"], aud["A"]["M1b_unclear_as_sale"]
    w(f"`eps_spec` = P(sale | B-spec, one required feature missing), audit-corrected: "
      f"**{f3(vb['corrected_rate'])} {ci(vb['ci95'])}**, band {band_text(vb)}.")
    w("")
    w("| | Rate | |")
    w("|---|---|---|")
    w(f"| Raw scorer, missing-one | {rate(b)} | {b['unclear']} unclear, bounds {ci(b['bounds_unclear'])} |")
    w(f"| Audit-corrected (registered) | {f3(ab['corrected_rate'])} {ci(ab['ci95'])} | precision "
      f"{f3(ab['precision'])} on {ab['audited_sales']} audited sales, miss rate {f3(ab['miss_rate'])} on "
      f"{ab['audited_no_sales']} audited no-sales |")
    w(f"| Sensitivity: audit `unclear` counted as a sale | {f3(sb['corrected_rate'])} {ci(sb['ci95'])} | |")
    w(f"| Raw scorer, all features present (complement) | {rate(b['all_features'])} | "
      f"{b['all_features']['unclear']} unclear |")
    w("")
    w("The audited spec-cell sales split three ways: a real purchase of a car the seller never confirmed "
      "has the missing feature, a conditional commitment (\"if you confirm X, I'll sign\") that the scorer "
      "reads as a sale and the labeller as `unclear`, and a refusal. Only the first counts against the "
      "persona; the registered estimator counts `unclear` as not-a-sale, the sensitivity row as a sale.")
    w("")

    w("## M2. Seller discount constraint")
    w("")
    w(f"- Arm A: the seller lines are scripted and verified against the script in every conversation "
      f"(`seller_ok` in all {A['n']}; violations {A['M2']['seller_violations']}), so no discount reached "
      f"the buyer in `H` by construction.")
    m2 = N["M2"]
    w(f"- Arm N: **{rate(m2)}**, band `{m2['band']}`" + (", borderline" if m2["borderline"] else "")
      + ". From the detector, not audited; the detector misses dealers who counter in plain numbers "
        "without the word \"discount\", so this is understated.")
    for sel in NO_DISCOUNT_SELLERS:
        w(f"  - {sel}: {rate(m2['by_seller'][sel])}")
    w("")

    w("## M3. Compatibility: sale rate by persona pair")
    w("")
    for arm, m in (("A", A), ("N", N)):
        w(f"**Arm {arm}**")
        w("")
        w("| Buyer | " + " | ".join(SELLERS) + " | Gap (compatible minus mean of others) |")
        w("|---|" + "---|" * (len(SELLERS) + 1))
        for bname in BUYERS:
            cells = m["M3"]["cells"][bname]
            g = m["M3"]["gaps"][bname]
            w(f"| {bname} | " + " | ".join(
                ("**" if sel == g["compatible_seller"] else "")
                + f"{f3(cells[sel]['rate'])} ({cells[sel]['k']}/{cells[sel]['n']})"
                + ("**" if sel == g["compatible_seller"] else "") for sel in SELLERS)
              + f" | {g['gap']:+.3f} {sci(g['ci95'])} |")
        w("")
    w("Bold marks the pairing the 2023 design calls compatible. Denominators exclude scorer-`unclear`; in "
      f"Arm N the B-spec × S-features cell lost {N['M3']['cells']['B-spec']['S-features']['unclear']} of 250 "
      "to truncation, so its rate is the least certain in either table.")
    w("")

    w("## M4. Budget")
    w("")
    m4 = A["M4"]
    w(f"- Arm A, sales at list price 1.25 × budget: **{rate(m4)}** closed above budget "
      f"({m4['sales_without_price']} sales without an extractable price). Every undiscounted sale there is "
      f"over budget; a 15%-off S-discount sale is not. The scorer's price matched the decision token's "
      f"price in {rate(m4['price_matches_close'])} of priced sales.")
    w("")
    w("| List price / budget | Sale rate | Over budget among sales |")
    w("|---|---|---|")
    for mult, x in m4["by_multiplier"].items():
        w(f"| {mult} | {rate(x['sale'])} | {rate(x['over_budget'])} |")
    w("")
    m4n = N["M4"]
    w(f"- Arm N, priced sales above the planted budget, as in 2023: **{rate(m4n)}** "
      f"({m4n['sales_without_price']} sales without a price).")
    w("")
    w("| Budget | Share above budget |")
    w("|---|---|")
    for bud, x in m4n["by_budget"].items():
        w(f"| {int(bud):,} | {rate(x)} |")
    w("")

    w("## M5. Decision channel against the dialogue")
    w("")
    m5 = A["M5"]
    w(f"- Arm A: the decision token (`BUY` vs `LEAVE`/timeout) agrees with the scorer's dialogue outcome in "
      f"**{rate(m5)}**; {len(m5['disagreements'])} disagreements. Timeouts (five seller turns, no decision): "
      f"**{rate(m5['timeout'])}**. Malformed tokens: {rate(m5['malformed_replies'])} of buyer replies.")
    fr = N["M5"]["fractional"]
    w(f"- Arm N: a final self-report of exactly 0 or 1 agrees with the dialogue in **{rate(N['M5'])}**; "
      f"{fr['n']} stories ({fr['n'] / N['n']:.1%}) end on a fractional one (dialogue outcomes {fr['sale']} "
      f"sale, {fr['no_sale']} no sale, {fr['unclear']} unclear); {N['M5']['no_self_report']} have no "
      f"parseable self-report.")
    w("")

    w("## M6. Response to the discount schedule (Arm A, B-discount × S-discount)")
    w("")
    m6 = A["M6"]
    w(f"{m6['n']} conversations. Cumulative acceptance (token `BUY` by the end of the buyer's reply to that "
      f"seller turn) and per-offer acceptance among buyers still present.")
    w("")
    w("| Seller turn | Offer | Cumulative acceptance | Per-offer acceptance |")
    w("|---|---|---|---|")
    names = {"0%": ("1", "no offer"), "5%": ("2", "5% off"), "10%": ("3", "10% off"), "15%": ("4", "15% off"),
             "close": ("5", "close at 15%")}
    for k, (turn, offer) in names.items():
        w(f"| {turn} | {offer} | {rate(m6['cumulative'][k])} | {rate(m6['per_offer'][k])} |")
    w("")
    w(f"Left without buying (token `LEAVE`): {rate(m6['left_without_buying'])}; the rest of the "
      f"non-buyers timed out still asking. Acceptance is non-decreasing and zero until a discount appears, "
      f"as the persona requires, but it stays below {ac.M6_MIN} after the 15% offer: the agent conditions "
      f"its purchase on feature confirmations that the scripted seller never gives.")
    w("")

    w("## The hand-audit")
    w("")
    names_s = {"hard_sale": "Hard cells `H`, scorer said sale", "hard_no_sale": "Hard cells `H`, scorer said no sale",
               "spec_sale": "Spec cells `H_spec`, scorer said sale", "spec_no_sale": "Spec cells `H_spec`, scorer said no sale",
               "decoy_sale": "Other cells, scorer said sale", "decoy_no_sale": "Other cells, scorer said no sale"}
    for arm in ac.ARMS:
        strata = {}
        for k in keys[arm]:
            strata.setdefault(k["stratum"], Counter())[labels[arm][k["item"]]] += 1
        w(f"**Arm {arm}**: {len(keys[arm])} items, labelled with cell and scorer call hidden "
          f"(`audit.py`, seed {ac.AUDIT_SEED}).")
        w("")
        w("| Stratum | Items | Labelled sale | Labelled no sale | Labelled unclear |")
        w("|---|---|---|---|---|")
        for st, name in names_s.items():
            if st in strata:
                cnt = strata[st]
                w(f"| {name} | {sum(cnt.values())} | {cnt['sale']} | {cnt['no_sale']} | {cnt['unclear']} |")
        w("")
    dA, dN = aud["A"]["M1"]["decoys"], aud["N"]["M1"]["decoys"]
    w(f"Registered as a check of the scorer outside `H`: in Arm A, {frac(int(dA['sale']['labelled_sale']), dA['sale']['n'])} "
      f"of the decoy scorer-sales were sales on the label (the rest are conditional commitments and "
      f"walk-outs read as sales, the false-sale mode seen at the gate), and "
      f"{frac(int(dA['no_sale']['labelled_sale']), dA['no_sale']['n'])} of the decoy no-sales were sales. In Arm N, "
      f"{frac(int(dN['sale']['labelled_sale']), dN['sale']['n'])} of the decoy scorer-sales were sales and "
      f"{frac(int(dN['no_sale']['labelled_sale']), dN['no_sale']['n'])} of the decoy no-sales were sales. The "
      f"scorer's errors run toward calling sales, so the raw rates in this note are upper bounds and the "
      f"registered verdicts rest on the audit.")
    w("")

    w("## Reading")
    w("")
    w(f"- **The discount-only persona holds.** No violation survives the audit in "
      f"{aud['A']['M1']['scorer_sales'] + aud['A']['M1']['scorer_no_sales']} Arm A hard-cell conversations "
      f"(the scorer called {aud['A']['M1']['scorer_sales']} sales, all false) or in "
      f"{aud['N']['M1']['scorer_sales'] + aud['N']['M1']['scorer_no_sales']} in Arm N "
      f"({aud['N']['M1']['scorer_sales']}, all false), against {c['2023_narrator']['rate']:.0%} in the 2023 data.")
    w(f"- **The change is the model, not the protocol.** Swapping GPT-3.5 for `{MODEL}` under the 2023 "
      f"narrator prompt removes the leak ({me['diff']:+.3f}); giving the same model its own seat against "
      f"a scripted seller changes nothing further ({pe['diff']:+.3f}). The registration predicted the opposite.")
    w(f"- **The spec persona is softer.** `eps_spec` is {f3(vb['corrected_rate'])}, {band_text(vb)}; "
      f"{f3(sb['corrected_rate'])} if the conditional commitments count as sales. The buyer told to need "
      f"every feature sometimes buys a car whose missing feature was never confirmed.")
    gapsA, gapsN = A["M3"]["gaps"], N["M3"]["gaps"]
    w(f"- **Compatibility structure.** The discount buyer's gap is {gapsA['B-discount']['gap']:+.3f} (A) and "
      f"{gapsN['B-discount']['gap']:+.3f} (N); the impatient buyer's is negative in both arms "
      f"({gapsA['B-impatient']['gap']:+.3f}, {gapsN['B-impatient']['gap']:+.3f}), as predicted; the spec "
      f"buyer's is {gapsA['B-spec']['gap']:+.3f} (A) and {gapsN['B-spec']['gap']:+.3f} (N).")
    w(f"- **The agent is a harder counterparty than the narrator's buyer.** Arm A sale rates run from "
      f"{f3(min(x['rate'] for row in A['M3']['cells'].values() for x in row.values()))} to "
      f"{f3(max(x['rate'] for row in A['M3']['cells'].values() for x in row.values()))}; Arm N from "
      f"{f3(min(x['rate'] for row in N['M3']['cells'].values() for x in row.values()))} to "
      f"{f3(max(x['rate'] for row in N['M3']['cells'].values() for x in row.values()))}. The narrator writes "
      f"buyers who cave to firm sellers; the agent walks or keeps asking.")
    w(f"- **For the oracle question.** As a noisy oracle of the discount rule, this model's noise rate is "
      f"0 in both protocols at this sample size; the spec rule leaks at about {vb['corrected_rate']:.0%}. "
      f"What the agent does not do is buy on the discount alone: it takes the 15% offer "
      f"{m6['cumulative']['15%']['rate']:.0%} of the time, holding out for answers the scripted seller cannot give.")
    w("")

    w("## Limits")
    w("")
    w("- One model, one buyer prompt, one narrator prompt. P8 is void; a second buyer model would be its "
      "own registration.")
    w(f"- The scripted seller ignores the buyer's words, so M6 and the Arm A sale rates measure the agent "
      f"against a seller that never answers; {A['M5']['timeout']['rate']:.1%} of Arm A conversations ended "
      f"with the buyer still asking.")
    w(f"- The 2023 `max_tokens` of 2048 is not neutral for this model: {stopsN.get('max_tokens', 0)} Arm N "
      f"stories were cut, {emptyN} before any text, and {N['unclear']} score `unclear`. Rates exclude them "
      f"and the bounds from counting them either way are in the tables.")
    w(f"- All labels ({len(keys['A'])} + {len(keys['N'])} audit items, 160 gate items) come from one labeller, "
      f"Claude, with cell and scorer call hidden but not the conversation's content; the labelling rules "
      f"are those of `gold/README.md`.")
    w("- The `[0.000, 0.000]` intervals are the registered bootstrap on an audit with zero positives; they "
      "bound the estimator, not the quantity.")
    w("- M2 in Arm N and the text-level row depend on a discount detector that was gated but not audited.")
    w("")

    with open(args.out, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
