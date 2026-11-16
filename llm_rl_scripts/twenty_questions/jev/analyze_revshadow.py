"""Score the reverse-shadow study (``STRESS_REVSHADOW_PREREG.md``): RS0-RS4.

Reads the per-job records written by ``stress_revshadow`` and writes ``analysis.json``
and ``fig_revshadow.png`` into the run directory.

    PYTHONHASHSEED=0 PYTHONPATH=. python -m \
        llm_rl_scripts.twenty_questions.jev.analyze_revshadow --run-dir <out>

Counterfactual outcomes per game (each exact for that game; none sees what a
changed admission would do to later games -- the two tail brackets bound that):

* ``committed``: the committed outcome, with its budget-end answers split into
  wrong and right-unconfirmed (the check that the references reproduce).
* ``base``: `soft` under the committed ``top`` rule. At the budget end it answers its
  leading known unless UNKNOWN leads. An unconfirmed answer is *right-unconfirmed*
  when that known is the secret and *wrong* otherwise. In the ``always`` regime this
  re-applies ``top`` to the escalated budget-end games.
* ``veto`` (G-end): as ``base``, but a budget-end answer becomes ``expert`` when the
  hard view had struck the leading known.
* ``alarm`` (G-alarm): ``expert`` whenever the hard set emptied during the game
  (the observer sees only non-final answers, so the alarm always precedes the end).
* ``both`` (G-both): the alarm, then the veto.
* ``veto1`` / ``alarm1`` / ``both1``: the tolerance-1 variant (struck on two
  contradicting answers). Reported only.

Rates: *wrong* = unconfirmed answers that are not the secret, over all games; *FPR* =
in-pool games ending with the expert, over in-pool games (the end-game study's
definitions; there wrong also counted right-unconfirmed answers, reported here as
``wrong_incl_right_unconf``).
"""
import argparse
import glob
import gzip
import json
import os
from collections import defaultdict
from typing import Dict, List

WON, EXPERT, WRONG, FALSE_CONFIRM = 0, 1, 2, 3
BUDGET = 20
N_JOBS = 219
RS1_MIN = 0.5
RS2_MIN = 0.10
RS3_WRONG = 0.05
RS3_FPR = 0.20
RS4_MIN = 0.6
TAIL_SIGMA = 0.75
GATES = ("committed", "base", "veto", "alarm", "both", "veto1", "alarm1", "both1")
# outcome labels for the counterfactuals
O_WON, O_EXPERT, O_WRONG, O_RIGHT, O_FC = "won", "expert", "wrong", "right_unconf", "false_confirm"


def load(run_dir: str) -> List[dict]:
    recs = []
    for p in sorted(glob.glob(os.path.join(run_dir, "jobs", "*.json.gz"))):
        with gzip.open(p, "rt") as f:
            r = json.load(f)
        r["id"] = os.path.basename(p)[:-len(".json.gz")]
        recs.append(r)
    return recs


def cell(rec: dict) -> dict:
    j = rec["job"]
    c = {"src": rec["src"], "flip": j["flip"]}
    if rec["src"] == "st2":
        c["d"] = j["d"]
    else:
        c["sigma"] = j["sigma"]
    return c


def games(rec: dict):
    g = rec["games"]
    for i in range(len(g["obj"])):
        yield {k: g[k][i] for k in g}


def outcomes(r: dict) -> Dict[str, str]:
    """The counterfactual outcome of one game under each gate."""
    o, out = r["obj"], r["outcome"]
    answered = r["rh_u_leads"] == 0              # budget end, a known led UNKNOWN
    if answered:
        base = O_RIGHT if r["rh_end_top"] == o else O_WRONG
    else:
        base = {WON: O_WON, EXPERT: O_EXPERT, FALSE_CONFIRM: O_FC}.get(out)
        if base is None:
            raise AssertionError("a committed wrong must be a budget-end answer")
    veto = O_EXPERT if (answered and r["rh_end_struck"] == 1) else base
    veto1 = O_EXPERT if (answered and r["rh_end_contra"] >= 2) else base
    alarm = O_EXPERT if r["rh_alarm"] >= 0 else base
    alarm1 = O_EXPERT if r["rh_alarm1"] >= 0 else base
    committed = base if out == WRONG else {WON: O_WON, EXPERT: O_EXPERT,
                                           FALSE_CONFIRM: O_FC}[out]
    return {"committed": committed, "base": base, "veto": veto, "alarm": alarm,
            "both": O_EXPERT if r["rh_alarm"] >= 0 else veto,
            "veto1": veto1, "alarm1": alarm1,
            "both1": O_EXPERT if r["rh_alarm1"] >= 0 else veto1}


def _share(num: int, den: int) -> float:
    return num / den if den else float("nan")


class Rates:
    def __init__(self):
        self.n = 0
        self.n_in = 0
        self.c: Dict[str, Dict[str, int]] = {gt: defaultdict(int) for gt in GATES}

    def add(self, r: dict) -> None:
        self.n += 1
        self.n_in += bool(r["in_pool"])
        for gt, oc in outcomes(r).items():
            self.c[gt][oc] += 1
            if r["in_pool"] and oc == O_EXPERT:
                self.c[gt]["in_pool_expert"] += 1

    def out(self) -> dict:
        res = {"games": self.n, "in_pool": self.n_in}
        for gt in GATES:
            c = self.c[gt]
            res[gt] = {"wrong": _share(c[O_WRONG], self.n),
                       "fpr": _share(c["in_pool_expert"], self.n_in),
                       "won": _share(c[O_WON], self.n),
                       "right_unconf": _share(c[O_RIGHT], self.n),
                       "expert": _share(c[O_EXPERT], self.n),
                       "wrong_incl_right_unconf": _share(c[O_WRONG] + c[O_RIGHT], self.n)}
        return res


class Tally:
    """Counts for one cell (or one scope)."""

    def __init__(self):
        self.c = defaultdict(int)
        self.rates = Rates()
        self.saved: List[int] = []           # questions an alarm escalation saves
        self.surv_wrong = defaultdict(int)   # |S| at the end of soft's wrong answers

    def add(self, r: dict) -> None:
        c = self.c
        o, ip = r["obj"], r["in_pool"]
        self.rates.add(r)
        c["games"] += 1
        c["novel"] += not ip
        answered = r["rh_u_leads"] == 0
        if answered:
            right = r["rh_end_top"] == o
            if right:
                c["ans_right"] += 1
                c["ans_right_struck"] += r["rh_end_struck"] == 1
            else:
                c["ans_wrong"] += 1
                c["ans_wrong_struck"] += r["rh_end_struck"] == 1
                c["ans_wrong_struck1"] += r["rh_end_contra"] >= 2
                s = r["rh_surv_end"]
                self.surv_wrong["0" if s == 0 else "1" if s == 1 else "2+"] += 1
        if r["rh_alarm"] >= 0:
            c["alarm"] += 1
            self.saved.append(r["n_questions"] - r["rh_alarm"])
        if not ip:
            c["nov_alarm"] += r["rh_alarm"] >= 0
            soft_exp = r["outcome"] == EXPERT and not answered
            c["nov_soft_expert"] += soft_exp
            c["nov_not_soft_expert"] += not soft_exp
            c["nov_alarm_not_soft_expert"] += (not soft_exp) and r["rh_alarm"] >= 0
            c["nov_alarm_or_soft_expert"] += soft_exp or r["rh_alarm"] >= 0
        if ip and r["outcome"] == WON:
            c["ip_won"] += 1
            c["ip_won_alarm"] += r["rh_alarm"] >= 0
            c["ip_won_alarm1"] += r["rh_alarm1"] >= 0

    def out(self) -> dict:
        c = self.c
        sv = sorted(self.saved)
        return {
            "games": c["games"],
            "veto_catch": {"n": c["ans_wrong"], "share": _share(c["ans_wrong_struck"], c["ans_wrong"]),
                           "share_tol1": _share(c["ans_wrong_struck1"], c["ans_wrong"])},
            "veto_gives_up_right": {"n": c["ans_right"],
                                    "share": _share(c["ans_right_struck"], c["ans_right"])},
            "alarm_before_win": {"n": c["ip_won"], "share": _share(c["ip_won_alarm"], c["ip_won"]),
                                 "share_tol1": _share(c["ip_won_alarm1"], c["ip_won"])},
            "novel_alarm": {"n": c["novel"], "share": _share(c["nov_alarm"], c["novel"]),
                            "soft_expert": _share(c["nov_soft_expert"], c["novel"]),
                            "alarm_or_soft_expert": _share(c["nov_alarm_or_soft_expert"], c["novel"]),
                            "alarm_given_soft_did_not_escalate": _share(
                                c["nov_alarm_not_soft_expert"], c["nov_not_soft_expert"])},
            "alarm_questions_saved": ({"n": len(sv), "median": sv[len(sv) // 2],
                                       "q25": sv[len(sv) // 4], "q75": sv[(3 * len(sv)) // 4]}
                                      if sv else None),
            "surv_end_in_wrong": dict(self.surv_wrong),
            "rates": self.rates.out(),
        }


def analyze(run_dir: str, make_figures: bool = True) -> dict:
    recs = load(run_dir)
    out: Dict[str, object] = {"run_dir": run_dir, "n_jobs": len(recs)}
    # ---- RS0: mechanics -------------------------------------------------------
    not_ident = [r["id"] for r in recs if r["identical"] is not True]
    bad_rederive = [r["id"] for r in recs if r["rederive_failures"] != 0 or r["rederived"] <= 0]
    c_bad = 0
    c_n = 0
    for r in recs:
        if r["job"]["flip"] != 0.0:
            continue
        for g in games(r):
            if g["in_pool"]:
                c_n += 1
                c_bad += (g["rh_secret_contra"] != 0 or g["rh_alarm"] >= 0
                          or (g["rh_end_struck"] == 1 and g["rh_end_top"] == g["obj"]))
    rs0 = {"a_identical": not not_ident and len(recs) == N_JOBS,
           "b_rederived": not bad_rederive,
           "c_secret_never_struck_at_flip0": c_bad == 0 and c_n > 0,
           "n_jobs": len(recs), "not_identical": not_ident[:20],
           "rederive_bad": bad_rederive[:20], "flip0_in_pool_games": c_n,
           "flip0_in_pool_violations": c_bad,
           "games_rederived": sum(r["rederived"] for r in recs)}
    rs0["pass"] = rs0["a_identical"] and rs0["b_rederived"] and rs0["c_secret_never_struck_at_flip0"]
    out["RS0"] = rs0
    # ---- cells and scopes -----------------------------------------------------
    cells: Dict[str, Tally] = defaultdict(Tally)
    scopes: Dict[str, Tally] = defaultdict(Tally)
    for r in recs:
        c = cell(r)
        if c["src"] == "st2":
            key = "st2_d{}_f{}".format(c["d"], c["flip"])
        else:
            key = "{}_s{}_f{}".format(c["src"], c["sigma"], c["flip"])
        tail = c["src"] != "st2" and c["sigma"] == TAIL_SIGMA and c["flip"] == 0.0
        for g in games(r):
            cells[key].add(g)
            if tail:
                scopes["tail_" + c["src"]].add(g)
            if c["src"] == "st2":
                if c["d"] <= 2:
                    scopes["st2_d_le_2"].add(g)
                if c["flip"] > 0:
                    scopes["st2_noisy"].add(g)
                    scopes["st2_noisy_f{}".format(c["flip"])].add(g)
                scopes["st2_all"].add(g)
    sc = {k: v.out() for k, v in scopes.items()}
    out["scopes"] = sc
    out["cells"] = {k: cells[k].out() for k in sorted(cells)}

    def get(scope, *path):
        x = sc.get(scope)
        for p in path:
            if x is None:
                return float("nan")
            x = x.get(p)
        return float("nan") if x is None else x

    def ge(v, t):
        return v == v and v >= t

    def le(v, t):
        return v == v and v <= t
    # ---- RS1: the veto catches soft's wrong answers ---------------------------
    rs1 = {s: {"share": get(s, "veto_catch", "share"), "n": get(s, "veto_catch", "n")}
           for s in ("tail_st3", "tail_end", "st2_d_le_2")}
    for s in rs1:
        rs1[s]["pass"] = ge(rs1[s]["share"], RS1_MIN)
    rs1["pass"] = all(rs1[s]["pass"] for s in ("tail_st3", "tail_end", "st2_d_le_2"))
    out["RS1"] = rs1
    # ---- RS2: the raw alarm is too costly under noise --------------------------
    v = get("st2_noisy_f0.1", "alarm_before_win", "share")
    out["RS2"] = {"share_f0.10": v, "n": get("st2_noisy_f0.1", "alarm_before_win", "n"),
                  "share_f0.05": get("st2_noisy_f0.05", "alarm_before_win", "share"),
                  "pass": ge(v, RS2_MIN)}
    # ---- RS3: on the tail, soft + veto is autonomous and safe ------------------
    rs3 = {}
    for s in ("tail_st3", "tail_end"):
        w = get(s, "rates", "veto", "wrong")
        f = get(s, "rates", "veto", "fpr")
        rs3[s] = {"wrong": w, "fpr": f, "pass": le(w, RS3_WRONG) and le(f, RS3_FPR),
                  "base": get(s, "rates", "base"), "committed": get(s, "rates", "committed")}
    rs3["pass"] = rs3["tail_st3"]["pass"] and rs3["tail_end"]["pass"]
    rs3["one_bracket"] = rs3["tail_st3"]["pass"] != rs3["tail_end"]["pass"]
    out["RS3"] = rs3
    # ---- RS4: the alarm sees novel objects -------------------------------------
    v = get("tail_st3", "novel_alarm", "share")
    out["RS4"] = {"share": v, "n": get("tail_st3", "novel_alarm", "n"), "pass": ge(v, RS4_MIN),
                  "reported": get("tail_st3", "novel_alarm")}
    # ---- decision ---------------------------------------------------------------
    build = out["RS1"]["pass"] and out["RS3"]["pass"]
    one = out["RS1"]["pass"] and out["RS3"]["one_bracket"]
    out["follow_on"] = {
        "build_played_arm": bool(build or one),
        "rs3_one_bracket_only": bool(one),
        "include_alarm": bool((build or one) and out["RS4"]["pass"]),
        "alarm_behind_reinstate": bool((build or one) and out["RS4"]["pass"] and out["RS2"]["pass"]),
    }
    with open(os.path.join(run_dir, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    if make_figures:
        figure(out, run_dir)
    return out


def figure(a: dict, run_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for src, mk, lab in (("st3", "o", "ST-3 bracket"), ("end", "s", "always bracket")):
        xs, ys = [], []
        for s in (0.0, 0.25, 0.5, 0.75):
            k = "{}_s{}_f0.0".format(src, s)
            if k in a["cells"]:
                xs.append(s)
                ys.append(a["cells"][k]["veto_catch"]["share"])
        ax.plot(xs, ys, marker=mk, label=lab)
    ax.axhline(RS1_MIN, color="grey", ls=":", lw=1)
    ax.set_xlabel("tail sigma (flip 0)")
    ax.set_ylabel("share of soft's wrong answers the veto struck")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("RS1: veto catch rate")
    ax.legend(fontsize=8)
    ax = axes[1]
    for s, col in (("tail_st3", "C0"), ("tail_end", "C1")):
        sc = a["scopes"].get(s)
        if not sc:
            continue
        for gt, mk in (("base", "o"), ("veto", "s"), ("alarm", "^"), ("both", "D")):
            r = sc["rates"][gt]
            ax.scatter(r["fpr"], r["wrong"], color=col, marker=mk, s=40,
                       label="{} {}".format(s.replace("tail_", ""), gt))
    ax.axhline(RS3_WRONG, color="grey", ls=":", lw=1)
    ax.axvline(RS3_FPR, color="grey", ls=":", lw=1)
    ax.set_xlabel("FPR (in-pool games sent to the expert)")
    ax.set_ylabel("wrong (unconfirmed answer, not the secret)")
    ax.set_title("RS3: tail sigma 0.75, flip 0")
    ax.legend(fontsize=7, ncol=2)
    fig.text(0.01, 0.005, "Source: analyze_revshadow.py on results/{} (analysis.json). "
             "Per-game counterfactuals; the two tail brackets bound admission "
             "feedback.".format(os.path.basename(os.path.normpath(run_dir))),
             fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(run_dir, "fig_revshadow.png"), dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    res = analyze(a.run_dir, not a.no_figures)
    print("RS0 {}  RS1 {}  RS2 {}  RS3 {}  RS4 {}".format(
        *("PASS" if res[k]["pass"] else "FAIL" for k in ("RS0", "RS1", "RS2", "RS3", "RS4"))))
    print("follow-on: {}".format(res["follow_on"]))


if __name__ == "__main__":
    main()
