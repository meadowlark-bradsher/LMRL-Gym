"""Score the shadow-mode study (``STRESS_SHADOW_PREREG.md``): SH0-SH5.

Reads the per-job records written by ``stress_shadow`` and writes ``analysis.json``
and ``fig_shadow.png`` into the run directory.

    PYTHONHASHSEED=0 PYTHONPATH=. python -m \
        llm_rl_scripts.twenty_questions.jev.analyze_shadow --run-dir <out>
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
N_JOBS = 255
SH1_MIN = 0.5
SH2_MIN = 0.7
SH3_MIN = 0.5
SH4_MAX = 0.3
SH5_MIN = 0.10
N_REAL = 158


def load(run_dir: str) -> List[dict]:
    recs = []
    for p in sorted(glob.glob(os.path.join(run_dir, "jobs", "*.json.gz"))):
        with gzip.open(p, "rt") as f:
            r = json.load(f)
        r["id"] = os.path.basename(p)[:-len(".json.gz")]
        recs.append(r)
    return recs


def games(rec: dict):
    """Per game: a dict with the committed outcome columns, the shadow columns, and
    whether the secret's parent (ST-2) was already known when the game started."""
    g = rec["games"]
    pidx = rec.get("parent_idx")
    known = set()
    for i in range(len(g["obj"])):
        o = g["obj"][i]
        par = pidx[o] if pidx is not None else -1
        row = {k: g[k][i] for k in g}
        row["parent"] = par
        row["parent_known"] = par >= 0 and par in known
        yield row
        if g["outcome"][i] == EXPERT and not g["in_pool"][i]:
            known.add(o)


def _share(num: int, den: int) -> float:
    return num / den if den else float("nan")


class Tally:
    def __init__(self):
        self.c = defaultdict(int)
        self.first_turns: List[int] = []
        self.rev_leader = 0
        self.rev_revived = 0
        self.rev_n = 0

    def add(self, r: dict) -> None:
        c = self.c
        o, ip, out = r["obj"], r["in_pool"], r["outcome"]
        dis = r["sh_first_disagree"] >= 0
        c["games"] += 1
        c["disagree_games"] += dis
        c["disagree_turns"] += r["sh_disagree"]
        if dis:
            self.first_turns.append(r["sh_first_disagree"])
        if ip:
            c["in_pool"] += 1
            if dis:
                c["ip_dis"] += 1
                c["ip_dis_lead_secret"] += r["sh_leader_first"] == o
            if out in (EXPERT, WRONG):                 # SH3/SH4 scope
                c["ip_lost"] += 1
                c["ip_lost_lead_secret"] += r["sh_leader_dec"] == o
            if out == EXPERT:
                c["ip_expert"] += 1
                c["ip_expert_lead_secret"] += r["sh_leader_dec"] == o
        else:
            c["novel"] += 1
            if out == EXPERT:
                c["nov_expert"] += 1
                # cost side: a soft-chosen test would spend a turn, then escalate anyway
                c["nov_expert_struck_turnleft"] += (r["sh_struck_dec"] == 1
                                                    and r["sh_turns_left_dec"] >= 1)
            if o >= N_REAL and r["parent"] >= 0:      # an ST-2 look-alike
                c["la_novel"] += 1
                if r["parent_known"]:
                    # SH2: P0a tests the parent by name no later than the first
                    # disagreeing turn (or at all, when no turn disagrees).
                    pt = [t for t, ob in r["sh_guesses"] if ob == r["parent"]]
                    c["la_pk"] += 1
                    c["la_pk_parent_guessed"] += bool(pt)
                    c["la_pk_tested_first"] += bool(pt) and (
                        not dis or min(pt) <= r["sh_first_disagree"])
                if r["parent_known"] and dis:
                    c["la_pk_dis"] += 1
                    c["la_pk_dis_lead_parent"] += r["sh_leader_first"] == r["parent"]
                    c["la_pk_dis_lead_ref"] += r["sh_leader_first"] == r["sh_ref"]
                    c["la_pk_dis_ref_is_parent"] += r["sh_ref"] == r["parent"]
        if out == WRONG:
            c["wrong"] += 1
            c["wrong_lead_secret"] += r["sh_leader_dec"] == o
        for rev, lead in r["sh_revivals"]:
            self.rev_n += 1
            self.rev_revived += rev == o
            self.rev_leader += lead == o

    def metrics(self) -> dict:
        c = self.c
        ft = sorted(self.first_turns)

        def q(p):
            return ft[min(len(ft) - 1, int(p * len(ft)))] if ft else -1
        return {
            "games": c["games"],
            "disagree_share": _share(c["disagree_games"], c["games"]),
            "disagree_turns_per_game": _share(c["disagree_turns"], c["games"]),
            "first_disagree_turn_q25_50_75": [q(0.25), q(0.5), q(0.75)],
            "sh1_inpool_disagree_leader_is_secret": _share(c["ip_dis_lead_secret"], c["ip_dis"]),
            "sh1_n": c["ip_dis"],
            "sh2_parent_tested_before_disagreement": _share(c["la_pk_tested_first"],
                                                            c["la_pk"]),
            "sh2_n": c["la_pk"],
            "lookalike_parent_guessed": _share(c["la_pk_parent_guessed"], c["la_pk"]),
            "lookalike_disagree_leader_is_parent": _share(c["la_pk_dis_lead_parent"],
                                                          c["la_pk_dis"]),
            "lookalike_disagree_leader_is_ref": _share(c["la_pk_dis_lead_ref"], c["la_pk_dis"]),
            "lookalike_disagree_ref_is_parent": _share(c["la_pk_dis_ref_is_parent"],
                                                       c["la_pk_dis"]),
            "lookalike_disagree_n": c["la_pk_dis"],
            "sh3_inpool_lost_leader_is_secret": _share(c["ip_lost_lead_secret"], c["ip_lost"]),
            "sh3_n": c["ip_lost"],
            "inpool_expert_leader_is_secret": _share(c["ip_expert_lead_secret"], c["ip_expert"]),
            "inpool_expert_n": c["ip_expert"],
            "fpr": _share(c["ip_expert"], c["in_pool"]),
            "novel_expert_struck_turn_left": _share(c["nov_expert_struck_turnleft"],
                                                    c["nov_expert"]),
            "wrong_rate": _share(c["wrong"], c["games"]),
            "wrong_leader_is_secret": _share(c["wrong_lead_secret"], c["wrong"]),
            "revivals": self.rev_n,
            "revival_leader_is_secret": _share(self.rev_leader, self.rev_n),
            "revival_revived_is_secret": _share(self.rev_revived, self.rev_n),
        }


def key(rec: dict) -> tuple:
    j = rec["job"]
    if rec["src"] == "st2":
        return ("st2", j["d"], j["flip"])
    arm = "norepeat" if j["no_repeat"] else "primary"
    return (rec["src"], j["sigma"], j["flip"], arm)


def analyze(run_dir: str) -> dict:
    recs = load(run_dir)
    cells: Dict[tuple, Tally] = defaultdict(Tally)
    groups: Dict[str, Tally] = defaultdict(Tally)
    sh0c_hits = 0
    for rec in recs:
        k = key(rec)
        j = rec["job"]
        for r in games(rec):
            cells[k].add(r)
            if rec["src"] == "st2":
                if j["flip"] > 0:
                    groups["st2_noisy"].add(r)
                    groups["st2_f{:g}".format(j["flip"])].add(r)
                if j["d"] <= 2:
                    groups["st2_d_le_2"].add(r)
            if j["flip"] == 0.0 and r["in_pool"] and r["sh_first_disagree"] >= 0 \
                    and r["sh_leader_first"] == r["obj"]:
                sh0c_hits += 1
    m = {k: t.metrics() for k, t in cells.items()}
    gm = {k: t.metrics() for k, t in groups.items()}
    v = {}
    ident = [r["identical"] for r in recs]
    v["SH0"] = {"parts": {
        "a_identical": len(recs) == N_JOBS and all(x is True for x in ident),
        "b_rederived": all(r["rederive_failures"] == 0 and r["rederived"] > 0 for r in recs),
        "c_truth_never_struck_at_flip0": sh0c_hits == 0},
        "jobs": len(recs), "not_identical": [r["id"] for r in recs if r["identical"] is not True],
        "flip0_hits": sh0c_hits}
    noisy = gm.get("st2_noisy", {})
    d2 = gm.get("st2_d_le_2", {})
    tail = m.get(("end", 0.75, 0.0, "norepeat"), {})
    v["SH1"] = {"value": noisy.get("sh1_inpool_disagree_leader_is_secret", float("nan")),
                "n": noisy.get("sh1_n", 0), "min": SH1_MIN}
    v["SH1"]["pass"] = v["SH1"]["value"] >= SH1_MIN
    v["SH2"] = {"value": d2.get("sh2_parent_tested_before_disagreement", float("nan")),
                "n": d2.get("sh2_n", 0), "min": SH2_MIN,
                "parent_guessed_at_all": d2.get("lookalike_parent_guessed"),
                "reported_disagree_leader_is_parent": d2.get(
                    "lookalike_disagree_leader_is_parent"),
                "reported_disagree_n": d2.get("lookalike_disagree_n")}
    v["SH2"]["pass"] = v["SH2"]["value"] >= SH2_MIN
    v["SH3"] = {"value": noisy.get("sh3_inpool_lost_leader_is_secret", float("nan")),
                "n": noisy.get("sh3_n", 0), "min": SH3_MIN}
    v["SH3"]["pass"] = v["SH3"]["value"] >= SH3_MIN
    v["SH4"] = {"value": tail.get("inpool_expert_leader_is_secret", float("nan")),
                "n": tail.get("inpool_expert_n", 0), "max": SH4_MAX}
    v["SH4"]["pass"] = v["SH4"]["value"] <= SH4_MAX
    gain = (noisy.get("revival_leader_is_secret", float("nan"))
            - noisy.get("revival_revived_is_secret", float("nan")))
    v["SH5"] = {"value": gain, "revivals": noisy.get("revivals", 0),
                "leader_is_secret": noisy.get("revival_leader_is_secret"),
                "revived_is_secret": noisy.get("revival_revived_is_secret"), "min": SH5_MIN}
    v["SH5"]["pass"] = gain >= SH5_MIN
    for k in ("SH1", "SH2", "SH3", "SH4", "SH5"):
        v[k]["verdict"] = "PASS" if v[k]["pass"] else "FAIL"
    v["SH0"]["verdict"] = "PASS" if all(v["SH0"]["parts"].values()) else "FAIL"
    v["follow_on"] = ("build: soft-chosen revival test"
                      + (" (noise and crowding)" if not v["SH4"]["pass"] else " (noise only)")
                      if v["SH3"]["pass"] and v["SH5"]["pass"] else "do not build")
    return {"schema": "twenty_questions_stress_shadow_analysis_v1", "run_dir": run_dir,
            "verdicts": v, "groups": gm,
            "cells": [dict(zip(("src", "a", "b", "c"), k), **mm) for k, mm in sorted(
                m.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]}


def figure(out: dict, run_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = out["cells"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for flip, ls in ((0.05, "--"), (0.10, "-")):
        cs = sorted((c for c in cells if c["src"] == "st2" and c["b"] == flip),
                    key=lambda c: c["a"])
        ds = [c["a"] for c in cs]
        ax.plot(ds, [c["sh1_inpool_disagree_leader_is_secret"] for c in cs], ls,
                marker="o", color="tab:blue", label="SH1, flip {:g}".format(flip))
        ax.plot(ds, [c["sh3_inpool_lost_leader_is_secret"] for c in cs], ls,
                marker="s", color="tab:orange", label="SH3, flip {:g}".format(flip))
    ax.set_xlabel("look-alike distance d")
    ax.set_ylabel("share: soft's leader is the secret")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("ST-2: in-pool games", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    for src, arm, color, lab in (("st3", "primary", "tab:gray", "published P0a"),
                                 ("st3", "norepeat", "tab:green", "no_repeat, ST-3 rule"),
                                 ("end", "norepeat", "tab:red", "no_repeat, always")):
        cs = sorted((c for c in cells if c["src"] == src and c["c"] == arm and c["b"] == 0.0),
                    key=lambda c: c["a"])
        ax.plot([c["a"] for c in cs], [c["inpool_expert_leader_is_secret"] for c in cs],
                marker="o", color=color, label=lab)
    ax.set_xlabel("tail sigma")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("ST-3, flip 0: in-pool expert calls", fontsize=9)
    ax.legend(fontsize=7)
    fig.text(0.01, 0.01, "Shadow-mode soft beside P0a: share of games in which soft's leading "
             "known is the secret. analyze_shadow, {}".format(run_dir), fontsize=6)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = os.path.join(run_dir, "fig_shadow.png")
    fig.savefig(path, dpi=130)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    out = analyze(a.run_dir)
    out["figures"] = [os.path.basename(figure(out, a.run_dir))]
    with open(os.path.join(a.run_dir, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True, default=str)
    v = out["verdicts"]
    print("{} jobs".format(v["SH0"]["jobs"]))
    for k in ("SH0", "SH1", "SH2", "SH3", "SH4", "SH5"):
        extra = "" if k == "SH0" else "  value {:.3f} (n {})".format(
            v[k]["value"], v[k].get("n", v[k].get("revivals")))
        print("  {}  {}{}".format(k, v[k]["verdict"], extra))
    print("  follow-on: {}".format(v["follow_on"]))
    print("wrote {}".format(os.path.join(a.run_dir, "analysis.json")))


if __name__ == "__main__":
    main()
