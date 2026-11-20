"""Within-secret variance of total-cost differences in FLYW-1 Part 1 (seeds 1..N).

Sizes the noise-seed count for FLYW1_COST_PREREG.md. Two SEs are written: ``se_*`` with
FLYW-1's (map, secret) unit, and ``se_secret_*`` with the secret as the unit (the paired
difference averaged over the construction's maps at each seed, then its variance across
seeds), which is the registered one: a cell's maps share A's games and the keyed noise. Prints means too: Part 1 is already
read, and these are part of that prereg's disclosed basis. Ratio r means c_expert = 10 and
c_wrong = 10 / r, so r = 2.0 is c_wrong = 5.

    python -m llm_rl_scripts.twenty_questions.jev.flyw1_cost_basis \
        llm_rl_scripts/twenty_questions/jev/results/flyw1_part1_20261118 10 \
        llm_rl_scripts/twenty_questions/jev/results/flyw1_cost_basis_20261119/cost_var.json
"""
import gzip, json, os, sys, math, glob, statistics
from collections import defaultdict
from llm_rl_scripts.twenty_questions.jev.flyw1_decision import score_a_record
from llm_rl_scripts.twenty_questions.jev.analyze_flyw1 import load_maps
R = sys.argv[1]; NS = int(sys.argv[2]); OUT = sys.argv[3]
maps = load_maps()
def constr(m):
    if m == "cat17": return "category"
    for k in ("align", "mix05", "mix15", "rand"):
        if k in m: return k
def cost(r, ratio): return r["probes"] + (r["outcome"] != "right") * 10.0 + r["wrong"] * 10.0 / ratio
A = {}; ARM = defaultdict(dict)
for p in glob.glob(os.path.join(R, "jobs", "*.json.gz")):
    b = os.path.basename(p)
    j = json.load(gzip.open(p)); jb = j["job"]
    if jb["flip"] == 0.0 or jb["seed"] > NS or jb["arm"] not in ("A", "B", "D"): continue
    if jb["arm"] == "A": A[(jb["flip"], jb["seed"])] = j["games"]
    elif jb["map_id"] in maps: ARM[(jb["map_id"], jb["arm"], jb["flip"])][jb["seed"]] = {g["secret"]: g for g in j["games"]}
out = {}
for flip in (0.05, 0.1):
  for c in ("align", "mix05", "mix15", "rand", "category"):
    ms = [m for m in maps if constr(m) == c]
    for pair in (("B", "A"), ("D", "B")):
      for ratio in (2.0, 0.5, 0.2, 0.05):
        sum_var = 0.0; n_units = 0; diffs_all = []
        by_sec = defaultdict(lambda: defaultdict(list))   # secret -> seed -> diffs over maps
        for m in ms:
            cls = maps[m]["classes"]
            per = defaultdict(list)
            for s in range(1, NS + 1):
                def get(arm):
                    if arm == "A":
                        return {g["secret"]: score_a_record(g, cls) for g in A[(flip, s)]}
                    return ARM[(m, arm, flip)][s]
                x, y = get(pair[0]), get(pair[1])
                for sec in x:
                    d = cost(x[sec], ratio) - cost(y[sec], ratio)
                    per[sec].append(d); diffs_all.append(d)
                    by_sec[sec][s].append(d)
            for sec, v in per.items():
                sum_var += statistics.variance(v); n_units += 1
        se = math.sqrt(sum_var) / n_units / math.sqrt(NS)
        var_sec = sum(statistics.variance([statistics.mean(v) for v in per_seed.values()])
                      for per_seed in by_sec.values())
        se_sec = math.sqrt(var_sec) / len(by_sec) / math.sqrt(NS)
        out["{}|{}|{}-{}|r{}".format(c, flip, pair[0], pair[1], ratio)] = {
            "mean_diff": round(statistics.mean(diffs_all), 3), "se_at_N": round(se, 4),
            "se_at_5": round(se * math.sqrt(NS / 5), 4), "units": n_units,
            "se_secret_at_N": round(se_sec, 4),
            "se_secret_at_5": round(se_sec * math.sqrt(NS / 5), 4), "secrets": len(by_sec)}
json.dump(out, open(OUT, "w"), indent=1)
for k, v in out.items(): print(k, v)
