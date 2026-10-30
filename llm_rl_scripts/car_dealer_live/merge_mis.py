"""Merge the per-process CD-MIS-1 result dirs into one (runs.jsonl.gz concatenated, summaries merged).

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_live.merge_mis --dir results/live_mis_20261029
"""
import argparse, glob, json, os, shutil, sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args(argv)
    subs = sorted(d for d in glob.glob(os.path.join(a.dir, "*")) if os.path.isdir(d) and os.path.exists(os.path.join(d, "summary.json")))
    assert subs, "no per-process dirs"
    with open(os.path.join(a.dir, "runs.jsonl.gz"), "wb") as out:
        for d in subs:
            with open(os.path.join(d, "runs.jsonl.gz"), "rb") as f:
                shutil.copyfileobj(f, out)          # gzip members concatenate into one valid stream
    summ = [json.load(open(os.path.join(d, "summary.json"))) for d in subs]
    m = dict(summ[0]); m["processes"] = sorted({p for s in summ for p in s["processes"]})
    # per-process metadata and arm lists are unions over the sub-dirs (CD-MIS-2 runs different arms per family)
    meta = {}
    for s in summ:
        meta.update(s.get("process_meta") or {})
    if meta:
        m["process_meta"] = meta
    arms = []
    for s in summ:
        arms += [a for a in s["arms"] if a not in arms]
    m["arms"] = arms
    hedged = {}
    for s in summ:
        hedged.update(s.get("hedged_arms") or {})
    m["hedged_arms"] = hedged
    filt = {}
    for s in summ:
        filt.update(s.get("filters") or {})
    m["filters"] = filt
    m["episodes"] = sum(s["episodes"] for s in summ); m["error_stubs"] = sum(s["error_stubs"] for s in summ)
    m["elapsed_s"] = round(sum(s["elapsed_s"] for s in summ), 2); m["merged_from"] = [os.path.basename(d) for d in subs]
    json.dump(m, open(os.path.join(a.dir, "summary.json"), "w"), indent=2)
    print(f"merged {len(subs)} dirs: {m['episodes']} episodes, {m['error_stubs']} stubs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
