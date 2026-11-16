"""Exploratory follow-up to the shadow study (not registered, not scored).

Why is `soft`'s disagreeing leader so rarely the secret under noise? Replays one
committed noisy ST-2 P0a job with the shadow on, and also tracks, for every known,
the answers that contradict it under the hard rule (observed-answer likelihood
below 0.5). At the first disagreeing turn of each in-pool game it records the
leader's contradictions and the secret's rank and mass under `soft`.

Usage (one job, about 2 min, one core):
  PYTHONHASHSEED=0 PYTHONPATH=. python -m llm_rl_scripts.twenty_questions.jev.shadow_why_leader \
      exact_d1_la1_f0.1_s1_g0_modeswitch_guard
"""
import statistics as st
import sys

from . import stress_shadow as S


def main(job_id: str) -> None:
    rows = []
    orig_call, orig_finish = S.ShadowObserver.__call__, S.ShadowObserver.finish

    def call(self, turn, q, ans, bk, surv_before):
        if not hasattr(self, "_contra"):
            self._contra = {n: [] for n in self.names}
            self._rec = None
        for n in self.names:
            p = bk.like(q, bk.by_name[n])
            po = p if ans else 1 - p
            if po < 0.5:
                self._contra[n].append(round(po, 3))
        first_before = self.first
        orig_call(self, turn, q, ans, bk, surv_before)
        if first_before < 0 and self.first >= 0:
            post = self.sp.post
            leader = self.names[[self.idx[n] for n in self.names].index(self.leader_first)]
            self._rec = dict(leader=leader, p_leader=post[leader],
                             leader_contra=list(self._contra[leader]),
                             ranked=sorted(self.names, key=lambda n: -post[n]),
                             post=dict(post))

    def finish(self, ep):
        r = getattr(self, "_rec", None)
        s = ep.true_name
        if r and s in self._contra:          # in-pool game with a disagreement
            rows.append(dict(leader_is_secret=r["leader"] == s, p_leader=r["p_leader"],
                             p_secret=r["post"][s], secret_rank=r["ranked"].index(s),
                             leader_contra=r["leader_contra"]))
        orig_finish(self, ep)

    S.ShadowObserver.__call__, S.ShadowObserver.finish = call, finish
    committed = S._load("st2", job_id)
    S.replay("st2", committed["job"], "real")

    n = len(rows)
    ls = sum(r["leader_is_secret"] for r in rows)
    oth = [r for r in rows if not r["leader_is_secret"]]
    lc = [len(r["leader_contra"]) for r in oth]
    po = [r["leader_contra"][0] for r in oth if len(r["leader_contra"]) == 1]
    print(job_id)
    print(f" in-pool games with a disagreement: {n}; leader is the secret: {ls} ({ls / n:.3f})")
    print(f" other leaders: contradicting answers median {st.median(lc)}, "
          f"share with exactly one {sum(x == 1 for x in lc) / len(lc):.3f}")
    print(f" that one contradiction's likelihood: median {st.median(po)}, "
          f"share >= 0.3 {sum(x >= 0.3 for x in po) / len(po):.3f}")
    print(f" secret's soft rank (0 = leader): median {st.median(r['secret_rank'] for r in oth)}; "
          f"secret/leader mass median {st.median(r['p_secret'] / r['p_leader'] for r in oth):.3f}; "
          f"secret at mass 0: {sum(r['p_secret'] == 0 for r in oth)}")


if __name__ == "__main__":
    main(sys.argv[1])
