"""Checks for the CFR check's engine (``cfr.py``), per `DESIGN_SPLIT_CFR.md` §9 steps 2-3:

* Kuhn poker, a known answer: CFR+ reaches the first player's value -1/18, and uniform play has NashConv 11/12;
* the tree is the one ``spec_numbers_cfr.py`` counts;
* the ladder on the tree: A1, A3, B0 and B2 as behaviour profiles have the values of the readers' exact rollouts,
  and the best responses to B0, A1 and B2 have the values of A1, B2 and A3 (the three reproductions of §4);
* on G's instance at Q = 2, the merged bank leaves (A1, B0) and (A1, B2) unchanged (§3, §9 step 2), compared deal
  by deal without reading any value;
* a smallest Split Deal case (Q = 1 on the 10-type test instance, T3) is certified by CFR+.

Every Split Deal value here is on the 10-type test instance, not on G, except the deal-by-deal comparison.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/split_deal/test_cfr.py
"""

import numpy as np
import pytest

from . import cfr
from . import spec_numbers_cfr as sc
from .game import Instance
from .readers import A1Model, B2, rollout_a1_b0, rollout_a1_b2
from .twosided import Shared

TINY = Instance((2, 3, 4))                         # the 10-type test instance of test_readers.py
SMALL_CASE_ITERATIONS = 8000


def test_kuhn_known_answer():
    g = cfr.kuhn_game()
    nc = g.nashconv(cfr.solve(g, 2000))
    assert abs(nc["value"][0] + 1 / 18) < 1e-3 and abs(nc["value"][0] + nc["value"][1]) < 1e-12
    assert nc["nashconv"] < 1e-3
    assert abs(g.nashconv((g.uniform(0), g.uniform(1)))["nashconv"] - 11 / 12) < 1e-12


@pytest.mark.parametrize("bank,q,arm", [("full", 1, "T3"), ("full", 2, "game:none/all"), ("merged", 2, "T3"),
                                        ("merged", 2, "game:all/none")])
def test_tree_is_the_counted_tree(bank, q, arm):
    tab = cfr.g_tables(TINY, bank)
    g = cfr.split_game(tab, cfr.G_ARMS[arm], q)
    assert cfr.tree_counts(g) == sc.tree_counts(tab, q, cfr.G_ARMS[arm], list(range(len(tab.questions))))


def test_g_tree_counts_match_spec():
    tab = cfr.g_tables(Instance(cfr.G_INSTANCE), "merged")
    for arm in ("game:none/all", "game:all/none", "game:none/none"):
        g = cfr.split_game(tab, cfr.G_ARMS[arm], cfr.G_Q)
        assert cfr.tree_counts(g) == sc.tree_counts(tab, cfr.G_Q, cfr.G_ARMS[arm], list(range(len(tab.questions))))


@pytest.mark.parametrize("bank,q,arm", [("full", 1, "T2"), ("full", 1, "T3"), ("full", 1, "game:none/all"),
                                        ("full", 1, "game:all/none"), ("merged", 2, "T3"),
                                        ("merged", 2, "game:none/none")])
def test_ladder_profiles_and_reproductions(bank, q, arm):
    g = cfr.split_game(cfr.g_tables(TINY, bank), cfr.G_ARMS[arm], q)
    lad = cfr.Ladder(g)
    P = {nm: lad.profile(nm) for nm in ("A1", "A3", "B0", "B2")}
    ex = {pr: lad.exact(pr) for pr in ("A1/B0", "A1/B2", "A3/B2")}
    for pr, (ya, yb) in ex.items():
        a, b = pr.split("/")
        u = g.value((P[a], P[b]))
        assert abs(u[0] - float(ya)) < cfr.TOL and abs(u[1] - float(yb)) < cfr.TOL, pr
    assert abs(g.evaluate(0, None, P["B0"], br=True)["value"] - float(ex["A1/B0"][0])) < cfr.TOL
    assert abs(g.evaluate(1, None, P["A1"], br=True)["value"] - float(ex["A1/B2"][1])) < cfr.TOL
    assert abs(g.evaluate(0, None, P["B2"], br=True)["value"] - float(ex["A3/B2"][0])) < cfr.TOL


@pytest.mark.parametrize("bank,q,arm", [("full", 1, "T3"), ("merged", 2, "T3"), ("merged", 2, "game:none/all")])
def test_offer_pruning_changes_no_best_response(bank, q, arm):
    vals = []
    for prune in (True, False):
        g = cfr.split_game(cfr.g_tables(TINY, bank), cfr.G_ARMS[arm], q, prune=prune)
        lad = cfr.Ladder(g)                                   # raises if a ladder offer is off the menu
        P = {nm: lad.profile(nm) for nm in ("A1", "A3", "B0", "B2")}
        vals.append([g.evaluate(0, None, P[b], br=True)["value"] for b in ("B0", "B2")] +
                    [g.evaluate(1, None, P[a], br=True)["value"] for a in ("A1", "A3")] +
                    [x for pr in (("A1", "B2"), ("A3", "B2")) for x in g.value((P[pr[0]], P[pr[1]]))])
    assert np.allclose(vals[0], vals[1], atol=1e-12, rtol=0)
    assert g.M[0] == len(g.tab.allocs) + 1                    # unpruned: every allocation and walking


def test_best_response_policy_realises_its_value():
    g = cfr.split_game(cfr.g_tables(TINY, "full"), cfr.G_ARMS["T3"], 1)
    lad = cfr.Ladder(g)
    b2 = lad.profile("B2")
    br = g.evaluate(0, None, b2, br=True)
    assert abs(g.value((br["policy"], b2))[0] - br["value"]) < cfr.TOL
    assert all(np.isin(x, (0.0, 1.0)).all() for x in br["policy"])


def test_warm_start_begins_at_the_warm_profile():
    g = cfr.split_game(cfr.g_tables(TINY, "full"), cfr.G_ARMS["T3"], 1)
    lad = cfr.Ladder(g)
    warm = (lad.profile("A3"), lad.profile("B2"))
    one = cfr.solve(g, 1, warm=cfr.as_regrets(warm))
    assert np.allclose(g.value(one), g.value(warm), atol=cfr.TOL)


def _deals(bank, st):
    tab = cfr.g_tables(Instance(cfr.G_INSTANCE), bank)
    sh, n = Shared(tab), len(tab.types)
    m = A1Model(sh, st, cfr.G_Q, cfr.G_PI)
    b2s = {b: B2(sh, m, b) for b in range(n)}
    d0 = [[(o["a_deal"], o["b_deal"]) for o in (rollout_a1_b0(sh, m, ia, b) for b in range(n))] for ia in range(n)]
    d2 = [[(o["a_deal"], o["b_deal"]) for o in (rollout_a1_b2(sh, m, ia, b2s[b]) for b in range(n))]
          for ia in range(n)]
    return d0, d2


@pytest.mark.parametrize("arm", list(cfr.G_ARMS))
def test_merged_bank_leaves_a1_b0_and_a1_b2_unchanged_on_g(arm):
    full, merged = _deals("full", cfr.G_ARMS[arm]), _deals("merged", cfr.G_ARMS[arm])
    assert full[0] == merged[0], "(A1, B0) differs"
    assert full[1] == merged[1], "(A1, B2) differs"


def test_small_split_deal_case_is_certified():
    g = cfr.split_game(cfr.g_tables(TINY, "full"), cfr.G_ARMS["T3"], 1)
    nc = g.nashconv(cfr.solve(g, SMALL_CASE_ITERATIONS))
    assert nc["nashconv"] <= cfr.EPS_BAR
    assert min(nc["eps"]) >= -cfr.TOL                        # a best response is never worse than the profile

