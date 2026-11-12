"""Checks for gate SD3-E (``sweep_e.py``): the reductions against the readers' exact rollouts, the solve driver's
stopping and fallback, the gate logic on synthetic inputs, a smoke run end to end on a 6-type instance, the
registered consequences quoted verbatim, and regeneration of the stored results and note.

Every Split Deal value here is on a small test instance, not on G, except the regeneration tests.

    PYTHONPATH=. python3 -m pytest -q llm_rl_scripts/split_deal/test_sweep_e.py
"""

import json
import os
from fractions import Fraction

import pytest

from . import cfr
from . import sweep_e as se
from .game import Instance
from .private_d import HIGH

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL = "results/split_v03_e_20261111/sweep.json"
NOTE = os.path.join(ROOT, "llm_rl_scripts", "split_deal", "SWEEP_SPLIT_E.md")
DESIGN = os.path.join(ROOT, "llm_rl_scripts", "split_deal", "DESIGN_SPLIT_CFR.md")
TINY = Instance((2, 3, 4))                                    # 10 types
SMOKE = dict(se.G_CONFIG, instance=[3, 3, 4], q=1, every=50, budget=100, max_iterations=400, eps_bar=1e-2)


# -- the reductions against the readers ------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_t3():
    g = cfr.split_game(cfr.g_tables(TINY, "merged"), cfr.G_ARMS["T3"], 2)
    lad = cfr.Ladder(g)
    return g, lad, {nm: lad.profile(nm) for nm in ("A1", "A3", "B0", "B2")}


def _rollouts(lad, pairing):
    from .readers import rollout_a1_b0, rollout_a1_b2, rollout_a3_b2
    sh, m, n = lad.g.shared, lad.m, lad.n
    for ia in range(n):
        for b in range(n):
            yield ia, b, (rollout_a1_b0(sh, m, ia, b) if pairing == "A1/B0" else
                          rollout_a1_b2(sh, m, ia, lad.b2s[b]) if pairing == "A1/B2" else
                          rollout_a3_b2(sh, lad.a3s[ia], b))


@pytest.mark.parametrize("pairing", se.LADDER)
def test_outcomes_match_exact_rollouts(tiny_t3, pairing):
    g, lad, P = tiny_t3
    tab, sh, pi = g.tab, g.shared, g.pi_exact
    prof = (P[pairing[:2]], P[pairing[3:]])
    out = se.outcomes(g, prof)
    val = g.value(prof)
    assert abs(out["all"]["YA"] - val[0]) < cfr.TOL and abs(out["all"]["YB"] - val[1]) < cfr.TOL
    acc = {sc: {"YA": Fraction(0), "agree_a": 0, "agree_b": 0, "E": Fraction(0), "nE": 0, "n": 0} for sc in se.SCOPES}
    for ia, b, o in _rollouts(lad, pairing):
        ka, kb = o["a_deal"], o["b_deal"]
        ga = sh.seat[ia].gain
        ya = pi * (ga[ka] if ka is not None else 0) + (1 - pi) * (ga[kb] if kb is not None else 0)
        ta, tb = tab.types[ia], tab.types[b]
        ss = tab.s_star(ta, tb)
        e = Fraction(0)
        for k, w in ((ka, pi), (kb, 1 - pi)):
            if k is not None:
                e += w * (tab.inst.utility(ta[0], tab.allocs[k]) + tab.inst.utility(tb[0], tab.inst.rest(tab.allocs[k]))
                          - tab.d(ta) - tab.d(tb))
        for sc in ("all", ta[1]):
            a = acc[sc]
            a["YA"] += ya
            a["agree_a"] += ka is not None
            a["agree_b"] += kb is not None
            a["n"] += 1
            if ss:
                a["E"] += e / ss
                a["nE"] += 1
    for sc in se.SCOPES:
        a = acc[sc]
        assert abs(out[sc]["YA"] - float(a["YA"] / a["n"])) < cfr.TOL
        assert abs(out[sc]["agree_a"] - a["agree_a"] / a["n"]) < cfr.TOL
        assert abs(out[sc]["agree_b"] - a["agree_b"] / a["n"]) < cfr.TOL
        assert abs(out[sc]["E"] - float(a["E"] / a["nE"])) < cfr.TOL


@pytest.mark.parametrize("pairing", ("A1/B0", "A1/B2"))
def test_high_stats_match_exact_rollouts(tiny_t3, pairing):
    g, lad, P = tiny_t3
    h = se.high_stats(g, (P[pairing[:2]], P[pairing[3:]]))
    jh = g.tab.questions.index(HIGH)
    n = lad.n
    cnt = {}
    for ia, b, o in _rollouts(lad, pairing):
        lvl = g.tab.types[ia][1]
        for j, ans, rd in o["decisions"]:
            if j == jh:
                c = cnt.setdefault((str(rd), lvl), [0, 0])
                c[0] += 1
                c[1] += bool(ans)
    assert {lv for _, lv in cnt} == {"lo", "hi"}                      # B asks HIGH of both levels: not vacuous
    for (rd, lvl), (asked, answered) in cnt.items():
        assert abs(h["rounds"][rd][lvl]["asked"] - asked / n ** 2) < cfr.TOL
        assert abs(h["rounds"][rd][lvl]["answered"] - answered / n ** 2) < cfr.TOL
    dec = {lvl: sum(a - b for (rd, lv), (a, b) in cnt.items() if lv == lvl) for lvl in ("lo", "hi")}
    if dec["lo"] + dec["hi"]:
        assert abs(h["p_lo_given_decline"] - dec["lo"] / (dec["lo"] + dec["hi"])) < cfr.TOL


# -- the solve driver -------------------------------------------------------------------------------

def test_solve_stops_at_first_certified_checkpoint():
    r = se.solve_job(dict(SMOKE, eps_bar=10.0), "T3", "uniform")
    assert r["certified"] and r["stage"] == "budget" and r["iterations"] == SMOKE["every"] and len(r["log"]) == 1


def test_solve_falls_back_then_records_not_certified():
    cfg = dict(SMOKE, eps_bar=-1.0, every=50, budget=50, max_iterations=100)
    r = se.solve_job(cfg, "game:none/none", "warm")
    assert not r["certified"] and r["stage"] == "not certified" and r["variant"] == "dcfr"
    assert [x["variant"] for x in r["log"]] == ["cfr+", "cfr+", "dcfr", "dcfr"]
    assert [x["iteration"] for x in r["log"]] == [50, 100, 50, 100]


# -- the gate logic -----------------------------------------------------------------------------------

def _prof(ya_lo=0.4, ya_hi=0.4, yb=0.4, nashconv=0.0, high=None):
    sc = lambda ya: {"YA": ya, "YB": yb}                                                    # noqa: E731
    return {"outcomes": {"all": sc((ya_lo + ya_hi) / 2), "lo": sc(ya_lo), "hi": sc(ya_hi)}, "nashconv": nashconv,
            "high": high}


def _high(hi_rate=1.0, post=0.95, asked=0.5):
    return {"p_asked": {"lo": asked, "hi": asked}, "answer_rate": {"lo": 0.5, "hi": hi_rate},
            "p_lo_given_decline": post}


def _doc(t3=None, t2=None, cells=None, a3b2_t3=None, a3b2_t2=None):
    lad = {"T3": {"profiles": {"A3/B2": a3b2_t3 or _prof(0.45, 0.4, 0.35, 0.02)}},
           "T2": {"profiles": {"A3/B2": a3b2_t2 or _prof(0.43, 0.4, 0.35)}}}
    solves = {"T3": t3 or {}, "T2": t2 or {}}
    solves.update(cells or {})
    return {"ladder": lad, "solves": solves}


def _eq(p, ok=True):
    return {"certified": ok, "profile": p}


def test_gate_e1_e2_e4():
    d = _doc(t3={"uniform": _eq(_prof(0.42, 0.4, 0.38, high=_high())), "warm": _eq(_prof(0.42, 0.4, 0.38, high=_high()))},
             t2={"uniform": _eq(_prof(0.41, 0.4, 0.38))})
    g = se.gate(d)
    assert g["E1"]["verdict"] == "holds"                                     # 0.02 > 0.01
    assert g["E2"]["verdict"] == "holds"                                     # 0.38 > 0.35
    assert g["E4"]["verdict"] == "holds" and len(g["E4"]["readings"]) == 2   # 0.01 < 0.02, two (T3, T2) pairs
    d["ladder"]["T3"]["profiles"]["A3/B2"]["nashconv"] = 0.01
    assert se.gate(d)["E1"]["verdict"] == "fails"                            # not more than 0.01
    d["solves"]["T3"]["warm"] = _eq(_prof(0.42, 0.4, 0.30, high=_high()))
    assert se.gate(d)["E2"]["verdict"] == "equilibrium-dependent"
    d["solves"]["T3"]["warm"] = _eq(_prof(0.42, 0.4, 0.30, high=_high()), ok=False)   # uncertified: not read
    assert se.gate(d)["E2"]["verdict"] == "holds"
    d["solves"]["T3"] = {"uniform": _eq(_prof(), ok=False)}
    assert se.gate(d)["E2"]["verdict"] == se.gate(d)["E4"]["verdict"] == "not certified"


def test_gate_e3_reading():
    assert se.e3_reading(_high())["verdict"] == "holds"
    assert se.e3_reading(_high(hi_rate=0.94))["verdict"] == "fails"
    assert se.e3_reading(_high(post=0.89))["verdict"] == "fails"
    assert se.e3_reading(_high(post=None))["verdict"] == "holds"             # no declines on path: read as met
    assert se.e3_reading(_high(asked=0.049))["verdict"] == "not testable"    # support
    assert se.e3_reading(None)["verdict"] == "not testable"
    d = _doc(t3={"uniform": _eq(_prof(high=_high())), "warm": _eq(_prof(high=_high(asked=0.0)))})
    assert se.gate(d)["E3"]["verdict"] == "equilibrium-dependent"           # holds on one, not testable on the other


def test_gate_e5():
    def cell(ya, yb):
        return {"uniform": _eq(_prof(ya, ya, yb))}
    pd = {"T2": cell(0.4, 0.4), "game:none/all": cell(0.5, 0.2), "game:none/none": cell(0.3, 0.3),
          "game:all/none": cell(0.2, 0.5)}                                     # temptation > reward > punishment > sucker
    d = _doc(t2=pd["T2"], cells={k: v for k, v in pd.items() if k != "T2"})
    assert se.gate(d)["E5"]["verdict"] == "fails"
    d["solves"]["game:none/all"] = cell(0.3, 0.25)                           # disclosure no longer dominated
    d["solves"]["game:all/none"] = cell(0.25, 0.3)
    assert se.gate(d)["E5"]["verdict"] == "holds"
    d["solves"]["game:none/none"] = {"uniform": _eq(_prof(), ok=False)}
    assert se.gate(d)["E5"]["verdict"] == "not certified"


def test_consequences_are_quoted_verbatim():
    with open(DESIGN) as f:
        design = " ".join(f.read().split())
    for text in se.CONSEQUENCES.values():
        plain = text.replace("’", "'").replace("“", "\"").replace("”", "\"")
        assert plain in design, plain[:60]


# -- end to end on a small instance ----------------------------------------------------------------------

def test_smoke_compute_and_render():
    doc = se._jsonable(se.compute(SMOKE, workers=1))
    assert doc["checks"] == {"tree_vs_exact_ok": True, "reproductions_ok": True, "best_response_never_worse": True}
    assert set(doc["solves"]) == set(se.STRUCTS) and all(set(r) == set(se.STARTS) for r in doc["solves"].values())
    assert doc["sd3k_q3_full"] is None                                     # only read on G's instance
    doc["commit"], doc["seconds"] = "smoke", 0
    text = se.render(doc, "smoke.json")
    assert text.startswith("# Split Deal v0.3 — gate SD3-E, the CFR check on 3-3-4")
    par = se._jsonable(se.compute(SMOKE, workers=2))                       # the run's pool path
    par["commit"], par["seconds"] = "smoke", 0
    strip = lambda d: json.loads(json.dumps(d, sort_keys=True), parse_float=lambda x: round(float(x), 12))  # noqa: E731
    for dd in (doc, par):
        for runs in dd["solves"].values():
            for r in runs.values():
                r["seconds"] = r["cpu"] = 0
        for r in dd["ladder"].values():
            r["seconds"] = r["cpu"] = 0
    assert strip(par) == strip(doc)


# -- regeneration ---------------------------------------------------------------------------------------

def _stored():
    path = os.path.join(ROOT, *REL.split("/"))
    if not os.path.exists(path):
        pytest.skip("sweep results missing")
    with open(path) as f:
        return json.load(f)


def test_regenerate_small_cell_ladder():
    stored = _stored()
    fresh = se._jsonable(se.ladder_job(stored["config"], "game:none/none"))
    old = dict(stored["ladder"]["game:none/none"])
    for d in (fresh, old):
        d["seconds"] = d["cpu"] = 0
    assert json.loads(json.dumps(fresh)) == old


def test_regenerate_note():
    stored = _stored()
    if os.path.exists(NOTE):
        with open(NOTE) as f:
            assert f.read() == se.render(stored, REL)
