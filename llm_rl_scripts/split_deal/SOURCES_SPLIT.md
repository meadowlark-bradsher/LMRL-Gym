# Split Deal — the surveyed systems, checked against their sources (NEG-27)

Checked 2026-11-10 against primary sources (papers, repositories, dataset cards). Read only; nothing was installed
or run. Most pages were read through a summarizing fetcher, so a claim marked *unverified* below means the page
was truncated or did not state it. Three load-bearing claims were re-read directly: the Deal or No Deal value
constraints, the NegMAS license, and the existence of TERMS-Bench's oracle reference.

The question for each system is whether an exact oracle could score it. That needs four things:

1. an outcome space that can be enumerated;
2. finite private types with a stated prior;
3. a counterpart that is a fixed, published rule (not an LLM or a recording);
4. a score that measures what we care about.

## Summary

| System | Enumerable outcomes | Finite known prior | Counterpart | Exact oracle feasible? |
|---|---|---|---|---|
| Deal or No Deal | yes: item counts, 3 kinds, 5–7 items | finite; the released scenarios define an empirical prior | recorded humans | for outcomes and scores; the human opponent is not modelled |
| CaSiNo | yes: 64 splits | yes: 6 priority orderings per side | recorded humans | for outcomes and scores; the human opponent is not modelled |
| TERMS-Bench | a continuous price, made discrete for the paper's own dynamic program | yes: explicit regime priors (values in appendices, *unverified*) | a fixed stochastic rule | yes, and the paper already does it; no code released |
| NegMAS | yes, for discrete issues | none built in; we supply it | any negotiator | yes, as a library (AGPL-3.0-or-later) |
| CraigslistBargain | a continuous price | buyer target is one of 3 fixed multiples of the list price | recorded humans | price only |
| AgenticPay | price, plus an optional contract mode | no stated sampling of private values | LLM | no, as released |
| PACT | a continuous price over 20 rounds | a named mix of four distributions, parameters unpublished | LLM | no |
| A2A-NT | no: price extracted from free text by an LLM | fixed constants per product | LLM | no |
| ProMediate | no: free-text stances, new options can appear | hand-authored profiles | LLM participants and mediator | no |

## What changes for the redesign

- **TERMS-Bench is the nearest prior work** and should be cited in any write-up. It already pairs a fixed stochastic
  counterpart with explicit type priors, and reports gaps to an oracle policy. It is a single-issue price game and
  has no words-as-evidence channel. Split Deal differs in being multi-issue (so information can create value), in
  making words the information channel, and in being two-sided.
  Source: Zhang et al., "TERMS-Bench: Diagnosing LLM Negotiation Agents Beyond Deal Rate", arXiv 2605.13909.
- **Deal or No Deal's constraints are as `DESIGN_SPLIT.md` §2 and §8 recalled them.** Every item is valued by at least
  one side, some item is valued by both, each side's values sum to 10, there are 5–7 items of 3 kinds, and
  conflicting final selections pay both sides zero. Two additions: no agreement within 10 turns also pays zero,
  and the paper says "non-negative" values without saying integer (*unverified* against the released data).
  Source: Lewis et al. 2017, arXiv 1706.05125.
- **CaSiNo's walk-away is public and equal**: 5 points each, the worth of one high-priority unit. Points are 5, 4
  and 3 per unit by priority, with a maximum of 36. Its outside option is therefore structured like Split Deal
  v0.2's. Source: Chawla et al., NAACL 2021, arXiv 2103.15721.
- **NegMAS is AGPL-3.0-or-later.** That is what its PyPI metadata (0.17.0) and LICENSE file say, although the README
  and PyPI description say BSD 3-Clause. Using it only as a test dependency for the L0 cross-check (NEG-29) is a
  different matter from vendoring it, but the choice is Meadowlark's. The relevant functions are
  `negmas.preferences.pareto_frontier` and `nash_points`, and `negmas.inout.Scenario.calc_stats`. Signatures vary by
  version.

## Corrections to the survey

- **AgenticPay task count.** The README says 160 tasks; the paper (arXiv 2602.06008 v1) says 111, and its abstract
  "over 110". "Multimodal" and "multidimensional contracts" appear only in the README. Both sides are LLMs.
- **"A2A Negotiation Arena"** is A2A-NT (Zhu et al., arXiv 2506.00073). "Arena" appears only in a reference to
  NegotiationArena.
- **TERMS-Bench's "private reservation, urgency, stance"** is the counterpart's hidden type. The agent under test
  holds only its own reservation price.
- **PACT** is right as described (price only, 20 rounds). The survey omits two things: trades can recur within a
  game, and the distribution parameters are unpublished. No license file was found (*unverified*).
- **ProMediate** is right that it has six multi-party, multi-topic cases. Its score is LLM-judged consensus change,
  so it has no ground-truth optimum.
- **Deal or No Deal, CaSiNo and CraigslistBargain** dataset sizes check out: 5,808 dialogues over 2,236 scenarios;
  1,030 dialogues with 396 strategy-annotated; 6,682 dialogues.

## Not checked

- NegMAS's protocol, in particular whether it has a query move.
- The fields of `ScenarioStats`.
- TERMS-Bench's appendices: numeric priors and kernel hyperparameters.
- Whether the Deal or No Deal released values are all integers.

## Sources

AgenticPay github.com/SafeRL-Lab/AgenticPay, arXiv 2602.06008 · PACT github.com/lechmazur/pact · TERMS-Bench
arXiv 2605.13909, terms-bench.github.io · A2A-NT arXiv 2506.00073, github.com/ShenzheZhu/A2A-NT · ProMediate
arXiv 2510.25224, github.com/microsoft/ProMediate-Eval · NegMAS pypi.org/project/negmas,
github.com/yasserfarouk/negmas, negmas.readthedocs.io · Deal or No Deal arXiv 1706.05125 · CaSiNo arXiv 2103.15721,
github.com/kushalchawla/CaSiNo · CraigslistBargain arXiv 1808.09637, huggingface.co/datasets/stanfordnlp/craigslist_bargains
