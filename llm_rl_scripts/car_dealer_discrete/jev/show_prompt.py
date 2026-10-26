"""Print the EXACT text Jev sees, generated from policy.py — never transcribed.

One canonical state (buyer, ranking price>warranty>delivery, easy regime, after a
seller opening of P2-W0-D0) rendered under the three conditions used in the study:
baseline, salience, and the policy aid (bayes). No API call; uses the mock client.

    PYTHONPATH=. python3 -m llm_rl_scripts.car_dealer_discrete.jev.show_prompt
"""

from __future__ import annotations

from fractions import Fraction

from LLM_RL.jev.client import JevClient, MockJevBackend
from llm_rl_scripts.car_dealer_discrete.game import ALL_DEALS
from llm_rl_scripts.car_dealer_discrete.jev.aid import AcceptanceAid
from llm_rl_scripts.car_dealer_discrete.jev.policy import JevNegotiationPolicy
from llm_rl_scripts.car_dealer_discrete.protocol import ACCEPT, OFFER, REJECT, WALK, Action, Observation, TurnRecord


def canonical_obs() -> Observation:
    legal = (Action(ACCEPT), Action(REJECT)) + tuple(Action(OFFER, d.id) for d in ALL_DEALS) + (Action(WALK),)
    return Observation(
        role="buyer", own_ranking=("price", "warranty", "delivery"), reservation=Fraction(2, 5),
        max_proposals=6, proposals_used=1, outstanding_offer="P2-W0-D0",
        history=(TurnRecord("seller", Action(OFFER, "P2-W0-D0")),), legal_actions=legal,
    )


def main() -> int:
    obs = canonical_obs()
    client = JevClient(MockJevBackend())
    conds = {
        "baseline": JevNegotiationPolicy(client),
        "salience": JevNegotiationPolicy(client, salience=True),
        "aid (bayes)": JevNegotiationPolicy(client, aid=AcceptanceAid()),
    }
    print("Canonical state: buyer, ranking price > warranty > delivery, r = 0.40, after seller opened P2-W0-D0.\n")
    for name, pol in conds.items():
        opts, _ = pol.build_options(obs)
        print("=" * 78); print(f"CONDITION: {name}"); print("=" * 78)
        print("--- instructions ---"); print(pol.build_instructions(obs))
        print("--- state ---"); print(pol.build_state(obs))
        print("--- example options (of %d) ---" % len(opts))
        for key in ("ACCEPT P2-W0-D0", "OFFER P1-W1-D1", "OFFER P0-W2-D2", "WALK"):
            print(f"  [{key}] {opts[key]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
