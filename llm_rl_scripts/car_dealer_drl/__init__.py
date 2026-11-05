"""Standalone deep-RL focal policy for the live Car-Dealer game (CD-DRL).

Rung R0 (this package, first cut): a small PyTorch per-row scorer + value head
trained by **behavior cloning** against the exact ``exact_ask`` oracle
(``car_dealer_live.Cell.ask``) over the *same shortlist menu Jev sees*. No RL,
no reward shaping yet — R0's only job is to show the featurization and a tiny
net can reproduce the exact menu policy, which is the prerequisite for the later
rungs (R1 potential-based shaping with Phi=V*, R2 reward-only VOI discovery, R3
best-response to the CD-MIS opponent population, R4 weaker/absent oracles).

The policy is a drop-in ``policy(state, belief) -> Action`` over the menu rows,
exactly like the scripted and Jev arms in ``car_dealer_live``.
"""
