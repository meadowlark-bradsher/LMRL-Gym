"""Standalone deep-RL focal policy for the live Car-Dealer game (CD-DRL).

A small PyTorch per-row scorer + value head in the **planner/actor** seat of the
three-role frame, choosing from the *same shortlist menu Jev sees* over the
matched belief. Built as a curriculum that leans on the exact ``car_dealer_live``
oracle hard at first and weans off it:

* **R0** (``dataset``/``train``/``run_r0``): behavior cloning of ``exact_ask`` --
  shows the featurization and a tiny net can *reproduce* the exact menu policy.
* **R1** (``env``/``train_r1``/``run_r1``): reinforcement learning (advantage
  actor-critic) with **potential-based reward shaping** Phi = V*. The oracle
  enters only as the shaping potential, never as an action label; the net learns
  its policy from reward. Phi = V* makes the one-step shaped reward the optimal
  advantage, so RL recovers V* with dense help. A pre-registered shaping-off
  ablation isolates the shaping's contribution.
* R2: reward-only VOI discovery (does model-free RL find ASK without the oracle?);
  R3: best-response to the CD-MIS opponent population; R4: weaker/absent oracles.

The policy is a drop-in ``policy(state, belief) -> Action`` over the menu rows,
exactly like the scripted and Jev arms in ``car_dealer_live``.
"""
