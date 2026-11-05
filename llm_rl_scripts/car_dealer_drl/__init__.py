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
* **R2** (``ask_eval``/``run_r2``): reward-only VOI discovery. RL with *no oracle*
  (no BC labels, no PBRS potential) from the raw terminal reward; the net sees only
  the menu the Jev arm sees. Does it discover ASK where asking carries value? R2
  reports a pre-registered discovery verdict (captured-VOI fraction and
  ASK-optimal recall against the exact labels), mirroring the Jev 0/2555 finding.
* **R3** (``env_mis``/``evaluate_mis``/``train_r3``/``run_r3``): best-response to the
  CD-MIS opponent population. The TRUE opponent is drawn from the misspecification
  grid (``opponents_mis``) while the agent plans/observes only with its ASSUMED
  (registered T1) kernel -- shortlist menu, robustified belief filter
  (``filters_mis``), Phi = V*_assumed shaping. One policy is trained against a
  mixture of true opponents and scored against O_assumed (the registered oracle
  deployed under misspecification) and O_true (the clairvoyant per-draw ceiling),
  with a pre-registered transfer split to held-out opponents.
* **R4** (``potentials``/``run_r4``): weaker / absent oracles. Holding the nstep1
  learner fixed (R1 setup) and varying only the shaping potential Phi, a degradation
  ladder measures how much of R1's V* recovery survives: Phi=V* (exact), Phi=V* under
  a wrong kernel, Phi=V*_noask (inquiry-blind), Phi=entropy (oracle-free; the deepink
  potential) and Phi=0. Finding: every VALUE potential recovers V* fully (V* is
  kernel-robust, and the small value of inquiry is recovered from the true reward
  even by an inquiry-blind potential), but the oracle-free entropy potential fails at
  every scale -- rewarding information rather than value misleads here, the opposite
  of 20Q. The oracle's contribution is its VALUE content, not merely dense shaping.
  The H=6 arm (``run_r4_h6``) shows the ordering holds at the longer horizon
  (exact-Phi +0.61 vs entropy +0.17) and records an amortization finding: the exact
  oracle is O(1) after a one-time solve while the depth-capped approximation is
  recomputed lazily, so exact-Phi is the efficient value potential, not the
  approximate one.

The policy is a drop-in ``policy(state, belief) -> Action`` over the menu rows,
exactly like the scripted and Jev arms in ``car_dealer_live``.
"""
