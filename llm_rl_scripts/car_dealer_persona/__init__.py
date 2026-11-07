"""Persona-planting reliability audit for the original LMRL-Gym Car Dealer task.

Does a planted persona (strategy background, brand, features, budget) govern what the
counterparty does? If it does, an LLM counterparty is a noisy realization of a known
type and can serve as a measured oracle for elicitation studies
(``elicit_negotiation``); the violation rate is the noise level.

CD-PERSONA-0 is the offline 2023 baseline, on the released dataset:

* ``PERSONA_AUDIT_TARGET.md`` -- the paper (arXiv 2311.18232) lined up against the code
  and the released data.
* ``PREREGISTRATION_BASELINE2023.md`` -- measures, predictions and the scorer gate.
* ``data`` -- sha256-pinned loading, de-duplication, frozen DEV / GOLD / RESERVE /
  CONFIRMATORY splits.
* ``scorer`` -- persona-blind outcome scorer: sale / no_sale / unclear from the dialogue
  turns alone, agreed price, discount offered, and the re-parsed self-report.
* ``gold/`` and ``validate`` -- 160 hand-labelled conversations and the validity gate.
* ``audit`` -- blind hand-audit of scorer-called outcomes in the hard-constraint cells,
  and the audit-corrected rate.
* ``analyze`` -- the confirmatory analyzer (measures M1-M5, verdict bands, predictions).

The confirmatory analysis has not been run: ``analyze`` refuses to read the confirmatory
split without ``--confirmatory``, which is to be passed only after the registration is
frozen.
"""
