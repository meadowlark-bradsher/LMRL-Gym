"""Discretized Car Dealer: a finite bilateral negotiation game.

A new, scored redesign *inspired by* LMRL-Gym Car Dealer (see
``discretized_car_dealer_experiment.md``). This is NOT the original
seller-price benchmark: both agents are decision-makers, both receive
separate scores, and the finite type/action spaces make exhaustive
outcome scoring possible.

Game version: ``car-dealer-discrete-v0.1``.

Milestone 1 (this package) is deliberately agent-agnostic: an exact,
inspectable game engine with exhaustive truth-table scoring and a legal
action/turn state machine. No model (Jev / LLM / BED) appears here.
"""

GAME_VERSION = "car-dealer-discrete-v0.1"
