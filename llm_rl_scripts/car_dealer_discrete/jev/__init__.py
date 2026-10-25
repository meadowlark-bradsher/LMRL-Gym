"""Jev-only agents for the discretized Car Dealer game (J/J cell).

Pure Jev-only: the policy offers exactly the currently-legal actions as the
options of a single ``Choice`` question and returns the selected one. No belief
filter, no text generation — so this measures the outcome/efficiency score
families (spec sections 4.1/4.2/4.4) but not the epistemic family (4.3).
"""
