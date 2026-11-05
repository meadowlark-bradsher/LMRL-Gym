"""One-sided elicited negotiation (toy).

A focal SELLER must OFFER one deal to a buyer whose preferences are PRIVATE. The
seller may ASK binary questions first; the buyer answers through a (possibly
strategic/noisy) kernel; each question carries a TOKEN COST subtracted from the
payoff. The seller's payoff is its own utility of the offered deal if the buyer
accepts, else 0. Information lets the seller target an ambitious-but-acceptable
offer -- so this is value of information that is *inter-communicative* (held by
another agent, acquired by asking, answered strategically), not the single-agent
read-efficiency notion of "information value".

It extends the CD-DRL finding (``car_dealer_drl``) that a shaping/acquisition signal
is only as good as its VALUE content: here the exact value-of-information oracle is
the measuring stick, and later increments test where an EIG (expected information
gain) acquisition rule tracks it and where it does not -- the regime question for a
negotiating-agent harness.

Increment 1 (this commit): ``game`` (types, utilities, acceptance, question bank,
answer kernels, belief ops) and ``oracle`` (the exact VOI dynamic program: V*,
V*_noask, VOI), with tests and a calibration. No EIG baseline, learned policy or
rung scaffold yet.
"""
