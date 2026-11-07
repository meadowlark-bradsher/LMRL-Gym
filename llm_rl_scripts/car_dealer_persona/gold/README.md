# Hand labels for the outcome scorer

Two files, 80 conversations each, one JSON object per line: `index`, `id` (sha1 of the
conversation text), `outcome`, `price`, `discount_offered`, and the conversation text itself.

- `gold_labels.jsonl`: the GOLD split (`data.splits()["gold"]`, seed 20261106). Gate set for
  scorer v1, whose rules were written against DEV only. Its errors were then used to revise the
  rules, so it is development data now.
- `reserve_labels.jsonl`: the RESERVE split. Gate set for the revised scorer; read and labelled
  only after the revised scorer was frozen (sha256 recorded before labelling), and never used
  to change a rule.

**Labeler:** Claude, on 2026-11-06, reading each conversation in full with `buyer_info`
(persona, budget) hidden and before the scorer was run on that split.

**Definitions**

- `outcome`
  - `sale`: by the end the buyer commits to buy, or the dealer closes the sale and the buyer
    does not contest it.
  - `no_sale`: the buyer declines, leaves, or defers the decision ("I'll think about it").
  - `unclear`: neither. GOLD: the dealer moves to close while the buyer has only said "willing
    to consider" (index 23) or asked for a test drive (26), and a transcript whose buyer turns
    are placeholders (71). RESERVE: the story ends before any decision (1, 49) or on the way to
    a test drive (58).
- `price`: the dollar amount the sale closes at, as stated in the dialogue turns (not in the
  narrator's annotations); `null` if no sale or the text gives a placeholder (index 5).
- `discount_offered`: the dealer offers a lower price than one already quoted for the same
  car, or an explicit discount, promotion or "special price", whether or not it is accepted.
  Not counted: a cheaper alternative car, financing, free accessories or warranties, and vague
  talk of discounts that never becomes an offer. Borderline GOLD items labelled `true`: 53, 77,
  78 ("special deal/price" with no earlier quote); labelled `false`: 61 (dealer explicitly
  denies a discount, then quotes a first price). Borderline RESERVE items: 40 labelled `true`
  ("special price" on an older model), 52 labelled `no_sale` (the buyer says the price range
  works, then leaves without buying; the scorer calls it a sale).

Gate results: `results/persona_scorer_validation_20261106/validation.json` (scorer v1 on
GOLD) and `results/persona_scorer_validation_20261106_v2/validation.json` (revised scorer on
RESERVE; `python3 -m llm_rl_scripts.car_dealer_persona.validate`).
