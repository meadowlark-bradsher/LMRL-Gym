# Does the live opponent target the focal side? (CD-NOUL-2 pilot, offline)

_Rendered by `opponent_targeting.py render` from `results/noul2_targeting_20261109/targeting.json` (pilot `results/noul2_pilot_20261108`, commit `5064896`, 1800 opponent proposals). No API calls; every number is formatted from the JSON. Cluster bootstrap B = 5000 (seed 0) over the (buyer ranking, seller ranking) type pairs. Exploratory: not one of the §6 gates._

`spec(d)` = the focal side's utility of the proposed deal under its true ranking, minus its mean utility under the five other rankings. It averages 0 for any proposal rule that ignores the focal side's ranking. `T1` is the expected value under the registered T1 kernel's proposal distribution in the same public state with the same opponent ranking. Utilities are in [0, 1] (not Y).

## Targeting, per brief

| brief | proposals | spec, live opponent | spec, T1 in same state | live − T1 | focal utility, live − T1 | own utility, live − T1 |
|---|---|---|---|---|---|---|
| candid | 562 | +0.0471 [+0.0027, +0.0906] | -0.0017 [-0.0347, +0.0314] | +0.0488 [+0.0238, +0.0746] | +0.2810 [+0.2538, +0.3099] | -0.2071 [-0.2200, -0.1929] |
| guarded | 623 | +0.0480 [+0.0062, +0.0905] | -0.0082 [-0.0408, +0.0250] | +0.0562 [+0.0352, +0.0771] | +0.2645 [+0.2379, +0.2916] | -0.2004 [-0.2127, -0.1889] |
| adversarial | 615 | +0.0377 [-0.0052, +0.0797] | -0.0048 [-0.0378, +0.0288] | +0.0426 [+0.0216, +0.0657] | +0.2421 [+0.2140, +0.2726] | -0.1923 [-0.2022, -0.1819] |
| pooled | 1800 | +0.0435 [+0.0023, +0.0845] | -0.0049 [-0.0379, +0.0281] | +0.0484 [+0.0284, +0.0690] | +0.2609 [+0.2382, +0.2852] | -0.1996 [-0.2091, -0.1903] |

## By how many focal moves the opponent had seen (pooled briefs)

Bin 0 is the opponent's opening proposal, before any focal move: the placebo.

| focal moves seen | proposals | spec, live opponent | spec, T1 | live − T1 |
|---|---|---|---|---|
| 0 | 417 | +0.0053 [-0.0368, +0.0497] | +0.0021 [-0.0334, +0.0377] | +0.0032 [-0.0190, +0.0266] |
| 1 | 762 | +0.0590 [+0.0073, +0.1102] | -0.0034 [-0.0379, +0.0315] | +0.0624 [+0.0352, +0.0902] |
| 2 | 621 | +0.0496 [+0.0148, +0.0838] | -0.0121 [-0.0411, +0.0171] | +0.0617 [+0.0378, +0.0857] |

## Controls: is it inference or reaction? (pooled; proposals after at least one focal offer)

`meet in the middle` = the attribute-wise midpoint of the focal side's last offer and the opponent's ideal deal (half levels toward the opponent), a reactive rule with no model of the focal side. `echo` = the focal side's own last offer. `best at same concession` = the deal best for the focal side's true ranking among those worth at least as much to the opponent as the live proposal (perfect targeting at the same cost to the opponent).

| rule | spec |
|---|---|
| live opponent | +0.0802 [+0.0333, +0.1258] |
| T1 in same state | -0.0116 [-0.0416, +0.0193] |
| meet in the middle | +0.1043 [+0.1012, +0.1073] |
| echo the focal offer | +0.1457 [+0.1413, +0.1497] |
| best at same concession | +0.1305 [+0.0751, +0.1852] |
| live − meet in the middle | -0.0241 [-0.0715, +0.0221] |

921 proposals. The live proposal equals the meet-in-the-middle deal in 0.085 of them and is within one attribute level of it in 0.501 (mean distance 1.60 levels; 27 deals, distance 0–6).

## After the opponent heard an answer (pooled, bins ≥ 1; descriptive means, no CI)

| opponent had asked and been answered | proposals | episodes | mean spec, live | mean spec, T1 |
|---|---|---|---|---|
| false | 1365 | 751 | +0.0465 | -0.0101 |
| true | 18 | 18 | +0.1385 | -0.0801 |

## Stated reasoning (heuristic text count, descriptive)

Share of the opponent's moves whose reasoning (the reply before `DECISION:`) names the counterpart, and the share with a sentence that names it together with a preference word (want, care, prefer, priority, value, matter, sticking point, shown, willing, ask, like).

| brief | moves | names counterpart | attributes a preference | example sentences |
|---|---|---|---|---|
| candid | 842 | 0.544 | 0.264 | “Warranty matters least to me, so I'm giving the buyer a long one, and I'm easing delivery to medium.” · “I'll try to improve delivery while giving the seller the high price they want.” · “The seller turned down a medium warranty, so I'll offer a short one to see if that is what they care about.” |
| guarded | 895 | 0.516 | 0.227 | “If they counter instead, their last offer would probably look like the current one, which I could still take.” · “A medium warranty is a real step toward the seller, who seems to want a short one.” · “I'll keep the short warranty the buyer asked for and ask for a high price in return.” |
| adversarial | 880 | 0.527 | 0.208 | “Slow delivery is a concession the seller seems to value, and this deal still clears my threshold (about 0.46 satisfaction).” · “Price is my top priority, so I want to know whether the seller is competing with me on it before I put a proposal on the table.” · “It also keeps the long warranty and slow delivery the seller asked for, so the only thing they give up is some price.” |
