"""Exact inference, claim parsing and scripted players (pure python)."""
import collections
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dethy_rl.expert import (ASSIGNMENTS, TYPES, Claim, ScriptedTeam, mafia_posterior,  # noqa: E402
                             parse_claims, simulate, told_mafia)

# 1. night-1 evidence from ALL five speakers (the Mafia fabricates a claim) is often ambiguous. An earlier version
#    of this check assumed only the four Cops spoke, so the silent fifth player was trivially the Mafia: a leak.
rng = random.Random(0)
n = hit = 0
for truth in ASSIGNMENTS:
    cops = [i for i in range(5) if truth[i] != "Mafia"]
    mafia = truth.index("Mafia")
    for targets in itertools.product(*([t for t in range(5) if t != c] for c in cops)):
        claims = [Claim(c, t, told_mafia(truth[c], truth[t] == "Mafia")) for c, t in zip(cops, targets)]
        claims.append(Claim(mafia, rng.choice([x for x in range(5) if x != mafia]), rng.random() < .5))
        post = mafia_posterior(claims, [], me=cops[0])
        top = max(post.values())
        leaders = [p for p, v in post.items() if v == top]
        n += 1
        hit += leaders == [mafia]
assert 0.15 < hit / n < 0.4, hit / n       # nowhere near certainty (the leaky version reported 1.0)

# 2. a Cop's own night-1 result alone is worth exactly chance, whatever its true type
for typ in TYPES:
    hit = tot = 0.0
    for truth in ASSIGNMENTS:
        me = truth.index(typ)
        for t in range(5):
            if t == me:
                continue
            post = mafia_posterior([Claim(me, t, told_mafia(typ, truth[t] == "Mafia"))], [], me)
            cands = [p for p in range(5) if p != me]
            best = max(post.get(p, 0) for p in cands)
            leaders = [p for p in cands if post.get(p, 0) == best]
            hit += (truth.index("Mafia") in leaders) / len(leaders); tot += 1
    assert abs(hit / tot - 0.25) < 1e-9, (typ, hit / tot)

# 2b. a Naive Cop's constant reports are still evidence (a game worked out by a colleague): they reveal its type,
#     which lets the other reports be interpreted.
M, N_ = True, False
game = [Claim(0, 4, M), Claim(1, 4, M), Claim(2, 4, M), Claim(3, 0, N_), Claim(4, 1, N_),      # day 1
        Claim(0, 1, N_), Claim(1, 0, N_), Claim(4, 0, N_)]                                       # day 2
dead = [2, 3]                                          # Player_2 eliminated, Player_3 killed
with_naive = mafia_posterior(game, dead, me=4)
without_naive = mafia_posterior([c for c in game if c.speaker != 4], dead, me=4)
assert set(with_naive) == {1}                          # Player_1 is the Mafia, with certainty
assert len(without_naive) > 1                          # without the Naive Cop's reports it is not settled

# 3. claim parser
c = lambda text, spk=1: [(x.target, x.told_mafia) for x in parse_claims(spk, text)]
assert c("On night 1 I investigated Player_2 and was told Mafia.") == [(2, True)]
assert c("On night 1 I investigated Player_2 and was told Not Mafia.") == [(2, False)]
assert c("My investigation of Player 3 came back as not the Mafia.") == [(3, False)]
assert c("I checked Player_4: Mafia! Also I looked at Player_0, he came back clean.") == [(4, True), (0, False)]
assert c("I think Player_3 is the Mafia.") == []                       # accusation, not a claim
assert c("Player_2 and Player_3 both seem fine, I investigated them and they are innocent.") == []  # two targets
assert c("I investigated Player_1 and it was Mafia.", spk=1) == []     # cannot investigate yourself
assert c("Nothing to report.") == []

# 4. scripted games (no model): sharing helps a lot, and a lone claimant is NOT automatically the Mafia
def town_rate(n=300, **kw):
    return sum(simulate(s, **kw).winner == "Town" for s in range(n)) / n
pooled = town_rate()
half = town_rate(share_prob=0.5)
silent = town_rate(share_prob=0.0)
own = town_rate(use_claims=False)
print(f"scripted Town win rates: all share={pooled:.2f}  half share={half:.2f}  silent={silent:.2f}  own notes={own:.2f}")
assert pooled > 0.8
assert pooled > half > silent + 0.1
assert silent < 0.6 and own < 0.6          # without sharing the Cops cannot beat ~45%
print("expert ok")
