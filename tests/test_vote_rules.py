"""Exhaustive vote check: every possible vote pattern with 4 players alive (4^4 = 256, including 'no one')
must follow the configured lynch rule."""
import collections
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dethy_rl.env import NO_LYNCH, DethyEnv  # noqa: E402


def four_alive(rule):
    random.seed(123)  # same night-1 kill every time
    g = DethyEnv(seed=8, first_kill_night=1, lynch_rule=rule)
    g.step_night({p: random.choice(g.allowed_targets(p)) for p in g.acting_players()})
    while g.phase == "dialogue":
        g.step_dialogue(g.acting_players()[0], "x")
    return g


for rule, needed in (("plurality", 2), ("majority", 3)):
    players = list(four_alive(rule).alive)
    assert len(players) == 4
    options = {p: [q for q in players if q != p] + [NO_LYNCH] for p in players}
    eliminated_by_top = collections.Counter()
    for combo in itertools.product(*(options[p] for p in players)):
        g = four_alive(rule)
        g.step_vote(dict(zip(players, combo)))
        eliminated = len(g.alive) < 4
        counts = collections.Counter(combo)
        top = max(counts.values())
        leaders = [v for v, c in counts.items() if c == top]
        unique_player_leader = len(leaders) == 1 and leaders[0] != NO_LYNCH
        expected = unique_player_leader and top >= needed
        assert eliminated == expected, (rule, combo, eliminated, expected)
        if eliminated:
            eliminated_by_top[top] += 1
    print(rule, "ok; eliminations by vote count of the winner:", dict(eliminated_by_top))
print("vote rules ok")
