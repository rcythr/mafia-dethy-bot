"""CPU-only smoke test: env + rollout with a fake worker (no torch/vLLM needed)."""
import asyncio
import random
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dethy_rl.rollout import collect_trajectories  # noqa: E402
from dethy_rl.vllm_worker import AgentResponse  # noqa: E402


class FakeTok:
    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakeWorker:
    tokenizer = FakeTok()

    def encode_suffix(self, text):
        return [7, 8]

    def prompt_ids(self, env, pid, think=False):
        return self.encode(env.build_prompt(pid, think))

    def think_suffix_ids(self):
        return [7, 8]

    def encode(self, text):
        return [ord(c) % 255 for c in text[-20:]]

    async def generate_agent_responses(self, reqs):
        await asyncio.sleep(0)
        out = []
        for r in reqs:
            if r.phase in ("dialogue", "think"):
                out.append(AgentResponse(r.player_id, [104, 105], [-1.0, -1.0], "hi"))
            else:
                t = random.choice(r.allowed_players)
                out.append(AgentResponse(r.player_id, [ord(str(t))], [-0.5], str(t), [ord(str(p)) for p in r.allowed_players]))
        return out


cfg = NS(rollout=NS(seed=1, max_concurrent_lobbies=8), training=NS(games_per_epoch=16),
         env=NS(rewards={}, min_dialogue_rounds=1, max_dialogue_rounds=3, think_tokens=0, allow_no_lynch=True, lynch_rule="plurality"))
b = asyncio.run(collect_trajectories(FakeWorker(), cfg))
assert b["steps"] and sum(s["done"] for s in b["steps"]) == 16 * 5
# thinking on: think steps precede each night/vote decision and the decision prompt extends the think prompt
cfg.env.think_tokens = 8
bt = asyncio.run(collect_trajectories(FakeWorker(), cfg))
by = {}
for s in bt["steps"]:
    by.setdefault((s["lobby_id"], s["player_id"]), []).append(s)
for traj in by.values():
    for a, b2 in zip(traj, traj[1:]):
        if a["phase"] == "think":
            assert b2["phase"] in ("night", "vote")
            assert b2["prompt_ids"][:len(a["prompt_ids"]) + len(a["action_ids"])] == a["prompt_ids"] + a["action_ids"]
assert sum(s["phase"] == "think" for s in bt["steps"]) > 0 and bt["avg_think_tokens"] == 2
cfg.env.think_tokens = 0
# different epochs must produce different role assignments for the same lobby ids
def roles_of(batch):
    return tuple(sorted({(s["lobby_id"], s["player_id"], s["role"]) for s in batch["steps"]}))
e0 = asyncio.run(collect_trajectories(FakeWorker(), cfg, epoch=0))
e0b = asyncio.run(collect_trajectories(FakeWorker(), cfg, epoch=0))
e1 = asyncio.run(collect_trajectories(FakeWorker(), cfg, epoch=1))
assert roles_of(e0) == roles_of(e0b) and roles_of(e0) != roles_of(e1)
# spectator game logs: one per game, with the cast, per-stage events and an outro
assert len(b["game_logs"]) == 16
for md in b["game_logs"]:
    assert md.startswith("# Game ") and "## Cast" in md and "## Outro" in md
    assert "| True role |" in md and "→" in md and ("won." in md)
    assert md.count("| Player_") >= 5
print(len(b["steps"]), b["town_win_rate"], b["avg_episode_reward"])


# --- env rule checks: night kill, 1-3 dialogue rounds, random order per round
from dethy_rl.env import DethyEnv  # noqa: E402

rounds_seen, orders, kills = set(), set(), 0
for seed in range(300):
    env = DethyEnv(seed=seed)
    while not env.done:
        if env.phase == "night":
            n = len(env.alive)
            env.step_night({p: random.choice(env.allowed_targets(p)) for p in env.alive})
            assert len(env.alive) == n - 1
            kills += 1
        elif env.phase == "dialogue":
            rounds_seen.add(env.num_rounds)
            seen = []
            while env.phase == "dialogue":
                p = env.acting_players()[0]
                seen.append(p)
                env.step_dialogue(p, "x")
            assert len(seen) == env.num_rounds * len(env.alive)
            orders.add(tuple(seen[:len(env.alive)]))
        else:
            env.step_vote({p: random.choice(env.alive) for p in env.alive})
    assert env.winner in ("Town", "Mafia")
assert rounds_seen == {1, 2, 3} and len(orders) > 5 and kills > 0
print("env checks ok")


# --- no-lynch: abstaining keeps everyone alive and goes to the next night; games still terminate
from dethy_rl.env import NO_LYNCH  # noqa: E402

e = DethyEnv(seed=5)
e.step_night({p: random.choice(e.allowed_targets(p)) for p in e.alive})
while e.phase == "dialogue":
    e.step_dialogue(e.acting_players()[0], "x")
assert NO_LYNCH in e.allowed_targets(e.alive[0])
alive_before = list(e.alive)
rewards, done = e.step_vote({p: NO_LYNCH for p in e.alive})
assert not done and e.alive == alive_before and e.phase == "night" and e.day == 2
assert "No one was eliminated" in e.public_transcript()
# abstaining earns neither the Cop vote bonus nor the penalty, and the old (no-option) rules still work
off = DethyEnv(seed=5, allow_no_lynch=False)
off.step_night({p: random.choice(off.allowed_targets(p)) for p in off.alive})
assert all(NO_LYNCH not in off.allowed_targets(p) for p in off.alive)
assert "or 9" not in off.build_prompt(off.alive[0])

lengths, outcomes = {}, {}
for seed in range(1000):
    g = DethyEnv(seed=seed)
    days = 0
    while not g.done:
        if g.phase == "night":
            g.step_night({p: random.choice(g.allowed_targets(p)) for p in g.alive})
        elif g.phase == "dialogue":
            g.step_dialogue(g.acting_players()[0], "x")
        else:
            days += 1
            g.step_vote({p: random.choice(g.allowed_targets(p)) for p in g.alive})
        assert days < 10
    lengths[days] = lengths.get(days, 0) + 1
    outcomes[g.winner] = outcomes.get(g.winner, 0) + 1
assert max(lengths) >= 2, lengths   # games now sometimes run past one vote
print("no-lynch ok", dict(sorted(lengths.items())), outcomes)


# --- plurality rule: most votes wins; a tie for first eliminates no one; majority mode is stricter
def day_one(seed, **kw):
    g = DethyEnv(seed=seed, **kw)
    g.step_night({p: random.choice(g.allowed_targets(p)) for p in g.alive})
    while g.phase == "dialogue":
        g.step_dialogue(g.acting_players()[0], "x")
    return g

def vote_result(votes_fn, **kw):
    g = day_one(8, **kw)
    a, b, c, d = g.alive
    before = list(g.alive)
    g.step_vote(votes_fn(a, b, c, d))
    return g, before

g, before = vote_result(lambda a, b, c, d: {a: b, b: c, c: d, d: a})    # 1-1-1-1 tie for first
assert g.alive == before and g.phase == "night" and "no player got enough votes" in g.public_transcript()
g, before = vote_result(lambda a, b, c, d: {a: d, b: d, c: a, d: a})    # 2-2 tie
assert g.alive == before
g, before = vote_result(lambda a, b, c, d: {a: d, b: d, c: b, d: a})    # 2-1-1: plurality eliminates d
assert len(g.alive) == 3
g, before = vote_result(lambda a, b, c, d: {a: d, b: NO_LYNCH, c: NO_LYNCH, d: a})   # 'no one' leads 2-1-1
assert g.alive == before
g, before = vote_result(lambda a, b, c, d: {a: d, b: d, c: d, d: a})    # clear winner
assert len(g.alive) == 3
g, before = vote_result(lambda a, b, c, d: {a: d, b: d, c: b, d: a}, lynch_rule="majority")  # 2 of 4 is not > half
assert g.alive == before
print("lynch rules ok")


# --- spectator log content
lg = DethyEnv(seed=21)
lg.step_night({p: random.choice(lg.allowed_targets(p)) for p in lg.alive})
while lg.phase == "dialogue":
    lg.step_dialogue(lg.acting_players()[0], "Player_1: \"hello\"")
lg.step_vote({p: NO_LYNCH for p in lg.alive})
md = lg.render_game_log("T")
assert "hello" in md and 'Player_1: "' not in md.replace("**Player_1", "")  # cleaned text, speaker label by the log
assert "nobody was eliminated (\"no one\" got the most votes)" in md
assert "The game did not finish." in md
print("spectator log ok")
