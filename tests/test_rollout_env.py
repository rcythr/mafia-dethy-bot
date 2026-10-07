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
         env=NS(rewards={}, min_dialogue_rounds=1, max_dialogue_rounds=3, think_tokens=0))
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
