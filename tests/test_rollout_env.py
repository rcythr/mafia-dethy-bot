"""CPU-only smoke test: env + rollout with a fake worker (no torch/vLLM needed)."""
import asyncio
import random
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rollout import collect_trajectories  # noqa: E402
from src.vllm_worker import AgentResponse  # noqa: E402


class FakeTok:
    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakeWorker:
    tokenizer = FakeTok()

    def encode(self, text):
        return [ord(c) % 255 for c in text[-20:]]

    async def generate_agent_responses(self, reqs):
        await asyncio.sleep(0)
        out = []
        for r in reqs:
            if r.phase == "dialogue":
                out.append(AgentResponse(r.player_id, [104, 105], [-1.0, -1.0], "hi"))
            else:
                t = random.choice(r.allowed_players)
                out.append(AgentResponse(r.player_id, [ord(str(t))], [-0.5], str(t), [ord(str(p)) for p in r.allowed_players]))
        return out


cfg = NS(rollout=NS(seed=1, max_concurrent_lobbies=8), training=NS(games_per_epoch=16))
b = asyncio.run(collect_trajectories(FakeWorker(), cfg))
assert b["steps"] and sum(s["done"] for s in b["steps"]) == 16 * 5
print(len(b["steps"]), b["town_win_rate"], b["avg_episode_reward"])
