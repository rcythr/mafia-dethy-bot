"""One LLM seat + four scripted players: the right seat is the LLM, and its statistics are recorded."""
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
    calls = 0

    def prompt_ids(self, env, pid, think=False):
        return [ord(c) % 100 for c in env.build_prompt(pid, think)[-8:]]

    def think_suffix_ids(self):
        return [7]

    async def generate_agent_responses(self, reqs):
        await asyncio.sleep(0)
        FakeWorker.calls += len(reqs)
        out = []
        for r in reqs:
            if r.phase == "dialogue":
                text = "On night 1 I investigated Player_1 and was told Mafia."
                out.append(AgentResponse(r.player_id, [104], [-1.0], text))
            else:
                t = random.choice(r.allowed_players)
                out.append(AgentResponse(r.player_id, [ord(str(t))], [-.5], str(t),
                                         [ord(str(p)) for p in r.allowed_players]))
        return out


cfg = NS(rollout=NS(seed=3, max_concurrent_lobbies=16), training=NS(games_per_epoch=24),
         env=NS(rewards={}, min_dialogue_rounds=1, max_dialogue_rounds=3, think_tokens=0, allow_no_lynch=True,
                lynch_rule="majority"),
         tracing=NS(lobbies_per_epoch=0))

full = asyncio.run(collect_trajectories(FakeWorker(), cfg))
full_calls = FakeWorker.calls

for role in ("Sane", "Insane", "Naive", "Paranoid", "Mafia"):
    FakeWorker.calls = 0
    b = asyncio.run(collect_trajectories(FakeWorker(), cfg, scripted=dict(llm_role=role, share_prob=1.0)))
    assert b["steps"] and all(s["role"] == role for s in b["steps"]), role   # only the chosen seat is the LLM
    assert FakeWorker.calls < 0.5 * full_calls                               # the other four cost no LLM calls
    assert b["num_games"] == 24 and 0 <= b["town_wins"] <= 24
    if role == "Mafia":
        assert b["llm_votes"] == 0                                           # vote statistics are for Cop seats
    else:
        assert b["llm_votes"] > 0
        assert 0 <= b["llm_vote_hits"] <= b["llm_votes"] and 0 <= b["llm_expert_agrees"] <= b["llm_votes"]
        assert b["llm_claims"] > 0 and b["llm_claims_true"] <= b["llm_claims"]   # the canned claim is mostly false
        assert b["llm_notes"] >= 24 and b["llm_notes_disclosed"] <= b["llm_notes"]
print("scenarios ok")
