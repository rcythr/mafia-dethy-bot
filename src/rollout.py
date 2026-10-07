"""Concurrent asyncio game lobbies and trajectory collection."""
import asyncio
from typing import Any, Dict, List

from src.env import DethyEnv
from src.vllm_worker import AgentRequest


async def run_lobby(lobby_id: int, worker, cfg, sem: asyncio.Semaphore) -> Dict[str, Any]:
    async with sem:
        env = DethyEnv(seed=cfg.rollout.seed * 1_000_003 + lobby_id)
        steps: List[Dict[str, Any]] = []
        last_step: Dict[int, Dict[str, Any]] = {}
        returns = {p: 0.0 for p in env.roles}

        def add_reward(rewards: Dict[int, float]) -> None:
            for p, r in rewards.items():
                returns[p] += r
                if p in last_step:  # credit the player's most recent decision
                    last_step[p]["reward"] += r

        while not env.done:
            phase = env.phase
            reqs = []
            for pid in env.acting_players():
                reqs.append(AgentRequest(pid, worker.encode(env.build_prompt(pid)), phase,
                                         env.allowed_targets(pid)))
            # every await yields to the event loop so other lobbies make progress
            resps = await worker.generate_agent_responses(reqs)

            actions: Dict[int, Any] = {}
            for req, resp in zip(reqs, resps):
                step = dict(
                    lobby_id=lobby_id, player_id=req.player_id, phase=phase,
                    prompt_ids=req.prompt_ids, action_ids=resp.action_ids,
                    old_log_probs=resp.old_log_probs, allowed_token_ids=resp.allowed_token_ids,
                    reward=0.0, done=False,
                )
                steps.append(step)
                last_step[req.player_id] = step
                if phase == "dialogue":
                    actions[req.player_id] = resp.text
                else:
                    tok = worker.tokenizer.decode(resp.action_ids).strip()
                    actions[req.player_id] = int(tok) if tok.isdigit() else -1

            if phase == "night":
                env.step_night(actions)
            elif phase == "dialogue":
                env.step_dialogue(actions)
            else:
                rewards, _ = env.step_vote(actions)
                add_reward(rewards)

        for s in last_step.values():
            s["done"] = True
        return {"steps": steps, "winner": env.winner, "returns": returns, "roles": env.roles}


async def collect_trajectories(worker, cfg) -> Dict[str, Any]:
    sem = asyncio.Semaphore(cfg.rollout.max_concurrent_lobbies)
    games = await asyncio.gather(*(run_lobby(i, worker, cfg, sem) for i in range(cfg.training.games_per_epoch)))
    steps = [s for g in games for s in g["steps"]]
    all_returns = [r for g in games for r in g["returns"].values()]
    return dict(
        steps=steps,
        town_win_rate=sum(g["winner"] == "Town" for g in games) / len(games),
        avg_episode_reward=sum(all_returns) / len(all_returns),
        avg_game_len=len(steps) / len(games),
    )
