"""Concurrent asyncio game lobbies and trajectory collection."""
import asyncio
from typing import Any, Dict, List

from dethy_rl.env import DethyEnv
from dethy_rl.vllm_worker import AgentRequest


async def run_lobby(lobby_id: int, worker, cfg, sem: asyncio.Semaphore) -> Dict[str, Any]:
    async with sem:
        env = DethyEnv(seed=cfg.rollout.seed * 1_000_003 + lobby_id, rewards=dict(cfg.env.rewards),
                       min_rounds=cfg.env.min_dialogue_rounds, max_rounds=cfg.env.max_dialogue_rounds)
        steps: List[Dict[str, Any]] = []
        last_step: Dict[int, Dict[str, Any]] = {}
        returns = {p: 0.0 for p in env.roles}
        vote_log = []  # (role, voted_for_mafia)

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
                    lobby_id=lobby_id, player_id=req.player_id, role=env.roles[req.player_id], phase=phase,
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
                rewards, _ = env.step_night(actions)
                add_reward(rewards)
            elif phase == "dialogue":
                for pid, text in actions.items():  # one speaker per turn, random order from env
                    env.step_dialogue(pid, text)
            else:
                vote_log += [(env.roles[p], t == env.mafia_id) for p, t in actions.items()
                             if env.roles[p] != "Mafia"]
                rewards, _ = env.step_vote(actions)
                add_reward(rewards)

        for s in last_step.values():
            s["done"] = True
        return {"steps": steps, "winner": env.winner, "returns": returns, "roles": env.roles, "votes": vote_log}


async def collect_trajectories(worker, cfg) -> Dict[str, Any]:
    sem = asyncio.Semaphore(cfg.rollout.max_concurrent_lobbies)
    games = await asyncio.gather(*(run_lobby(i, worker, cfg, sem) for i in range(cfg.training.games_per_epoch)))
    steps = [s for g in games for s in g["steps"]]
    all_returns = [r for g in games for r in g["returns"].values()]
    votes = [v for g in games for v in g["votes"]]

    def rate(roles):
        xs = [m for r, m in votes if r in roles]
        return sum(xs) / len(xs) if xs else float("nan")

    dlg = [len(s["action_ids"]) for s in steps if s["phase"] == "dialogue"]
    return dict(
        vote_mafia_rate_sane=rate({"Sane"}),            # should climb well above chance (~0.25)
        vote_mafia_rate_other_cops=rate({"Insane", "Naive", "Paranoid"}),
        avg_dialogue_tokens=sum(dlg) / max(len(dlg), 1),
        steps=steps,
        town_win_rate=sum(g["winner"] == "Town" for g in games) / len(games),
        avg_episode_reward=sum(all_returns) / len(all_returns),
        avg_game_len=len(steps) / len(games),
    )
