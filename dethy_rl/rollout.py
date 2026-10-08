"""Concurrent asyncio game lobbies and trajectory collection."""
import asyncio
from typing import Any, Callable, Dict, List, Optional

from dethy_rl.env import NO_LYNCH, DethyEnv
from dethy_rl.tracing import span
from dethy_rl.vllm_worker import AgentRequest


async def run_lobby(lobby_id: int, worker, cfg, sem: asyncio.Semaphore, trace: bool = False,
                    epoch: int = 0, adapter_for: Optional[Callable[[str], bool]] = None) -> Dict[str, Any]:
    """adapter_for("Mafia" | "Cop") -> whether that side uses the LoRA adapter (default: everyone does).
    Evaluation uses it to pit a trained side against the base model."""
    async with sem:
        # epoch is part of the seed so every epoch sees fresh role assignments and draws
        env = DethyEnv(seed=(cfg.rollout.seed * 1_000_003 + epoch) * 100_003 + lobby_id, rewards=dict(cfg.env.rewards),
                       min_rounds=cfg.env.min_dialogue_rounds, max_rounds=cfg.env.max_dialogue_rounds,
                       allow_no_lynch=cfg.env.allow_no_lynch, lynch_rule=cfg.env.lynch_rule,
                       rules_style=cfg.env.rules_style if hasattr(cfg.env, "rules_style") else "full",
                       first_kill_night=cfg.env.first_kill_night if hasattr(cfg.env, "first_kill_night") else 2)

        def uses_adapter(pid: int) -> bool:
            return True if adapter_for is None else adapter_for("Mafia" if env.roles[pid] == "Mafia" else "Cop")
        steps: List[Dict[str, Any]] = []
        last_step: Dict[int, Dict[str, Any]] = {}
        returns = {p: 0.0 for p in env.roles}
        vote_log = []  # (role, voted_for_mafia, abstained)

        def add_reward(rewards: Dict[int, float]) -> None:
            for p, r in rewards.items():
                returns[p] += r
                if p in last_step:  # credit the player's most recent decision
                    last_step[p]["reward"] += r

        def record(req, resp, phase: str) -> None:
            step = dict(
                lobby_id=lobby_id, player_id=req.player_id, role=env.roles[req.player_id], phase=phase,
                prompt_ids=req.prompt_ids, action_ids=resp.action_ids,
                old_log_probs=resp.old_log_probs, allowed_token_ids=resp.allowed_token_ids,
                reward=0.0, done=False,
            )
            steps.append(step)
            last_step[req.player_id] = step

        think = cfg.env.think_tokens > 0
        while not env.done:
            phase = env.phase
            pids = env.acting_players()
            base_ids = {pid: worker.prompt_ids(env, pid) for pid in pids}
            if think and phase in ("night", "vote"):
                # Stage 1: private reasoning (trained like any other action). Stage 2's prompt extends
                # stage 1's ids exactly, so training sees what the policy saw and the prefix cache hits.
                treqs = [AgentRequest(pid, worker.prompt_ids(env, pid, think=True), "think", trace=trace,
                                      adapter=uses_adapter(pid))
                         for pid in pids]
                for treq, tresp in zip(treqs, await worker.generate_agent_responses(treqs)):
                    record(treq, tresp, "think")
                    base_ids[treq.player_id] = (treq.prompt_ids + tresp.action_ids
                                                + worker.think_suffix_ids())
            reqs = [AgentRequest(pid, base_ids[pid], phase, env.allowed_targets(pid), trace=trace,
                                adapter=uses_adapter(pid)) for pid in pids]
            # every await yields to the event loop so other lobbies make progress
            resps = await worker.generate_agent_responses(reqs)

            actions: Dict[int, Any] = {}
            for req, resp in zip(reqs, resps):
                record(req, resp, phase)
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
                vote_log += [(env.roles[p], t == env.mafia_id, t == NO_LYNCH) for p, t in actions.items()
                             if env.roles[p] != "Mafia"]
                rewards, _ = env.step_vote(actions)
                add_reward(rewards)

        for s in last_step.values():
            s["done"] = True
        return {"steps": steps, "winner": env.winner, "returns": returns, "roles": env.roles, "votes": vote_log,
            "game_log": env.render_game_log(f"Game {lobby_id} (epoch {epoch})", returns),
            "transcript": env.public_transcript() + "\nRoles: " + ", ".join(f"Player_{p}={r}" for p, r in env.roles.items())}


async def _traced_lobby(lobby_id: int, worker, cfg, sem, trace: bool, epoch: int, adapter_for=None) -> Dict[str, Any]:
    """One MLflow trace per traced game; every LLM call in it becomes a child span."""
    with span(f"lobby_{lobby_id}", "CHAIN", trace) as sp:
        out = await run_lobby(lobby_id, worker, cfg, sem, trace, epoch, adapter_for)
        if sp is not None:
            sp.set_outputs({"winner": out["winner"], "roles": out["roles"], "returns": out["returns"],
                            "transcript": out["transcript"]})
    return out


async def collect_trajectories(worker, cfg, trace: bool = False, epoch: int = 0,
                               adapter_for: Optional[Callable[[str], bool]] = None) -> Dict[str, Any]:
    sem = asyncio.Semaphore(cfg.rollout.max_concurrent_lobbies)
    n_traced = cfg.tracing.lobbies_per_epoch if trace else 0
    games = await asyncio.gather(*(_traced_lobby(i, worker, cfg, sem, i < n_traced, epoch, adapter_for)
                                   for i in range(cfg.training.games_per_epoch)))
    steps = [s for g in games for s in g["steps"]]
    all_returns = [r for g in games for r in g["returns"].values()]
    votes = [v for g in games for v in g["votes"]]

    def rate(roles, idx=1):
        xs = [v[idx] for v in votes if v[0] in roles]
        return sum(xs) / len(xs) if xs else float("nan")

    def side_reward(is_mafia: bool) -> float:
        xs = [r for g in games for p, r in g["returns"].items() if (g["roles"][p] == "Mafia") == is_mafia]
        return sum(xs) / len(xs)

    dlg = [len(s["action_ids"]) for s in steps if s["phase"] == "dialogue"]
    thk = [len(s["action_ids"]) for s in steps if s["phase"] == "think"]
    return dict(
        # Cops cannot tell their own type, so the symmetric number is the one that matters; the by-type
        # split below is a diagnostic only
        vote_mafia_rate_cops=rate({"Sane", "Insane", "Naive", "Paranoid"}),
        vote_mafia_rate_sane=rate({"Sane"}),            # should climb well above chance (~0.25)
        vote_mafia_rate_other_cops=rate({"Insane", "Naive", "Paranoid"}),
        vote_nolynch_rate=rate({"Sane", "Insane", "Naive", "Paranoid"}, idx=2),
        avg_dialogue_tokens=sum(dlg) / max(len(dlg), 1),
        avg_think_tokens=sum(thk) / max(len(thk), 1),
        steps=steps,
        game_logs=[g["game_log"] for g in games],
        town_win_rate=sum(g["winner"] == "Town" for g in games) / len(games),
        town_wins=sum(g["winner"] == "Town" for g in games),
        num_games=len(games),
        avg_reward_cops=side_reward(False),
        avg_reward_mafia=side_reward(True),
        num_cop_votes=len(votes),
        avg_episode_reward=sum(all_returns) / len(all_returns),
        avg_game_len=len(steps) / len(games),
    )
