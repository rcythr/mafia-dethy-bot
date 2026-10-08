"""Evaluate a trained LoRA adapter against the base model with fixed seeds.

    python evaluate.py eval.adapter=adapters/<run>/epoch_19 eval.games=200

Use the same model/env overrides as the training run (for example env.rules_style). Four matchups are
played on the SAME role assignments, so the comparison is paired:

  base_vs_base                  everyone is the base model (the noise floor / baseline)
  trained_cops_vs_base_mafia    Cops use the adapter, the Mafia is the base model   -> did the Cops improve?
  base_cops_vs_trained_mafia    Cops are the base model, the Mafia uses the adapter -> did the Mafia improve?
  trained_vs_trained            everyone uses the adapter (what training self-play sees)

Without eval.adapter only base_vs_base runs. Results print as a table and are saved as JSON.
"""
import asyncio
import json
import time

import hydra
from omegaconf import DictConfig, open_dict

from dethy_rl.evalstats import two_proportion_p, wilson_interval

MATCHUPS = {
    "base_vs_base": lambda side: False,
    "trained_cops_vs_base_mafia": lambda side: side == "Cop",
    "base_cops_vs_trained_mafia": lambda side: side == "Mafia",
    "trained_vs_trained": lambda side: True,
}


def summarize(batch) -> dict:
    n, k = batch["num_games"], batch["town_wins"]
    lo, hi = wilson_interval(k, n)
    return dict(
        games=n, town_wins=k, town_win_rate=k / n, town_win_ci95=[lo, hi],
        avg_reward_cops=batch["avg_reward_cops"], avg_reward_mafia=batch["avg_reward_mafia"],
        vote_mafia_rate_sane=batch["vote_mafia_rate_sane"],
        vote_mafia_rate_other_cops=batch["vote_mafia_rate_other_cops"],
        vote_nolynch_rate=batch["vote_nolynch_rate"], steps_per_game=batch["avg_game_len"],
    )


def render_table(results: dict) -> str:
    base = results.get("base_vs_base")
    head = f"{'matchup':30s} {'Town win rate (95% CI)':26s} {'vs base':>9s} {'Sane vote':>10s} {'other Cops':>11s} {'no-lynch':>9s}"
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        lo, hi = r["town_win_ci95"]
        p = ""
        if base is not None and name != "base_vs_base":
            p = f"p={two_proportion_p(r['town_wins'], r['games'], base['town_wins'], base['games']):.3f}"
        lines.append(f"{name:30s} {r['town_win_rate']:.3f} ({lo:.3f}-{hi:.3f})".ljust(57)
                     + f" {p:>9s} {r['vote_mafia_rate_sane']:>10.3f} {r['vote_mafia_rate_other_cops']:>11.3f}"
                       f" {r['vote_nolynch_rate']:>9.3f}")
    return "\n".join(lines)


async def run_eval(cfg: DictConfig) -> dict:
    from dethy_rl.rollout import collect_trajectories
    from dethy_rl.vllm_worker import VllmWorker

    ec = cfg.eval
    with open_dict(cfg):
        cfg.training.games_per_epoch = ec.games
        cfg.tracing.enabled = False
    worker = VllmWorker(cfg)
    if ec.adapter:
        worker.set_lora(ec.adapter, lora_id=1)

    results = {}
    for name in ec.matchups:
        if name not in MATCHUPS:
            raise SystemExit(f"unknown matchup {name!r}; choose from {list(MATCHUPS)}")
        if name != "base_vs_base" and not ec.adapter:
            print(f"skipping {name}: no eval.adapter given")
            continue
        t0 = time.time()
        # same seed_epoch for every matchup => identical role assignments (paired comparison)
        batch = await collect_trajectories(worker, cfg, epoch=ec.seed_epoch, adapter_for=MATCHUPS[name])
        results[name] = summarize(batch)
        print(f"[{name}] {ec.games} games in {time.time() - t0:.0f}s")

    print("\n" + render_table(results))
    out = {"adapter": ec.adapter, "games": ec.games, "seed_epoch": ec.seed_epoch, "results": results}
    with open(ec.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {ec.out}")
    return out


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    asyncio.run(run_eval(cfg))


if __name__ == "__main__":
    main()
