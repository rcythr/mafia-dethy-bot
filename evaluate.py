"""Evaluate a trained LoRA adapter against the base model with fixed seeds.

    python evaluate.py eval.adapter=adapters/<run>/epoch_19 eval.games=200

Use the same model/env overrides as the training run (for example env.rules_style). Four matchups are
played on the SAME role assignments, so the comparison is paired:

  base_vs_base                  everyone is the base model (the noise floor / baseline)
  trained_cops_vs_base_mafia    Cops use the adapter, the Mafia is the base model   -> did the Cops improve?
  base_cops_vs_trained_mafia    Cops are the base model, the Mafia uses the adapter -> did the Mafia improve?
  trained_vs_trained            everyone uses the adapter (what training self-play sees)

Without eval.adapter only base_vs_base runs. Results print as a table and are saved as JSON.

eval.mode=scenarios instead plays ONE seat with the model and fills the other four with scripted players
(exact-inference Cops that report honestly, and a bluffing Mafia; see dethy_rl/expert.py):

  python evaluate.py eval.mode=scenarios eval.adapter=adapters/<run>/epoch_19

Five scenarios, one per role: the model as the Sane, Insane, Naive or Paranoid Cop, or as the Mafia, each for
the base model and the adapter. Because Cops cannot see their own type and the game is symmetric under flipping
every report, Sane = Insane and Naive = Paranoid as tests, so a gap between them measures how literally the
model reads the words "Mafia" / "Not Mafia". eval.mode=both runs everything.
"""
import asyncio
import json
import time

import hydra
from omegaconf import DictConfig, open_dict

from dethy_rl.evalstats import two_proportion_p, wilson_interval
from dethy_rl.expert import simulate

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
        vote_mafia_rate_cops=batch["vote_mafia_rate_cops"],
        vote_mafia_rate_sane=batch["vote_mafia_rate_sane"],
        vote_mafia_rate_other_cops=batch["vote_mafia_rate_other_cops"],
        vote_nolynch_rate=batch["vote_nolynch_rate"], steps_per_game=batch["avg_game_len"],
    )


def render_table(results: dict) -> str:
    base = results.get("base_vs_base")
    head = f"{'matchup':30s} {'Town win rate (95% CI)':26s} {'vs base':>9s} {'Cop votes->Mafia':>17s} {'no-lynch':>9s}"
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        lo, hi = r["town_win_ci95"]
        p = ""
        if base is not None and name != "base_vs_base":
            p = f"p={two_proportion_p(r['town_wins'], r['games'], base['town_wins'], base['games']):.3f}"
        lines.append(f"{name:30s} {r['town_win_rate']:.3f} ({lo:.3f}-{hi:.3f})".ljust(57)
                     + f" {p:>9s} {r['vote_mafia_rate_cops']:>17.3f} {r['vote_nolynch_rate']:>9.3f}")
    return "\n".join(lines)


def _frac(k, n):
    return k / n if n else float("nan")


def summarize_scenario(batch) -> dict:
    n, k = batch["num_games"], batch["town_wins"]
    return dict(
        games=n, town_wins=k, town_win_rate=k / n, town_win_ci95=list(wilson_interval(k, n)),
        votes=batch["llm_votes"], vote_hits=batch["llm_vote_hits"], expert_agrees=batch["llm_expert_agrees"],
        abstains=batch["llm_abstains"], claims=batch["llm_claims"], claims_true=batch["llm_claims_true"],
        notes=batch["llm_notes"], notes_disclosed=batch["llm_notes_disclosed"],
    )


def render_scenarios(results: dict, reference: dict) -> str:
    head = (f"{'scenario':22s}{'Town win (95% CI)':26s}{'vote->Mafia':>12s}{'= expert':>10s}"
            f"{'claims true':>13s}{'notes told':>12s}{'abstain':>9s}")
    lines = [head, "-" * len(head)]
    for key, r in results.items():
        lo, hi = r["town_win_ci95"]
        lines.append(f"{key:22s}{r['town_win_rate']:.3f} ({lo:.3f}-{hi:.3f})".ljust(48)
                     + f"{_frac(r['vote_hits'], r['votes']):>12.3f}{_frac(r['expert_agrees'], r['votes']):>10.3f}"
                       f"{_frac(r['claims_true'], r['claims']):>13.3f}{_frac(r['notes_disclosed'], r['notes']):>12.3f}"
                       f"{_frac(r['abstains'], r['votes']):>9.3f}")
    lines.append("")
    lines.append(f"scripted reference (all five seats scripted): Cop scenarios Town win {reference['cops']:.3f}, "
                 f"Mafia scenario {reference['mafia']:.3f}")
    # symmetry check: Sane vs Insane and Naive vs Paranoid are equivalent tests
    for variant in sorted({k.split("/")[1] for k in results}):
        for a, b in (("Sane", "Insane"), ("Naive", "Paranoid")):
            ra, rb = results.get(f"{a}/{variant}"), results.get(f"{b}/{variant}")
            if ra and rb and ra["votes"] and rb["votes"]:
                p = two_proportion_p(ra["vote_hits"], ra["votes"], rb["vote_hits"], rb["votes"])
                lines.append(f"[{variant}] vote->Mafia {a} {_frac(ra['vote_hits'], ra['votes']):.3f} vs {b} "
                             f"{_frac(rb['vote_hits'], rb['votes']):.3f}  (p={p:.3f}; a real gap = reading the "
                             "words literally)")
    return "\n".join(lines)


async def run_scenarios(worker, cfg, ec) -> dict:
    from dethy_rl.rollout import collect_trajectories

    with open_dict(cfg):
        cfg.training.games_per_epoch = ec.scenario_games
    results = {}
    for role in ec.scenario_roles:
        share = ec.mafia_scenario_share_prob if role == "Mafia" else ec.cop_share_prob
        for variant in ec.scenario_variants:
            if variant == "trained" and not ec.adapter:
                print(f"skipping {role}/trained: no eval.adapter given")
                continue
            t0 = time.time()
            batch = await collect_trajectories(
                worker, cfg, epoch=ec.seed_epoch,
                adapter_for=(lambda side, _t=(variant == "trained"): _t),
                scripted=dict(llm_role=role, share_prob=share, use_claims=True))
            results[f"{role}/{variant}"] = summarize_scenario(batch)
            print(f"[{role}/{variant}] {ec.scenario_games} games in {time.time() - t0:.0f}s")
    n = ec.scenario_games
    reference = dict(
        cops=sum(simulate(s, share_prob=ec.cop_share_prob).winner == "Town" for s in range(n)) / n,
        mafia=sum(simulate(s, share_prob=ec.mafia_scenario_share_prob).winner == "Town" for s in range(n)) / n)
    print("\n" + render_scenarios(results, reference))
    return dict(results=results, reference=reference)


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

    out = {"adapter": ec.adapter, "games": ec.games, "seed_epoch": ec.seed_epoch}
    if ec.mode in ("matchups", "both"):
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
        out["results"] = results
    if ec.mode in ("scenarios", "both"):
        out["scenarios"] = await run_scenarios(worker, cfg, ec)
    with open(ec.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {ec.out}")
    return out


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    asyncio.run(run_eval(cfg))


if __name__ == "__main__":
    main()
