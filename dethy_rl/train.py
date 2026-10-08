"""PPO + GAE training loop with LoRA weight sync to the vLLM engine."""
import asyncio
import gc
import logging
import math
import os
import random
import shutil
import time
from collections import defaultdict
from typing import Dict, List

import mlflow
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from dethy_rl.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from dethy_rl.rollout import collect_trajectories


def compute_gae(rewards: List[float], values: List[float], gamma: float, lam: float):
    """delta_t = r_t + gamma*V_{t+1} - V_t ; A_t = delta_t + gamma*lam*A_{t+1} ; R_t = A_t + V_t.
    The trajectory is a single player's ordered decisions; V after the final step is 0."""
    T = len(rewards)
    adv, last = [0.0] * T, 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    returns = [a + v for a, v in zip(adv, values)]
    return adv, returns


def lr_at(epoch: int, cfg) -> float:
    """Linear warmup to the base LR, then cosine decay down to base * lr_min_ratio."""
    t, base = cfg.training, cfg.training.learning_rate
    if epoch < t.lr_warmup_epochs:
        return base * (epoch + 1) / (t.lr_warmup_epochs + 1)
    progress = (epoch - t.lr_warmup_epochs) / max(1, t.epochs - 1 - t.lr_warmup_epochs)
    floor = t.lr_min_ratio
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def _ids(step, device):
    return torch.tensor([step["prompt_ids"] + step["action_ids"]], device=device)


def annotate_advantages(agent, steps: List[Dict], cfg) -> Dict[str, float]:
    """Value pass (no grad) -> per-player GAE -> normalised advantages stored on each step."""
    device = agent.value_head.weight.device
    agent.eval()
    for s in steps:
        s["value"] = agent.values(_ids(s, device), len(s["prompt_ids"]) - 1).float().item()
    trajs = defaultdict(list)
    for s in steps:
        trajs[(s["lobby_id"], s["player_id"])].append(s)
    for traj in trajs.values():
        adv, ret = compute_gae([s["reward"] for s in traj], [s["value"] for s in traj],
                               cfg.training.gamma, cfg.training.gae_lambda)
        for s, a, r in zip(traj, adv, ret):
            s["advantage"], s["return"] = a, r
    rets = torch.tensor([s["return"] for s in steps])
    vals = torch.tensor([s["value"] for s in steps])
    ev = 1.0 - ((rets - vals).var() / rets.var().clamp_min(1e-8)).item()

    # Normalise advantages per team: the lone Mafia has a different reward scale and far less
    # data than the 4 Cops, so a pooled normalisation would drown its signal.
    groups = defaultdict(list)
    for s in steps:
        groups[(s["role"] == "Mafia") if cfg.training.normalize_by_team else 0].append(s)
    for g in groups.values():
        advs = torch.tensor([s["advantage"] for s in g])
        mean = advs.mean()
        std = advs.std(unbiased=False).clamp_min(1e-6) if len(g) > 1 else torch.tensor(1.0)
        for s in g:
            s["advantage"] = float((s["advantage"] - mean) / std)
    return dict(explained_variance=ev, avg_value=vals.mean().item())


def ppo_step_loss(agent, step: Dict, cfg, train_policy: bool = True):
    device = agent.value_head.weight.device
    K, P = len(step["action_ids"]), len(step["prompt_ids"])
    if not train_policy:  # critic warm-up: only the value head learns, LoRA stays untouched
        value = agent.value_for_training(_ids(step, device), P - 1)
        value_loss = F.mse_loss(value.float().squeeze(), torch.tensor(step["return"], device=device))
        return value_loss, dict(value_loss=value_loss.item(), total_loss=value_loss.item())
    curr, ref, value = agent(_ids(step, device), num_action_tokens=K, value_pos=P - 1)
    curr = curr[0].float()  # (K, V)
    ref = ref[0].float()
    if step["allowed_token_ids"] is not None and cfg.training.masked_vote_logprobs:
        keep = torch.zeros(curr.shape[-1], dtype=torch.bool, device=device)
        keep[step["allowed_token_ids"]] = True
        curr = curr.masked_fill(~keep, -1e9)
        ref = ref.masked_fill(~keep, -1e9)

    logp_all = F.log_softmax(curr, dim=-1)
    value_loss = F.mse_loss(value.float().squeeze(), torch.tensor(step["return"], device=device))
    ref_logp_all = F.log_softmax(ref, dim=-1)
    actions = torch.tensor(step["action_ids"], device=device)
    logp = logp_all.gather(-1, actions[:, None]).squeeze(-1)
    old_logp = torch.tensor(step["old_log_probs"], device=device)

    ratio = torch.exp((logp - old_logp).clamp(-20, 20))
    adv = step["advantage"]
    eps = cfg.training.ppo_clip
    policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv).mean()

    probs = logp_all.exp()
    kl = (probs * (logp_all - ref_logp_all)).sum(-1).mean()       # KL(pi_theta || pi_ref)
    entropy = -(probs * logp_all).sum(-1).mean()

    total = (policy_loss + cfg.training.value_coef * value_loss
             + cfg.training.kl_coef * kl - cfg.training.entropy_coef * entropy)
    return total, dict(policy_loss=policy_loss.item(), value_loss=value_loss.item(),
                       kl=kl.item(), entropy=entropy.item(), total_loss=total.item(),
                       clip_frac=((ratio - 1).abs() > eps).float().mean().item(),
                       approx_kl_old=(old_logp - logp).mean().item())


def ppo_update(agent, optimizer, steps: List[Dict], cfg, train_policy: bool = True) -> Dict[str, float]:
    agent.train()
    accum = cfg.training.grad_accum_steps
    sums: Dict[str, float] = defaultdict(float)
    n = 0
    for _ in range(cfg.training.ppo_epochs):
        random.shuffle(steps)
        optimizer.zero_grad(set_to_none=True)
        for i, s in enumerate(steps, 1):
            loss, stats = ppo_step_loss(agent, s, cfg, train_policy)
            (loss / accum).backward()
            for k, v in stats.items():
                sums[k] += v
            n += 1
            if i % accum == 0 or i == len(steps):
                torch.nn.utils.clip_grad_norm_(agent.trainable_parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    return {k: v / max(n, 1) for k, v in sums.items()}


def _safe(fn, *args, **kwargs) -> None:
    """MLflow logging must never kill an unattended run (e.g. tracking server briefly down)."""
    try:
        fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"warning: MLflow call {getattr(fn, '__name__', fn)} failed: {e}")


async def run_training(cfg) -> None:
    from dethy_rl.agent import DethyAgent
    from dethy_rl.vllm_worker import VllmWorker

    # vLLM first: it reserves its GPU slice (gpu_memory_utilization), trainer takes the rest.
    t0 = time.time()
    worker = VllmWorker(cfg)
    t1 = time.time()
    agent = DethyAgent(cfg)
    print(f"startup: vLLM engine {t1 - t0:.0f}s, PyTorch trainer {time.time() - t1:.0f}s")
    # Two parameter groups so the critic head can learn faster than the LoRA weights (the LR schedule
    # scales both through lr_mult).
    optimizer = torch.optim.AdamW([
        {"params": agent.policy_parameters(), "lr_mult": 1.0},
        {"params": agent.value_parameters(), "lr_mult": cfg.training.get("value_lr_mult", 1.0)},
    ], lr=cfg.training.learning_rate)

    # Each run writes to its own adapter dir; resume by pointing training.resume_from at it.
    adapter_dir = cfg.training.resume_from or cfg.paths.adapter_dir
    start_epoch, run_id = 0, None
    if cfg.training.resume_from:
        ckpt = find_latest_checkpoint(adapter_dir)
        if ckpt is None:
            raise SystemExit(f"No complete checkpoint found in {adapter_dir}")
        last_epoch, ckpt_path, state = ckpt
        load_checkpoint(agent, optimizer, ckpt_path)
        worker.set_lora(ckpt_path, lora_id=last_epoch + 1)
        start_epoch, run_id = last_epoch + 1, state.get("run_id")
        print(f"Resuming from {ckpt_path}: continuing at epoch {start_epoch}")
    print(f"Checkpoints: {adapter_dir}   (resume with: training.resume_from={adapter_dir})")

    # GB10 (unified memory) reports GPU memory/power as 'Not Supported'; utilisation still works
    logging.getLogger("mlflow.system_metrics.metrics.gpu_monitor").setLevel(logging.ERROR)
    mlflow.set_tracking_uri(cfg.experiment.tracking_uri)
    mlflow.set_experiment(cfg.experiment.name)
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else cfg.experiment.get("run_name"),
                          log_system_metrics=cfg.experiment.get("system_metrics", True)) as run:
        if run_id is None:  # fresh run: record the resolved config (a resumed run keeps its original)
            flat = OmegaConf.to_container(cfg, resolve=True)
            _safe(mlflow.log_dict, flat, "config.yaml")
            _safe(mlflow.log_params, {f"{sec}.{k}": v for sec, d in flat.items() if isinstance(d, dict)
                                      for k, v in d.items() if sec != "hydra"})
        for epoch in range(start_epoch, cfg.training.epochs):
            lr = lr_at(epoch, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr * group.get("lr_mult", 1.0)
            trace = cfg.tracing.enabled and epoch % cfg.tracing.every_n_epochs == 0
            t_a = time.time()
            batch = await collect_trajectories(worker, cfg, trace=trace, epoch=epoch)
            steps = batch["steps"]
            t_b = time.time()
            adv_stats = annotate_advantages(agent, steps, cfg)
            t_c = time.time()
            warmup = epoch < cfg.training.value_warmup_epochs  # fit the critic before trusting its advantages
            stats = ppo_update(agent, optimizer, steps, cfg, train_policy=not warmup)
            t_d = time.time()
            stats.update(adv_stats, learning_rate=lr, sec_rollout=t_b - t_a, sec_annotate=t_c - t_b,
                         sec_update=t_d - t_c, num_steps=len(steps))
            stats.update({k: batch[k] for k in (
                "avg_episode_reward", "town_win_rate", "vote_mafia_rate_sane",
                "vote_mafia_rate_other_cops", "vote_nolynch_rate", "avg_dialogue_tokens",
                "avg_think_tokens")}, avg_steps_per_game=batch["avg_game_len"])
            _safe(mlflow.log_metrics, stats, step=epoch)
            if epoch % cfg.training.get("game_log_every", 1) == 0:  # read these to see what the agents do
                for i, md in enumerate(batch["game_logs"][:cfg.training.get("game_logs_per_epoch", 4)]):
                    _safe(mlflow.log_text, md, f"games/epoch_{epoch:03d}/game_{i:02d}.md")
            print(f"epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()))

            # Weight sync: save LoRA adapter, hand vLLM a new adapter id for the next rollout.
            path = os.path.join(adapter_dir, f"epoch_{epoch}")
            save_checkpoint(agent, optimizer, path, epoch, run.info.run_id)
            worker.set_lora(path, lora_id=epoch + 1)
            # Keep disk use bounded on long runs: drop old per-epoch checkpoints, but keep the newest
            # `keep_checkpoints` and every artifact_every-th epoch as a milestone.
            keep = cfg.training.get("keep_checkpoints", 3)
            stale = epoch - keep
            if keep > 0 and stale >= 0 and (stale + 1) % cfg.training.get("artifact_every", 10) != 0:
                shutil.rmtree(os.path.join(adapter_dir, f"epoch_{stale}"), ignore_errors=True)
            if (epoch + 1) % cfg.training.get("artifact_every", 10) == 0 or epoch == cfg.training.epochs - 1:
                _safe(mlflow.log_artifacts, path, artifact_path=f"adapters/epoch_{epoch}")
            del steps, batch
            gc.collect()
            torch.cuda.empty_cache()
