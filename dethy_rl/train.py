"""PPO + GAE training loop with LoRA weight sync to the vLLM engine."""
import asyncio
import gc
import os
import random
from collections import defaultdict
from typing import Dict, List

import mlflow
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

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


def _ids(step, device):
    return torch.tensor([step["prompt_ids"] + step["action_ids"]], device=device)


def annotate_advantages(agent, steps: List[Dict], cfg) -> None:
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
    advs = torch.tensor([s["advantage"] for s in steps])
    mean, std = advs.mean(), advs.std(unbiased=False).clamp_min(1e-6)
    for s in steps:
        s["advantage"] = float((s["advantage"] - mean) / std)


def ppo_step_loss(agent, step: Dict, cfg):
    device = agent.value_head.weight.device
    K, P = len(step["action_ids"]), len(step["prompt_ids"])
    curr, ref, value = agent(_ids(step, device), num_action_tokens=K, value_pos=P - 1)
    curr, ref = curr[0].float(), ref[0].float()  # (K, V)
    if step["allowed_token_ids"] is not None and cfg.training.masked_vote_logprobs:
        keep = torch.zeros(curr.shape[-1], dtype=torch.bool, device=device)
        keep[step["allowed_token_ids"]] = True
        curr = curr.masked_fill(~keep, -1e9)
        ref = ref.masked_fill(~keep, -1e9)

    logp_all = F.log_softmax(curr, dim=-1)
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
    value_loss = F.mse_loss(value.float().squeeze(), torch.tensor(step["return"], device=device))

    total = (policy_loss + cfg.training.value_coef * value_loss
             + cfg.training.kl_coef * kl - cfg.training.entropy_coef * entropy)
    return total, dict(policy_loss=policy_loss.item(), value_loss=value_loss.item(),
                       kl=kl.item(), entropy=entropy.item(), total_loss=total.item())


def ppo_update(agent, optimizer, steps: List[Dict], cfg) -> Dict[str, float]:
    agent.train()
    accum = cfg.training.grad_accum_steps
    sums: Dict[str, float] = defaultdict(float)
    n = 0
    for _ in range(cfg.training.ppo_epochs):
        random.shuffle(steps)
        optimizer.zero_grad(set_to_none=True)
        for i, s in enumerate(steps, 1):
            loss, stats = ppo_step_loss(agent, s, cfg)
            (loss / accum).backward()
            for k, v in stats.items():
                sums[k] += v
            n += 1
            if i % accum == 0 or i == len(steps):
                torch.nn.utils.clip_grad_norm_(agent.trainable_parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    return {k: v / max(n, 1) for k, v in sums.items()}


async def run_training(cfg) -> None:
    from dethy_rl.agent import DethyAgent
    from dethy_rl.vllm_worker import VllmWorker

    # vLLM first: it reserves its GPU slice (gpu_memory_utilization), trainer takes the rest.
    worker = VllmWorker(cfg)
    agent = DethyAgent(cfg)
    optimizer = torch.optim.AdamW(agent.trainable_parameters(), lr=cfg.training.learning_rate)

    mlflow.set_tracking_uri(cfg.experiment.tracking_uri)
    mlflow.set_experiment(cfg.experiment.name)
    with mlflow.start_run(run_name=cfg.experiment.get("run_name")):
        # Log the fully resolved Hydra config as params and as a reproducible artifact.
        flat = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_dict(flat, "config.yaml")
        mlflow.log_params({f"{sec}.{k}": v for sec, d in flat.items() if isinstance(d, dict)
                           for k, v in d.items() if sec != "hydra"})
        for epoch in range(cfg.training.epochs):
            batch = await collect_trajectories(worker, cfg)
            steps = batch["steps"]
            annotate_advantages(agent, steps, cfg)
            stats = ppo_update(agent, optimizer, steps, cfg)
            stats.update(avg_episode_reward=batch["avg_episode_reward"],
                         town_win_rate=batch["town_win_rate"], avg_steps_per_game=batch["avg_game_len"])
            mlflow.log_metrics(stats, step=epoch)
            print(f"epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()))

            # Weight sync: save LoRA adapter, hand vLLM a new adapter id for the next rollout.
            path = os.path.join(cfg.paths.adapter_dir, f"epoch_{epoch}")
            agent.save_adapter(path)
            worker.set_lora(path, lora_id=epoch + 1)
            if (epoch + 1) % cfg.training.get("artifact_every", 10) == 0:
                mlflow.log_artifacts(path, artifact_path=f"adapters/epoch_{epoch}")
            del steps, batch
            gc.collect()
            torch.cuda.empty_cache()
