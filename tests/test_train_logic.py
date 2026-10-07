"""CPU unit test of GAE / advantage / PPO loss with a tiny fake agent (skips if torch missing)."""
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("torch missing, skipped")
    raise SystemExit(0)
for m in ("mlflow", "omegaconf"):  # train.py imports these at module level; not needed here
    if m not in sys.modules:
        sys.modules[m] = types.SimpleNamespace(OmegaConf=None)
from dethy_rl.train import annotate_advantages, compute_gae, lr_at, ppo_update  # noqa: E402

# GAE: single terminal reward, gamma=lambda=1 => advantage = return - value
adv, ret = compute_gae([0, 0, 1.0], [0.1, 0.2, 0.3], 1.0, 1.0)
assert all(abs(r - 1.0) < 1e-9 for r in ret), ret

lcfg = NS(training=NS(learning_rate=1e-4, lr_warmup_epochs=5, lr_min_ratio=0.1, epochs=100))
lrs = [lr_at(e, lcfg) for e in range(100)]
assert lrs[0] < lrs[4] < lrs[5] and abs(lrs[5] - 1e-4) < 1e-12      # warmup rises to the peak
assert all(a >= b for a, b in zip(lrs[5:], lrs[6:]))               # then only decays
assert abs(lrs[-1] - 1e-5) < 1e-9                                  # ends at the floor

V = 12


class FakeAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 8)
        self.lm = nn.Linear(8, V)
        self.value_head = nn.Linear(8, 1)
        self.ref = nn.Linear(8, V)

    def trainable_parameters(self):
        return list(self.parameters())

    def forward(self, ids, num_action_tokens=0, value_pos=-1, compute_ref=True):
        h = self.emb(ids)
        logits = self.lm(h)[:, -num_action_tokens - 1:-1]
        ref = self.ref(h).detach()[:, -num_action_tokens - 1:-1] if compute_ref else None
        return logits, ref, self.value_head(h[:, value_pos]).squeeze(-1)

    def value_for_training(self, ids, value_pos):
        with torch.no_grad():
            h = self.emb(ids)[:, value_pos]
        return self.value_head(h).squeeze(-1)

    def values(self, ids, value_pos):
        with torch.no_grad():
            return self.value_head(self.emb(ids)[:, value_pos]).squeeze(-1)


cfg = NS(training=NS(gamma=0.99, gae_lambda=0.95, normalize_by_team=True, ppo_clip=0.2, kl_coef=0.1,
                     entropy_coef=0.01, value_coef=0.5, masked_vote_logprobs=True, ppo_epochs=2,
                     grad_accum_steps=4, max_grad_norm=1.0))
steps = []
for g in range(4):
    for p, role in enumerate(["Mafia", "Sane", "Naive"]):
        for t in range(3):
            steps.append(dict(lobby_id=g, player_id=p, role=role, prompt_ids=[1, 2, 3, 4],
                              action_ids=[5] if t == 2 else [6, 7], old_log_probs=[-2.0] * (1 if t == 2 else 2),
                              allowed_token_ids=[5, 6] if t == 2 else None,
                              reward=1.0 if t == 2 else 0.0, done=t == 2))
agent = FakeAgent()
stats = annotate_advantages(agent, steps, cfg)
assert "explained_variance" in stats
mafia = [s["advantage"] for s in steps if s["role"] == "Mafia"]
assert abs(sum(mafia) / len(mafia)) < 1e-5  # per-team normalised
opt = torch.optim.AdamW(agent.trainable_parameters(), lr=1e-2)
snap = {n: p.detach().clone() for n, p in agent.named_parameters()}
before = ppo_update(agent, opt, steps, cfg, train_policy=False)
assert "kl" not in before  # warm-up: value only
# warm-up must change the value head and nothing else
assert not torch.equal(snap["value_head.weight"], agent.value_head.weight)
for n in ("emb.weight", "lm.weight", "lm.bias"):
    assert torch.equal(snap[n], dict(agent.named_parameters())[n]), n
after = ppo_update(agent, opt, steps, cfg)
assert all(k in after for k in ("policy_loss", "kl", "entropy", "clip_frac", "approx_kl_old"))
print("train logic ok", {k: round(v, 3) for k, v in after.items()})
