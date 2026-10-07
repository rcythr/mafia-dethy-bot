"""Checkpoint save/resume round trip with a tiny random Llama + real PEFT LoRA (no downloads)."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError as e:
    print(f"{e.name} missing, skipped")
    raise SystemExit(0)
from dethy_rl.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint  # noqa: E402
from dethy_rl.lora_targets import select_lora_targets  # noqa: E402


class TinyAgent(nn.Module):
    """Same attributes/save format as DethyAgent, on a tiny random model."""

    def __init__(self):
        super().__init__()
        base = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                            num_hidden_layers=2, num_attention_heads=4))
        targets = select_lora_targets(base, ["q_proj", "k_proj", "v_proj", "o_proj"])
        self.policy_net = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=targets,
                                                          task_type="CAUSAL_LM"))
        self.value_head = nn.Linear(32, 1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save_adapter(self, path):
        self.policy_net.save_pretrained(path)
        torch.save(self.value_head.state_dict(), f"{path}/value_head.pt")


def train_a_bit(agent, opt, steps=3):
    ids = torch.randint(0, 64, (1, 8))
    for _ in range(steps):
        out = agent.policy_net(input_ids=ids).logits.float().mean() + agent.value_head(torch.randn(1, 32)).sum()
        opt.zero_grad()
        out.backward()
        opt.step()


def params(agent):
    return {n: p.detach().clone() for n, p in agent.named_parameters() if p.requires_grad}


torch.manual_seed(0)
a = TinyAgent()
opt = torch.optim.AdamW(a.trainable_parameters(), lr=1e-2)
train_a_bit(a, opt)

with tempfile.TemporaryDirectory() as d:
    assert find_latest_checkpoint(d) is None
    save_checkpoint(a, opt, os.path.join(d, "epoch_0"), 0, "run-abc")
    train_a_bit(a, opt)
    save_checkpoint(a, opt, os.path.join(d, "epoch_1"), 1, "run-abc")
    # a crash mid-save leaves a directory without state.json: it must be ignored
    os.makedirs(os.path.join(d, "epoch_2"))
    Path(d, "epoch_2", "adapter_model.safetensors").write_bytes(b"partial")

    epoch, path, state = find_latest_checkpoint(d)
    assert epoch == 1 and state == {"epoch": 1, "run_id": "run-abc"}, (epoch, state)

    # resume into a freshly initialised agent: weights and optimizer state must match exactly
    torch.manual_seed(0)  # same random base weights as `a` (real runs load the same pretrained base)
    b = TinyAgent()
    opt_b = torch.optim.AdamW(b.trainable_parameters(), lr=1e-2)
    load_checkpoint(b, opt_b, path)
    pa, pb = params(a), params(b)
    assert pa.keys() == pb.keys() and all(torch.equal(pa[k], pb[k]) for k in pa)
    sa, sb = opt.state_dict()["state"], opt_b.state_dict()["state"]
    assert len(sa) == len(sb) > 0
    assert all(torch.equal(sa[k]["exp_avg"], sb[k]["exp_avg"]) for k in sa)

    # and training continues identically from the resumed state
    torch.manual_seed(1); train_a_bit(a, opt, 1)
    torch.manual_seed(1); train_a_bit(b, opt_b, 1)
    pa, pb = params(a), params(b)
    assert all(torch.allclose(pa[k], pb[k], atol=1e-6) for k in pa)
print("checkpoint resume ok")
