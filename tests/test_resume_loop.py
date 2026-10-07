"""End-to-end loop test on CPU: fresh run -> simulated crash -> resume. Uses a tiny real PEFT model,
a fake vLLM worker and real MLflow (local store). Skips if the heavy deps are missing."""
import asyncio
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
try:
    import hydra
    import mlflow
    import torch
    import torch.nn as nn
    from hydra import compose, initialize_config_dir
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError as e:
    print(f"{e.name} missing, skipped")
    raise SystemExit(0)

import dethy_rl.agent as agent_mod  # noqa: E402
import dethy_rl.vllm_worker as worker_mod  # noqa: E402
from dethy_rl.lora_targets import select_lora_targets  # noqa: E402
from dethy_rl.train import run_training  # noqa: E402
from dethy_rl.vllm_worker import AgentResponse  # noqa: E402

V = 64


class FakeAgent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        torch.manual_seed(0)  # same "pretrained" base every time
        base = LlamaForCausalLM(LlamaConfig(vocab_size=V, hidden_size=32, intermediate_size=64,
                                            num_hidden_layers=2, num_attention_heads=4))
        targets = select_lora_targets(base, ["q_proj", "k_proj", "v_proj", "o_proj"])
        self.policy_net = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=targets,
                                                          task_type="CAUSAL_LM"))
        self.value_head = nn.Linear(32, 1)
        nn.init.zeros_(self.value_head.weight)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, ids, num_action_tokens=0, value_pos=-1, compute_ref=True):
        out = self.policy_net(input_ids=ids, output_hidden_states=True)
        logits = out.logits[:, -num_action_tokens - 1:-1]
        ref = None
        if compute_ref:
            with torch.no_grad(), self.policy_net.disable_adapter():
                ref = self.policy_net(input_ids=ids).logits[:, -num_action_tokens - 1:-1]
        return logits, ref, self.value_head(out.hidden_states[-1][:, value_pos].float()).squeeze(-1)

    def values(self, ids, value_pos):
        with torch.no_grad():
            h = self.policy_net(input_ids=ids, output_hidden_states=True).hidden_states[-1][:, value_pos]
            return self.value_head(h.float()).squeeze(-1)

    def value_for_training(self, ids, value_pos):
        with torch.no_grad():
            h = self.policy_net(input_ids=ids, output_hidden_states=True).hidden_states[-1][:, value_pos]
        return self.value_head(h.float()).squeeze(-1)

    def save_adapter(self, path):
        self.policy_net.save_pretrained(path)
        torch.save(self.value_head.state_dict(), f"{path}/value_head.pt")


CRASH_AT = {"epoch": None}


class FakeTok:
    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakeWorker:
    tokenizer = FakeTok()
    lora_ids = []

    def __init__(self, cfg):
        FakeWorker.lora_ids = []

    def set_lora(self, path, lora_id):
        FakeWorker.lora_ids.append(lora_id)

    def prompt_ids(self, env, pid, think=False):
        return [ord(c) % 60 for c in env.build_prompt(pid, think)[-12:]]

    def think_suffix_ids(self):
        return [7]

    async def generate_agent_responses(self, reqs):
        await asyncio.sleep(0)
        out = []
        for r in reqs:
            if r.phase in ("dialogue", "think"):
                out.append(AgentResponse(r.player_id, [10, 11], [-3.0, -3.0], "hi"))
            else:
                t = random.choice(r.allowed_players)
                out.append(AgentResponse(r.player_id, [ord(str(t))], [-1.5], str(t),
                                         [ord(str(p)) for p in r.allowed_players]))
        return out


agent_mod.DethyAgent = FakeAgent
worker_mod.VllmWorker = FakeWorker

import dethy_rl.train as train_mod  # noqa: E402
_orig_collect = train_mod.collect_trajectories


async def crashing_collect(worker, cfg, trace=False, epoch=0):
    if CRASH_AT["epoch"] == epoch:
        raise RuntimeError("simulated crash")
    return await _orig_collect(worker, cfg, trace=trace, epoch=epoch)


train_mod.collect_trajectories = crashing_collect


def make_cfg(tmp, extra=()):
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "conf"), version_base=None):
        return compose(config_name="config", overrides=[
            "training.epochs=4", "training.games_per_epoch=3", "training.value_warmup_epochs=1",
            "training.lr_warmup_epochs=1", "training.grad_accum_steps=4",
            "rollout.max_concurrent_lobbies=3", "tracing.enabled=false", "experiment.system_metrics=false",
            f"experiment.tracking_uri=file:{tmp}/mlruns", f"paths.adapter_dir={tmp}/adapters", *extra])


with tempfile.TemporaryDirectory() as tmp:
    # 1) fresh run that crashes while collecting epoch 2
    CRASH_AT["epoch"] = 2
    try:
        asyncio.run(run_training(make_cfg(tmp)))
        raise AssertionError("expected the simulated crash")
    except RuntimeError as e:
        assert "simulated crash" in str(e)
    done = sorted(p.name for p in Path(tmp, "adapters").iterdir())
    assert done == ["epoch_0", "epoch_1"], done
    assert all(Path(tmp, "adapters", d, "state.json").exists() for d in done)

    # 2) resume: must continue at epoch 2 in the same MLflow run and finish
    CRASH_AT["epoch"] = None
    asyncio.run(run_training(make_cfg(tmp, [f"training.resume_from={tmp}/adapters"])))
    assert FakeWorker.lora_ids[0] == 2, FakeWorker.lora_ids     # engine pointed at epoch_1's adapter first
    assert FakeWorker.lora_ids[-1] == 4                         # then epoch_2, epoch_3 -> ids 3, 4
    done = sorted(p.name for p in Path(tmp, "adapters").iterdir())
    assert done == ["epoch_0", "epoch_1", "epoch_2", "epoch_3"], done

    mlflow.set_tracking_uri(f"file:{tmp}/mlruns")
    runs = mlflow.search_runs(experiment_names=["dethy_mafia_ppo"], output_format="list")
    assert len(runs) == 1, f"resume should reuse the same MLflow run, got {len(runs)}"
    hist = mlflow.tracking.MlflowClient().get_metric_history(runs[0].info.run_id, "total_loss")
    assert sorted(m.step for m in hist) == [0, 1, 2, 3], [m.step for m in hist]
print("resume loop ok")
