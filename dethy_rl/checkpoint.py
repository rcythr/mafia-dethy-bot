"""Per-epoch checkpoints so a long run can resume after a crash.

Layout (one directory per epoch under the run's adapter dir):
    epoch_N/adapter_model.safetensors, adapter_config.json   LoRA adapter (also what vLLM loads)
    epoch_N/value_head.pt                                    critic head
    epoch_N/optimizer.pt                                     AdamW state
    epoch_N/state.json                                       written LAST; marks the checkpoint complete
"""
import json
import os
import re
from typing import Optional, Tuple

import torch


def save_checkpoint(agent, optimizer, path: str, epoch: int, run_id: Optional[str]) -> None:
    agent.save_adapter(path)  # adapter + value_head.pt
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    tmp = os.path.join(path, "state.json.tmp")
    with open(tmp, "w") as f:
        json.dump({"epoch": epoch, "run_id": run_id}, f)
    os.replace(tmp, os.path.join(path, "state.json"))  # atomic: a crash mid-save leaves no state.json


def find_latest_checkpoint(adapter_dir: str) -> Optional[Tuple[int, str, dict]]:
    """Newest COMPLETE checkpoint (has state.json) as (epoch, path, state), or None."""
    if not os.path.isdir(adapter_dir):
        return None
    best = None
    for name in os.listdir(adapter_dir):
        m = re.fullmatch(r"epoch_(\d+)", name)
        state_file = os.path.join(adapter_dir, name, "state.json")
        if m and os.path.isfile(state_file):
            epoch = int(m.group(1))
            if best is None or epoch > best[0]:
                with open(state_file) as f:
                    best = (epoch, os.path.join(adapter_dir, name), json.load(f))
    return best


def load_checkpoint(agent, optimizer, path: str) -> None:
    from peft import set_peft_model_state_dict

    st_file = os.path.join(path, "adapter_model.safetensors")
    if os.path.isfile(st_file):
        from safetensors.torch import load_file
        adapter_sd = load_file(st_file)
    else:
        adapter_sd = torch.load(os.path.join(path, "adapter_model.bin"), map_location="cpu")
    set_peft_model_state_dict(agent.policy_net, adapter_sd)
    device = agent.value_head.weight.device
    agent.value_head.load_state_dict(torch.load(os.path.join(path, "value_head.pt"), map_location=device))
    opt_state = torch.load(os.path.join(path, "optimizer.pt"), map_location="cpu")
    if len(opt_state["param_groups"]) != len(optimizer.param_groups):
        # e.g. a checkpoint from before the critic had its own parameter group: keep weights, restart Adam
        print("warning: optimizer layout changed since this checkpoint; starting with a fresh optimizer state")
        return
    current_mult = [g.get("lr_mult", 1.0) for g in optimizer.param_groups]
    optimizer.load_state_dict(opt_state)
    for g, m in zip(optimizer.param_groups, current_mult):  # the CURRENT config's multipliers win
        g["lr_mult"] = m
