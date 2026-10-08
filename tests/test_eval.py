"""evaluate.py end to end with a fake vLLM worker: the right side uses the adapter in each matchup."""
import asyncio
import collections
import json
import os
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from hydra import compose, initialize_config_dir
except ImportError as e:
    print(f"{e.name} missing, skipped")
    raise SystemExit(0)

import dethy_rl.vllm_worker as worker_mod  # noqa: E402
from dethy_rl.vllm_worker import AgentResponse  # noqa: E402
import evaluate  # noqa: E402

FIRST_NIGHT = collections.defaultdict(list)   # matchup -> adapter-flag counts of 5-player night calls
CURRENT = {"name": None}


class FakeTok:
    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakeWorker:
    tokenizer = FakeTok()

    def __init__(self, cfg):
        self.adapter = None

    def set_lora(self, path, lora_id):
        self.adapter = path

    def prompt_ids(self, env, pid, think=False):
        return [ord(c) % 100 for c in env.build_prompt(pid, think)[-8:]]

    def think_suffix_ids(self):
        return [7]

    async def generate_agent_responses(self, reqs):
        await asyncio.sleep(0)
        if reqs[0].phase == "night" and len(reqs) == 5:
            FIRST_NIGHT[CURRENT["name"]].append(sum(r.adapter for r in reqs))
        out = []
        for r in reqs:
            if r.phase in ("dialogue", "think"):
                out.append(AgentResponse(r.player_id, [104], [-1.0], "hi"))
            else:
                t = random.choice(r.allowed_players)
                out.append(AgentResponse(r.player_id, [ord(str(t))], [-.5], str(t),
                                         [ord(str(p)) for p in r.allowed_players]))
        return out


worker_mod.VllmWorker = FakeWorker

# record which matchup is running
_orig_collect = None
import dethy_rl.rollout as rollout_mod  # noqa: E402
_orig_collect = rollout_mod.collect_trajectories


async def tagged_collect(worker, cfg, trace=False, epoch=0, adapter_for=None):
    for name, fn in evaluate.MATCHUPS.items():
        if adapter_for is fn:
            CURRENT["name"] = name
    return await _orig_collect(worker, cfg, trace=trace, epoch=epoch, adapter_for=adapter_for)


rollout_mod.collect_trajectories = tagged_collect

with tempfile.TemporaryDirectory() as tmp:
    out_file = os.path.join(tmp, "res.json")
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["eval.games=24", f"eval.out={out_file}",
                                                      "eval.adapter=/fake/epoch_1", "rollout.max_concurrent_lobbies=24"])
    res = asyncio.run(evaluate.run_eval(cfg))
    saved = json.load(open(out_file))
    assert set(saved["results"]) == set(evaluate.MATCHUPS)
    for name, r in saved["results"].items():
        assert r["games"] == 24 and 0 <= r["town_win_ci95"][0] <= r["town_win_rate"] <= r["town_win_ci95"][1] <= 1
    expect = {"base_vs_base": 0, "trained_cops_vs_base_mafia": 4, "base_cops_vs_trained_mafia": 1,
              "trained_vs_trained": 5}
    for name, n_adapter in expect.items():
        assert set(FIRST_NIGHT[name]) == {n_adapter}, (name, FIRST_NIGHT[name])

    # no adapter: only the base-vs-base baseline runs
    FIRST_NIGHT.clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["eval.games=8", f"eval.out={out_file}"])
    res = asyncio.run(evaluate.run_eval(cfg))
    assert list(res["results"]) == ["base_vs_base"]
print("eval ok")
