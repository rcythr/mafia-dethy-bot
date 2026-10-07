# Dethy Mafia self-play RL

Trains a ~3B language model (`meta-llama/Llama-3.2-3B-Instruct`) to play **Dethy Mafia** against copies of itself, using PPO with a LoRA adapter. vLLM generates the games; PyTorch + PEFT trains the adapter. Config is Hydra, tracking is MLflow.

> **Status:** the environment, rollout collector, GAE and PPO loss have CPU tests. The real-model path (`agent.py`, `vllm_worker.py`) has **never been run**: it was written without GPU access. Expect to fix version-specific vLLM/transformers issues on first launch (see [Known risks](#known-risks)).

## The game

5 players: 1 **Mafia** and 4 **Cops**. Each Cop has a hidden sanity that decides what investigations tell them:

| Sanity   | Investigation result            |
|----------|---------------------------------|
| Sane     | the truth                       |
| Insane   | the opposite of the truth       |
| Naive    | always "Not Mafia"              |
| Paranoid | always "Mafia"                  |

A game alternates:

1. **Night.** Every Cop investigates one player (privately learns "Mafia"/"Not Mafia" per their sanity). The Mafia kills one other player. Both happen simultaneously.
2. **Day dialogue.** A random **1-3 rounds**. Each round has a freshly shuffled speaking order, and players speak one at a time, so later speakers see earlier messages.
3. **Day vote.** Everyone votes for one living player; ties are broken randomly. That player is eliminated and their *non-Mafia-ness* is announced (their sanity stays hidden).

Town wins when the Mafia is eliminated. The Mafia wins when 2 or fewer players remain (parity). With a night kill every night, games last at most a couple of days.

## Layout

```
conf/config.yaml        Hydra config (every knob lives here)
main.py                 Hydra entry point
dethy_rl/
  env.py                game state machine, prompts, reward logic (no torch/GPU)
  vllm_worker.py        async vLLM wrapper, LoRA hot-swap, vote masking
  rollout.py            concurrent async game lobbies, trajectory collection
  agent.py              4-bit base + LoRA + value head, reference-policy logits
  train.py              GAE, PPO loss, LR schedule, MLflow, main loop
tests/
  test_rollout_env.py   CPU-only: env rules + rollout with a fake worker
  test_train_logic.py   CPU-only: GAE, advantages, PPO loss with a tiny fake agent
```

## Running it

```bash
pip install -r requirements.txt
python main.py                                   # train with defaults
python main.py training.epochs=10 env.think_tokens=128   # Hydra overrides
mlflow ui --backend-store-uri ./mlruns           # view runs
python tests/test_rollout_env.py && python tests/test_train_logic.py   # no GPU needed
```

For the DGX Spark (128 GB unified memory) you don't need the 12 GB workarounds:

```bash
python main.py model.load_in_4bit=false vllm.quantization=null vllm.gpu_memory_utilization=0.4
```

For a fast smoke test on a small GPU, use a smaller model:
`model.name=meta-llama/Llama-3.2-1B-Instruct model.load_in_4bit=false vllm.quantization=null`.

## How it works

Each epoch:

1. **Rollout.** `collect_trajectories` runs `games_per_epoch` games as concurrent asyncio tasks (`run_lobby`). Every call into vLLM is an `await`, so while one lobby waits for tokens the others progress, and vLLM's continuous batching keeps the GPU busy. Night and vote turns prompt all alive players with `asyncio.gather`; dialogue turns are one speaker at a time.
2. **Annotate.** The value head scores every decision, then GAE is computed **per player trajectory** (all of one player's decisions in one game).
3. **PPO update.** Clipped surrogate objective + value loss + KL-to-reference penalty - entropy bonus, one sequence at a time with gradient accumulation.
4. **Weight sync.** The LoRA adapter is saved and vLLM is given a new `LoRARequest` id, so the next rollout uses the updated policy. Epoch 0 runs with no adapter (a freshly initialised LoRA equals the base model).

### Prompts

```
[shared public transcript]
You are Player_i. Private Role: <role>. Private Notes: <investigation results>  Phase: <phase>. Action:
```

The public transcript comes **first** and is byte-identical for every player, so vLLM's prefix cache reuses its KV across all agents. Everything private goes after it.

### Actions

- **Dialogue:** free text, up to `model.max_tokens`, stops at a newline.
- **Night target / vote:** a **single token**: the digit of an alive player. A logits processor masks every other token to `-inf`. Digit token ids are derived from the tokenizer (`tokenizer.encode(str(p), add_special_tokens=False)[0]`), never hardcoded.
- vLLM returns the sampled `action_ids` and their `old_log_probs` (`logprobs=1`). Prompts are sent as token ids so training sees exactly what the policy saw.

### Rewards

Terminal outcomes dominate; the smaller terms make the signal denser. All values are in `env.rewards`.

| Event | Town | Mafia |
|---|---|---|
| Mafia lynched (game over) | +1.0 | -1.0 |
| Mafia reaches parity (game over) | -1.0 | +1.0 |
| Town player lynched | -0.2 | +0.2 |
| Night kill | -0.2 | +0.2 |
| "Heat": 0.3 x fraction of day votes on the Mafia | + | - |
| Cop votes for the Mafia | +0.2 (that Cop only) | |
| Cop votes for a non-Mafia | -0.05 (that Cop only) | |
| Cop investigates the real Mafia | +0.1 (that Cop only) | |

Rewards are credited to the player's most recent decision (a night kill lands on that night's step, vote rewards on the vote step). Dead players' last step keeps collecting the team outcome. GAE then spreads credit back over earlier dialogue steps.

### Optional private reasoning

`env.think_tokens > 0` adds a "thinking" stage before each night action and vote: the model writes up to N private tokens, then picks the digit. Both stages are trained steps; the decision prompt extends the thinking prompt token-for-token (so prefix caching still hits, and training sees exact ids). Thoughts are never added to the transcript or notes. Default is off.

## Decisions and why

**Training stack**
- **vLLM for acting, PyTorch+PEFT for training.** vLLM's continuous batching makes self-play generation fast; training needs gradients, which vLLM can't give.
- **LoRA, not full fine-tuning.** Small trainable state, and vLLM can hot-load adapters, which makes weight sync a save + reload instead of copying 3B weights.
- **4-bit on a 12 GB GPU.** A bf16 3B model (~6.4 GB) can't fit in 30% of 12 GB, so vLLM loads bitsandbytes 4-bit and the trainer uses NF4. The two quantisation paths can drift slightly, which adds noise to PPO ratios. That is a main reason to prefer the DGX Spark, where both run bf16.
- **vLLM starts before the trainer** so it can reserve its memory slice (`gpu_memory_utilization: 0.3`), leaving the rest for PyTorch.
- **One shared policy plays every role.** Plain self-play. The role is in the prompt.

**Making the PPO loss fit in memory**
- **Only action-token logits are computed** (`logits_to_keep`), not the full-sequence logits. A 128k vocab over a long prompt would not fit in 12 GB. The reference logits come from the same weights with the adapter disabled (`disable_adapter()`), so there's no second model.
- **Exact KL** to the reference over the full vocabulary at each action position, not a sampled estimate.
- Batch size 1 with gradient accumulation and gradient checkpointing.

**Vote masking and log-probs**
- The mask is applied during sampling, so `old_log_probs` must describe the *masked* distribution. The config requests `logprobs_mode: processed_logprobs` and training applies the same alive-digit mask to the logits (`masked_vote_logprobs`). If these disagree, vote ratios won't start near 1.
- Default `vote_mask_mode: logits_processor` uses vLLM's V0-style per-request callable. On engines without it, use `allowed_token_ids`.

**Learning aids**
- **Critic warm-up:** the first 2 epochs train only the value head, so early policy updates aren't driven by garbage advantages.
- **Value head is fp32 and zero-initialised.** bf16 + AdamW at lr 1e-4 barely moves.
- **Advantages normalised per team** (Mafia vs Cops): the lone Mafia has a different reward scale and less data.
- **LR schedule:** linear warmup then cosine decay to 10% of peak (set per epoch).
- **Diagnostics for "is it learning?":** `vote_mafia_rate_sane` (a Sane Cop knows the truth, so this should climb well above the ~0.25 chance level), `vote_mafia_rate_other_cops`, `explained_variance`, `clip_frac`, `approx_kl_old`, `kl`, `entropy`, dialogue/think lengths.

**Game design**
- Added the **Mafia night kill** and **1-3 sequential dialogue rounds with shuffled order** after the first version, so discussion has back-and-forth and the Mafia has agency.
- Cops may target themselves for investigation, and anyone may vote for themselves; this follows "alive players" in the spec literally.
- A lynch/kill that causes parity pays only the parity reward, not also the -0.2/+0.2 term.
- The individual Cop terms use hidden information a Cop doesn't have. They're small shaping; zero them if agents start gaming them.

**Considered but not built** (add when metrics call for them)
- **Adaptive KL controller:** only if `kl` spikes or collapses.
- **Periodic evaluation against a frozen baseline:** self-play win rates hover near a constant even as play improves; the vote metrics are the fixed yardstick for now.

## Known risks

- vLLM's API moves quickly. `logits_processors` per request, `logprobs_mode`, LoRA with bitsandbytes quantisation, and aarch64 (DGX Spark) wheels all need checking on first launch. Pin versions in `requirements.txt` once you have a working set.
- `value_warmup_epochs` (2) overlaps with `lr_warmup_epochs` (5), so the policy's effective LR warmup is shorter than it looks.
- LoRA dropout (0.05) is active during PPO forward passes but not in vLLM, adding small ratio noise.
- Thinking mode multiplies generated tokens per game; on the Spark (lower memory bandwidth) that costs wall-clock time, not memory.
- Nothing has been tuned. Reward magnitudes, KL coefficient and game counts are starting points.
