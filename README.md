# Dethy Mafia self-play RL

Trains a ~3B language model (`meta-llama/Llama-3.2-3B-Instruct`) to play **Dethy Mafia** against copies of itself, using PPO with a LoRA adapter. vLLM generates the games; PyTorch + PEFT trains the adapter. Config is Hydra, tracking is MLflow.

> **Status:** the environment, rollout collector, GAE and PPO loss have CPU tests. The real-model path (`agent.py`, `vllm_worker.py`) has **never been run**: it was written without GPU access. Expect to fix version-specific vLLM/transformers issues on first launch (see [Known risks](#known-risks)).

## The game

5 players: 1 **Mafia** and 4 **Cops**. The four Cops have four different hidden sanity types, which decide what their investigations tell them. **Only the Mafia knows their true role: a Cop is told only that they are a Cop, not which type**, so a Cop can never be sure how far to trust their own results and has to cross-check with the other Cops:

| Sanity   | Investigation result            |
|----------|---------------------------------|
| Sane     | the truth                       |
| Insane   | the opposite of the truth       |
| Naive    | always "Not Mafia"              |
| Paranoid | always "Mafia"                  |

A game alternates:

1. **Night.** Every Cop investigates one player (privately learns "Mafia"/"Not Mafia" per their sanity). The Mafia kills one other player. Both happen simultaneously.
2. **Day dialogue.** A random **1-3 rounds**. Each round has a freshly shuffled speaking order, and players speak one at a time, so later speakers see earlier messages.
3. **Day vote.** Everyone votes for another living player, or for `9` = "no one". The player with the **most votes** is eliminated; if two or more tie for the most, or "no one" leads, nobody is eliminated and the next night begins (`env.lynch_rule: plurality`). An eliminated player's *non-Mafia-ness* is announced (their sanity stays hidden). `env.allow_no_lynch=false` removes the `9` option; `env.lynch_rule=majority` is a stricter variant that needs more than half of the living players.

Town wins when the Mafia is eliminated. The Mafia wins when 2 or fewer players remain (parity). The Mafia kills every night, so abstaining or tying only delays things: with random play about half of games run two days and Town wins about 17%.

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
  test_prompt.py        prompt invariants + prints example prompts (optionally with the real chat template)
```

## Running it

```bash
pip install -r requirements.txt
python main.py                                   # train with defaults
python main.py training.epochs=10 env.think_tokens=128   # Hydra overrides
mlflow ui --backend-store-uri ./mlruns           # view runs
python tests/test_rollout_env.py && python tests/test_train_logic.py   # no GPU needed
python tests/test_checkpoint.py && python tests/test_resume_loop.py     # resume tests (need peft, transformers, hydra, mlflow)
python tests/test_prompt.py [--out prompts.txt] [--model <hf-name>]    # check and print example prompts
```

Everything runs in bf16 (no quantisation). The target is a DGX Spark (128 GB unified memory); `vllm.gpu_memory_utilization` (default 0.3) is the share vLLM reserves and the trainer uses the rest. For a fast smoke test on a smaller GPU, use a smaller model: `model.name=meta-llama/Llama-3.2-1B-Instruct`.

## Resuming a run

Every epoch saves a checkpoint (LoRA adapter, value head, optimizer state) to `adapters/<timestamp>/epoch_N/`; the run prints that directory at startup. If a run dies, continue it with:

```bash
python main.py <same overrides> training.resume_from=adapters/<timestamp>
```

It reloads the newest *complete* checkpoint (a crash mid-save is ignored), points vLLM at that adapter, continues at the next epoch with the LR schedule intact, and keeps logging to the same MLflow run. Games for each epoch are seeded from the epoch number, so a resumed run plays fresh games. MLflow logging failures (e.g. the tracking server being briefly unreachable) print a warning and do not stop training.

## How it works

Each epoch:

1. **Rollout.** `collect_trajectories` runs `games_per_epoch` games as concurrent asyncio tasks (`run_lobby`). Every call into vLLM is an `await`, so while one lobby waits for tokens the others progress, and vLLM's continuous batching keeps the GPU busy. Night and vote turns prompt all alive players with `asyncio.gather`; dialogue turns are one speaker at a time.
2. **Annotate.** The value head scores every decision, then GAE is computed **per player trajectory** (all of one player's decisions in one game).
3. **PPO update.** Clipped surrogate objective + value loss + KL-to-reference penalty - entropy bonus, one sequence at a time with gradient accumulation.
4. **Weight sync.** The LoRA adapter is saved and vLLM is given a new `LoRARequest` id, so the next rollout uses the updated policy. Epoch 0 runs with no adapter (a freshly initialised LoRA equals the base model).

### Prompts

Prompts go through the model's **chat template** (`model.use_chat_template`): the rules are the system message and everything below is one user message. The date the Llama template embeds is pinned so the prefix stays identical across runs. Set it to false to feed raw text. Models with a thinking mode (Qwen3.x, Gemma 4) need it switched off, because we run our own scratchpad (`env.think_tokens`): it is already set in the config (`model.chat_template_kwargs.enable_thinking: false`); override it with `model.chat_template_kwargs.enable_thinking=true`, or add other keys with `++model.chat_template_kwargs.some_key=value`. The exact argument name depends on the model's template; check with `python tests/test_prompt.py --model <name> --chat-kwargs '{"enable_thinking": false}'`.

```
[shared public transcript, starting with the full rules]
Alive players: Player_0, ...

You are Player_i.
Private Role: <Mafia or Cop>.
Private Notes:
- <one investigation result per line>
Phase: <phase>.
Action:
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
- **No quantisation.** vLLM and the trainer both use bf16 weights, so the sampler and the trained policy are numerically close and PPO ratios start near 1. This assumes a large-memory GPU such as the DGX Spark; a 12 GB card can't hold vLLM's slice plus a bf16 trainer.
- **vLLM starts before the trainer** so it can reserve its memory slice (`gpu_memory_utilization: 0.3`), leaving the rest for PyTorch.
- **One shared policy plays every role.** Plain self-play. The role is in the prompt.

**Making the PPO loss fit in memory**
- **Only action-token logits are computed** (`logits_to_keep`), not the full-sequence logits. A 128k vocab over a long prompt makes full-sequence logits very large. The reference logits come from the same weights with the adapter disabled (`disable_adapter()`), so there's no second model.
- **Exact KL** to the reference over the full vocabulary at each action position, not a sampled estimate.
- Batch size 1 with gradient accumulation and gradient checkpointing.

**Vote masking and log-probs**
- The mask is applied during sampling, so `old_log_probs` must describe the *masked* distribution. The config requests `logprobs_mode: processed_logprobs` and training applies the same alive-digit mask to the logits (`masked_vote_logprobs`). If these disagree, vote ratios won't start near 1.
- Default `vote_mask_mode: allowed_token_ids` uses vLLM's built-in masking, which works on the V1 engine. `logits_processor` (our `VllmVoteLogitsProcessor`, a per-request callable) only works on the old V0 engine, so set `VLLM_USE_V1=0` yourself if you want it; current vLLM rejects it with `Unexpected keyword argument 'logits_processors'`.

**Learning aids**
- **Critic warm-up:** the first 2 epochs train only the value head, so early policy updates aren't driven by garbage advantages.
- **Value head is fp32 and zero-initialised.** bf16 + AdamW at lr 1e-4 barely moves.
- **Advantages normalised per team** (Mafia vs Cops): the lone Mafia has a different reward scale and less data.
- **LR schedule:** linear warmup then cosine decay to 10% of peak (set per epoch).
- **Diagnostics for "is it learning?":** `vote_mafia_rate_sane` (Cops of the truthful Sane type, who do not know that they are Sane; their results are reliable, so this should climb well above the ~0.25 chance level as the model learns which reports to trust), `vote_mafia_rate_other_cops`, `explained_variance`, `clip_frac`, `approx_kl_old`, `kl`, `entropy`, dialogue/think lengths.

**Game design**
- Added the **Mafia night kill** and **1-3 sequential dialogue rounds with shuffled order** after the first version, so discussion has back-and-forth and the Mafia has agency.
- Players cannot target themselves (kill, investigate or vote). The spec said "alive players", but early transcripts showed half the votes going to the voter themselves, which wastes the signal.
- A lynch/kill that causes parity pays only the parity reward, not also the -0.2/+0.2 term.
- The individual Cop terms use hidden information a Cop doesn't have. They're small shaping; zero them if agents start gaming them.

**Considered but not built** (add when metrics call for them)
- **Adaptive KL controller:** only if `kl` spikes or collapses.
- **Periodic evaluation against a frozen baseline:** self-play win rates hover near a constant even as play improves; the vote metrics are the fixed yardstick for now.

## Known risks

- vLLM's API moves quickly. `logits_processors` per request, `logprobs_mode`, LoRA adapter loading, and aarch64 (DGX Spark) wheels all need checking on first launch. Pin versions in `requirements.txt` once you have a working set.
- `value_warmup_epochs` (2) overlaps with `lr_warmup_epochs` (5), so the policy's effective LR warmup is shorter than it looks.
- LoRA dropout (0.05) is active during PPO forward passes but not in vLLM, adding small ratio noise.
- Thinking mode multiplies generated tokens per game; on the Spark (lower memory bandwidth) that costs wall-clock time, not memory.
- Nothing has been tuned. Reward magnitudes, KL coefficient and game counts are starting points.
