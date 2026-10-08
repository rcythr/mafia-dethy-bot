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

1. **Night.** Every Cop investigates one player (privately learns "Mafia"/"Not Mafia" per their sanity). **Nobody is killed on the first night** (the Mafia sleeps and is not prompted). From the second night on, the Mafia also kills one other player at the same time (`env.first_kill_night: 2`).
2. **Day dialogue.** A random **1-3 rounds**. Each round has a freshly shuffled speaking order, and players speak one at a time, so later speakers see earlier messages.
3. **Day vote.** Everyone votes for another living player, or for `9` = "no one". A player is eliminated only with **more than half of the living players' votes** (3 of 4, 2 of 3, 3 of 5; `env.lynch_rule: majority`); if the votes are split, tied, or most players abstain, nobody is eliminated and the next night begins. An eliminated player's *non-Mafia-ness* is announced (their sanity stays hidden). `env.allow_no_lynch=false` removes the `9` option; `env.lynch_rule=plurality` is a looser variant where the unique leader wins and a tie for first elects no one.

Town wins when the Mafia is eliminated. The Mafia wins when 2 or fewer players remain (parity). From night 2 the Mafia kills every night, so abstaining or splitting the vote only delays things. With random voting about 81% of games reach a third vote and Town wins about 15%, so Town has to coordinate on one suspect.

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
python tests/test_expert.py && python tests/test_scenarios.py && python tests/test_vote_rules.py
python tests/test_checkpoint.py && python tests/test_resume_loop.py     # resume tests (need peft, transformers, hydra, mlflow)
python tests/test_prompt.py [--out prompts.txt] [--model <hf-name>]    # check and print example prompts
```

Everything runs in bf16 (no quantisation). The target is a DGX Spark (128 GB unified memory); `vllm.gpu_memory_utilization` (default 0.3) is the share vLLM reserves and the trainer uses the rest. For a fast smoke test on a smaller GPU, use a smaller model: `model.name=meta-llama/Llama-3.2-1B-Instruct`.

## Spectator game logs

Each epoch, `training.game_logs_per_epoch` (default 4) full games are written to MLflow as Markdown under `games/epoch_NNN/game_KK.md` (`training.game_log_every` sets how often). They are for human review only; agents never see them. Each log has: a cast table (true role incl. sanity, what the agent was told, and fate), every night's investigations (marking results that were misleading) and kill, the dialogue by round, each vote with a ✔ for votes on the Mafia, the tally and result (with the reason if nobody was eliminated), and an outro with the winner, how the game ended, and each player's episode reward.

## Evaluating a checkpoint

Per-epoch MLflow metrics are noisy (32 games each), so use `evaluate.py` for a clean comparison. It plays fixed-seed games for four matchups on the same role assignments and prints win rates with 95% intervals and a p-value against the base-vs-base baseline:

```bash
python evaluate.py eval.adapter=adapters/<run>/epoch_19 eval.games=200   # use the same model/env overrides as training
```

| Matchup | Question it answers |
|---|---|
| `base_vs_base` | the baseline / noise floor |
| `trained_cops_vs_base_mafia` | did the Cops improve? |
| `base_cops_vs_trained_mafia` | did the Mafia improve? |
| `trained_vs_trained` | what self-play training sees |

Only the vLLM engine is loaded (no trainer), so it is quick: roughly 5 to 10 minutes per 200-game matchup. Results are saved to `eval_results.json` (`eval.out`).

### Scenario evaluation: one model seat, four scripted players

```bash
python evaluate.py eval.mode=scenarios eval.adapter=adapters/<run>/epoch_19    # or eval.mode=both
```

`dethy_rl/expert.py` contains exact inference (enumerating the 120 possible role assignments) and scripted players: Cops report their notes honestly and vote for the exact posterior's favourite; the Mafia bluffs and frames a Cop. In each scenario the model plays one role (Sane, Insane, Naive, Paranoid, or Mafia) and everyone else is scripted, for both the base model and the adapter. Reported per scenario: Town win rate, how often the model's vote hits the Mafia, how often it equals the exact reasoner's pick, how many of the claims it makes are true, how many of its own notes it discloses, and its abstention rate.

Things worth knowing when reading the numbers:

- Sane and Insane are equivalent tests (so are Naive and Paranoid): Cops cannot see their own type, and the game is unchanged if every report is flipped. A gap between them means the model is reading "Mafia" / "Not Mafia" literally. The table prints that comparison with a p-value.
- A Cop's own night-1 result is worth exactly chance (0.25) for every type, so the model can only beat chance on day 1 by using what other players say.
- Scripted reference (all five seats scripted, `eval.cop_share_prob=1`): Town wins about 88%. If no Cop shares its notes it is about 41%, with only own notes about 46%, and with random votes about 14%.
- The Mafia scenario uses `eval.mafia_scenario_share_prob` (0.5): with fully honest scripted Cops the Mafia would almost never win, which tells you little.
- Claim parsing is a best-effort text match; `claims true` is only as good as that parser.

## Speed and critic options

Measured on the DGX Spark (32 games/epoch, ~700 steps): rollout ~40 s, value pass ~260 s, policy update ~1,500 s. The trainer dominates, so:

- `model.gradient_checkpointing` is `auto` by default: it starts off (removes ~25% of update compute) and, if a step runs out of memory, switches checkpointing on and retries that step. Set `true` to force it on or `false` to never switch.
- `env.rules_style=compact` shrinks the rules prompt to ~40% of its size (about 600 fewer tokens per step, so roughly 25% cheaper steps). It also removes the worked example and most strategy hints, so evaluate it with `evaluate.py` before trusting it.
- `training.value_lr_mult` (default 10) gives the critic head its own, larger learning rate (the first run's `explained_variance` stayed at 0.1 to 0.4 with 1).

## Resuming a run

Every epoch saves a checkpoint (LoRA adapter, value head, optimizer state) to `adapters/<timestamp>/epoch_N/`; the run prints that directory at startup. If a run dies, continue it with:

```bash
python main.py <same overrides> training.resume_from=adapters/<timestamp>
```

To bound disk use, only the newest `training.keep_checkpoints` (3) epoch checkpoints are kept, plus every `artifact_every`-th epoch (10, 20, ...) as a milestone; set `keep_checkpoints=0` to keep everything. It reloads the newest *complete* checkpoint (a crash mid-save is ignored), points vLLM at that adapter, continues at the next epoch with the LR schedule intact, and keeps logging to the same MLflow run. Games for each epoch are seeded from the epoch number, so a resumed run plays fresh games. MLflow logging failures (e.g. the tracking server being briefly unreachable) print a warning and do not stop training.

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
- Added the **Mafia night kill** (from night 2; night 1 is quiet, as in Dethy) and **1-3 sequential dialogue rounds with shuffled order** after the first version, so discussion has back-and-forth and the Mafia has agency.
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
