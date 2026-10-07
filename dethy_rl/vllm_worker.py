"""Async vLLM wrapper with LoRA hot-swapping and alive-player vote masking."""
import asyncio
import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


class VllmVoteLogitsProcessor:
    """Per-request logits processor: -inf everywhere except the allowed single-token IDs."""

    def __init__(self, allowed_token_ids: Sequence[int]):
        self.allowed = list(allowed_token_ids)
        self._mask = None

    def __call__(self, token_ids: List[int], logits):
        import torch

        if self._mask is None or self._mask.device != logits.device or self._mask.shape != logits.shape:
            self._mask = torch.full_like(logits, float("-inf"))
            self._mask[self.allowed] = 0.0
        return logits + self._mask


@dataclass
class AgentRequest:
    player_id: int
    prompt_ids: List[int]
    phase: str                      # "night" | "dialogue" | "vote" | "think"
    allowed_players: List[int] = field(default_factory=list)


@dataclass
class AgentResponse:
    player_id: int
    action_ids: List[int]
    old_log_probs: List[float]
    text: str
    allowed_token_ids: Optional[List[int]] = None  # set for constrained phases


class VllmWorker:
    def __init__(self, cfg):
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        from transformers import AutoTokenizer

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
        kwargs = dict(
            model=cfg.model.name,
            enable_lora=True,
            max_lora_rank=max(16, cfg.lora.r),
            max_loras=1,
            gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
            max_model_len=cfg.model.max_model_len,
            max_num_seqs=cfg.vllm.max_num_seqs,
            enforce_eager=cfg.vllm.enforce_eager,
            enable_prefix_caching=True,   # shared-transcript-first prompts => prefix KV reuse
            dtype="bfloat16",
        )
        if cfg.vllm.get("logprobs_mode"):
            kwargs["logprobs_mode"] = cfg.vllm.logprobs_mode
        try:
            self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**kwargs))
        except TypeError:  # older vLLM without logprobs_mode
            kwargs.pop("logprobs_mode", None)
            self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**kwargs))
        self.lora_request = None  # epoch 0: zero-initialised LoRA == base model

    # ------------------------------------------------------------ weight sync
    def set_lora(self, adapter_path: str, lora_id: int) -> None:
        """Point all subsequent requests at a freshly saved adapter (new id => no stale cache)."""
        from vllm.lora.request import LoRARequest

        self.lora_request = LoRARequest(f"dethy_{lora_id}", lora_id, os.path.abspath(adapter_path))

    # ----------------------------------------------------------------- tokens
    def digit_token_ids(self, players: Sequence[int]) -> List[int]:
        return [self.tokenizer.encode(str(p), add_special_tokens=False)[0] for p in players]

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=True)

    def encode_suffix(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    # ------------------------------------------------------------- generation
    def _sampling_params(self, req: AgentRequest):
        from vllm import SamplingParams

        common = dict(temperature=self.cfg.rollout.temperature, top_p=1.0, top_k=-1, logprobs=1)
        if req.phase == "dialogue":
            return SamplingParams(max_tokens=self.cfg.model.max_tokens, stop=["\n"], **common), None
        if req.phase == "think":
            return SamplingParams(max_tokens=self.cfg.env.think_tokens, stop=["</think>"], **common), None
        allowed = self.digit_token_ids(req.allowed_players)
        if self.cfg.vllm.vote_mask_mode == "logits_processor":
            return SamplingParams(
                max_tokens=1, logits_processors=[VllmVoteLogitsProcessor(allowed)], **common
            ), allowed
        return SamplingParams(max_tokens=1, allowed_token_ids=allowed, **common), allowed

    async def _generate_one(self, req: AgentRequest) -> AgentResponse:
        params, allowed = self._sampling_params(req)
        final = None
        async for out in self.engine.generate(
            {"prompt_token_ids": req.prompt_ids},
            params,
            request_id=uuid.uuid4().hex,
            lora_request=self.lora_request,
        ):
            final = out
        comp = final.outputs[0]
        ids = list(comp.token_ids)
        lps = [step[tid].logprob for step, tid in zip(comp.logprobs, ids)]
        return AgentResponse(req.player_id, ids, lps, comp.text, allowed)

    async def generate_agent_responses(self, requests: Sequence[AgentRequest]) -> List[AgentResponse]:
        """Prompt all alive agents concurrently; vLLM continuously batches them."""
        return list(await asyncio.gather(*(self._generate_one(r) for r in requests)))
