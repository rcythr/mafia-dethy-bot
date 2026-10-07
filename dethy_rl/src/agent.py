"""PyTorch policy: 4-bit base + LoRA + value head, with a zero-copy reference policy."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


class DethyAgent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        quant = None
        if cfg.model.load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, quantization_config=quant, torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        base.config.use_cache = False
        lora = LoraConfig(
            r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
            target_modules=list(cfg.lora.target_modules), task_type="CAUSAL_LM",
        )
        self.policy_net = get_peft_model(base, lora)
        hidden = base.config.hidden_size
        self.value_head = nn.Linear(hidden, 1, dtype=torch.bfloat16).to(base.device)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(
        self, input_ids: torch.Tensor, num_action_tokens: int = 0, value_pos: int = -1,
        compute_ref: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        input_ids: (B, P+K) prompt followed by K action tokens.
        Returns logits that predict the last K tokens, shaped (B, K, V) (K = num_action_tokens;
        the full sequence if 0), reference logits from the same weights with the adapter disabled,
        and state values (B,) read from the final hidden state at `value_pos` (default: last token).
        Only the needed logit rows are materialised to keep the 128k-vocab head cheap.
        """
        keep = num_action_tokens + 1 if num_action_tokens else 0
        out = self.policy_net(input_ids=input_ids, output_hidden_states=True, logits_to_keep=keep)
        curr_logits = out.logits[:, :-1] if num_action_tokens else out.logits
        hidden = out.hidden_states[-1][:, value_pos]
        state_values = self.value_head(hidden.to(self.value_head.weight.dtype)).squeeze(-1)

        ref_logits = None
        if compute_ref:
            with torch.no_grad(), self.policy_net.disable_adapter():
                ref = self.policy_net(input_ids=input_ids, logits_to_keep=keep)
            ref_logits = ref.logits[:, :-1] if num_action_tokens else ref.logits
        return curr_logits, ref_logits, state_values

    @torch.no_grad()
    def values(self, input_ids: torch.Tensor, value_pos: int) -> torch.Tensor:
        out = self.policy_net(input_ids=input_ids, output_hidden_states=True, logits_to_keep=1)
        return self.value_head(out.hidden_states[-1][:, value_pos]).squeeze(-1)

    def save_adapter(self, path: str) -> None:
        self.policy_net.save_pretrained(path)
