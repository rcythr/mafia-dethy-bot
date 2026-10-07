"""PyTorch policy: bf16 base + LoRA + value head, with a zero-copy reference policy."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from dethy_rl.lora_targets import select_lora_targets


class DethyAgent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
        if cfg.model.get("gradient_checkpointing", True):  # saves memory, costs ~1/3 extra compute
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
        base.config.use_cache = False
        targets = select_lora_targets(base, cfg.lora.target_modules)
        print(f"LoRA targets: {len(targets)} entries")
        lora = LoraConfig(
            r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
            target_modules=targets, task_type="CAUSAL_LM",
        )
        self.policy_net = get_peft_model(base, lora)
        hidden = base.config.get_text_config().hidden_size  # multimodal configs keep it under text_config
        # fp32 (bf16 AdamW updates at lr 1e-4 barely move the weights); zero-init => V=0 at start
        self.value_head = nn.Linear(hidden, 1, dtype=torch.float32).to(base.device)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

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
        state_values = self.value_head(hidden.float()).squeeze(-1)

        ref_logits = None
        if compute_ref:
            with torch.no_grad(), self.policy_net.disable_adapter():
                ref = self.policy_net(input_ids=input_ids, logits_to_keep=keep)
            ref_logits = ref.logits[:, :-1] if num_action_tokens else ref.logits
        return curr_logits, ref_logits, state_values

    @torch.no_grad()
    def values(self, input_ids: torch.Tensor, value_pos: int) -> torch.Tensor:
        out = self.policy_net(input_ids=input_ids, output_hidden_states=True, logits_to_keep=1)
        return self.value_head(out.hidden_states[-1][:, value_pos].float()).squeeze(-1)

    def value_for_training(self, input_ids: torch.Tensor, value_pos: int) -> torch.Tensor:
        """Value prediction whose gradient reaches only the value head (critic warm-up): the
        backbone and LoRA run without grad, so the policy is not touched."""
        with torch.no_grad():
            out = self.policy_net(input_ids=input_ids, output_hidden_states=True, logits_to_keep=1)
            hidden = out.hidden_states[-1][:, value_pos].float()
        return self.value_head(hidden).squeeze(-1)

    def save_adapter(self, path: str) -> None:
        """LoRA adapter (loaded by vLLM) plus the value head (needed to resume training)."""
        self.policy_net.save_pretrained(path)
        torch.save(self.value_head.state_dict(), f"{path}/value_head.pt")
