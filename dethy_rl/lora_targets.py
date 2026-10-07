"""Pick the exact modules to attach LoRA to (torch only, so it is easy to unit test)."""
from typing import List, Sequence

import torch.nn as nn

# Towers we never adapt even if their layers happen to be plain nn.Linear.
_SKIP = ("vision", "visual", "audio", "image")


def select_lora_targets(model: nn.Module, suffixes: Sequence[str]) -> List[str]:
    """Return `suffixes` unchanged when every module they match is a plain nn.Linear (the normal
    text-only case). Otherwise (multimodal models such as Gemma 4, whose vision/audio towers wrap
    their projections in custom classes PEFT can't adapt) return the full names of just the
    matching nn.Linear layers outside the vision/audio towers."""
    suffixes = list(suffixes)
    matches = [(n, m) for n, m in model.named_modules() if n.rsplit(".", 1)[-1] in suffixes]
    if all(isinstance(m, nn.Linear) for _, m in matches):
        return suffixes
    return [n for n, m in matches
            if isinstance(m, nn.Linear) and not any(k in n for k in _SKIP)]
