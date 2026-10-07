"""CPU test for LoRA target selection (needs torch only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch.nn as nn
except ImportError:
    print("torch missing, skipped")
    raise SystemExit(0)
from dethy_rl.lora_targets import select_lora_targets  # noqa: E402

SUF = ["q_proj", "k_proj", "v_proj", "o_proj"]


class Attn(nn.Module):
    def __init__(self, cls):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (cls() for _ in range(4))


class Clippable(nn.Module):  # stands in for Gemma4ClippableLinear: wraps a Linear, is not one
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)


class TextOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Attn(lambda: nn.Linear(4, 4)) for _ in range(2)])


class Multimodal(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = TextOnly()
        self.vision_tower = Attn(Clippable)
        self.audio_tower = Attn(Clippable)


# text-only models are untouched: the suffix list is returned as is
assert select_lora_targets(TextOnly(), SUF) == SUF
# multimodal: only the language model's Linear projections, by full name
names = select_lora_targets(Multimodal(), SUF)
assert len(names) == 8 and all(n.startswith("language_model.layers.") for n in names), names
print("lora targets ok")
