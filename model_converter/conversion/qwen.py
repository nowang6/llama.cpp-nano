from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass

from .base import TextModel, gguf, logger, ModelBase


@ModelBase.register("Qwen3ForCausalLM", "Qwen3Model")
class Qwen3Model(TextModel):
    """Qwen3 model converter — simplified for Qwen3-0.6B only."""
    model_arch = gguf.MODEL_ARCH.QWEN3

    def set_vocab(self):
        # Qwen3 uses GPT-2 BPE tokenizer (vocab.json + merges.txt)
        self._set_vocab_gpt2()
