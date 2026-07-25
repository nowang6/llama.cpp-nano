from __future__ import annotations

from .base import (
    ModelBase, TextModel, ModelType, SentencePieceTokenTypes,
    logger, get_model_architecture, LazyTorchTensor,
)
from typing import Type

__all__ = [
    "ModelBase", "TextModel", "ModelType", "SentencePieceTokenTypes",
    "get_model_architecture", "LazyTorchTensor", "logger",
    "get_model_class", "print_registered_models",
]


# Only support Qwen3 models
TEXT_MODEL_MAP: dict[str, str] = {
    "Qwen3ForCausalLM": "qwen",
    "Qwen3Model": "qwen",
}


def get_model_class(name: str) -> Type[ModelBase]:
    """Return the model class for the given architecture name. Only Qwen3 is supported."""
    if name not in TEXT_MODEL_MAP:
        raise NotImplementedError(f"Architecture {name!r} not supported! Only Qwen3 models are supported.")
    module_name = TEXT_MODEL_MAP[name]
    __import__(f"conversion.{module_name}")
    return ModelBase._model_classes[ModelType.TEXT][name]


def print_registered_models() -> None:
    logger.error("Supported models:")
    for name in sorted(TEXT_MODEL_MAP.keys()):
        logger.error(f"  - {name}")
