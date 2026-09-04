"""D32-1216-R12 decoder-only model."""

from .config import ModelConfig
from .model import CausalLMOutput, DecoderLM

__all__ = ["CausalLMOutput", "DecoderLM", "ModelConfig"]
