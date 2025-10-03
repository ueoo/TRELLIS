from typing import *

import torch
import torch.nn as nn

from ..modules.lora import apply_lora_to_model, mark_only_lora_as_trainable
from .sparse_structure_flow import SparseStructureFlowModel
from .structured_latent_flow import SLatFlowModel


class LoRASparseStructureFlowModel(SparseStructureFlowModel):
    def __init__(
        self,
        *args,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_train_bias: bool = False,
        lora_target_modules: Sequence[str] = ("to_qkv", "to_q", "to_kv", "to_out"),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Apply LoRA adapters to attention projections
        apply_lora_to_model(
            self,
            target_modules=lora_target_modules,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            train_bias="all" if lora_train_bias else "none",
        )

        # Freeze non-LoRA params (and optionally biases)
        mark_only_lora_as_trainable(self, train_bias=lora_train_bias)

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = False):
        # Allow loading base checkpoints without LoRA keys
        return super().load_state_dict(state_dict, strict=False)


class LoRASLatFlowModel(SLatFlowModel):
    def __init__(
        self,
        *args,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_train_bias: bool = False,
        lora_target_modules: Sequence[str] = ("to_qkv", "to_q", "to_kv", "to_out"),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        apply_lora_to_model(
            self,
            target_modules=lora_target_modules,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            train_bias="all" if lora_train_bias else "none",
        )

        mark_only_lora_as_trainable(self, train_bias=lora_train_bias)

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = False):
        return super().load_state_dict(state_dict, strict=False)
