from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    """
    A drop-in replacement for nn.Linear that adds a low-rank residual (LoRA).

    Forward: y = x W^T + b + scale * ( (x A) B )

    Where A in R^{in_features x r}, B in R^{r x out_features}, and scale = alpha / r.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 0,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        assert r >= 0, "LoRA rank r must be >= 0"
        self.lora_r = int(r)
        self.lora_alpha = float(alpha)
        self.lora_dropout_p = float(dropout)
        self.lora_scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 0.0
        if self.lora_r > 0:
            # Using A: in->r, B: r->out
            self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.lora_r))
            self.lora_B = nn.Parameter(torch.zeros(self.lora_r, self.out_features))
            # Kaiming init for A, zeros for B as in common LoRA practice
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
            self.lora_dropout = nn.Dropout(self.lora_dropout_p) if self.lora_dropout_p > 0 else nn.Identity()
        else:
            # Placeholders to keep state_dict stable
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        if self.lora_r == 0 or self.lora_scaling == 0.0:
            return base_out
        x_d = self.lora_dropout(x)
        # (x @ A) @ B == x @ (A @ B)
        lora_out = x_d.matmul(self.lora_A).matmul(self.lora_B)
        return base_out + lora_out * self.lora_scaling


def _wrap_linear_with_lora(
    linear: nn.Linear,
    *,
    r: int,
    alpha: float,
    dropout: float,
    train_bias: Literal["none", "all", "lora_only"] = "none",
) -> LoRALinear:
    """
    Replace a nn.Linear with LoRALinear while preserving weights/bias and parameter names.
    """
    new_linear = LoRALinear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        r=r,
        alpha=alpha,
        dropout=dropout,
    )
    # Ensure the wrapped layer matches the original layer's device and dtype
    new_linear = new_linear.to(device=linear.weight.device, dtype=linear.weight.dtype)
    # Copy base params
    with torch.no_grad():
        new_linear.weight.copy_(linear.weight.data)
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias.data)

    # Freeze base by default; trainer will optimize only LoRA params
    new_linear.weight.requires_grad = False
    if new_linear.bias is not None:
        new_linear.bias.requires_grad = train_bias in ("all",)

    if new_linear.lora_A is not None:
        new_linear.lora_A.requires_grad = True
        new_linear.lora_B.requires_grad = True

    return new_linear


def apply_lora_to_model(
    model: nn.Module,
    *,
    target_modules: Sequence[str] = ("to_qkv", "to_q", "to_kv", "to_out"),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    train_bias: Literal["none", "all", "lora_only"] = "none",
) -> None:
    """
    Recursively apply LoRA to attention projection linears commonly named as
    'to_qkv', 'to_q', 'to_kv', 'to_out' under attention blocks.

    Args:
        model: root module to modify in-place
        target_modules: attribute names to wrap when present on a submodule
        r, alpha, dropout, train_bias: LoRA hyperparameters
    """
    targets = set(target_modules)

    def maybe_wrap(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            # Recurse first
            maybe_wrap(child)
            # Then inspect child for target attributes
            for attr in list(targets):
                if hasattr(child, attr):
                    linear = getattr(child, attr)
                    if isinstance(linear, nn.Linear) and not isinstance(linear, LoRALinear):
                        setattr(
                            child,
                            attr,
                            _wrap_linear_with_lora(linear, r=r, alpha=alpha, dropout=dropout, train_bias=train_bias),
                        )

    maybe_wrap(model)


def mark_only_lora_as_trainable(model: nn.Module, train_bias: bool = False) -> None:
    """
    Set requires_grad=False for all params except LoRA adapter params (and biases if requested).
    """
    for name, param in model.named_parameters():
        is_lora = ("lora_A" in name) or ("lora_B" in name)
        is_bias = name.endswith(".bias")
        param.requires_grad = bool(is_lora or (train_bias and is_bias))
