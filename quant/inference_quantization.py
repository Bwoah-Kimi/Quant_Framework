import warnings
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantConfig:
    enabled: bool = False
    linear_quant_mode: str = "w_only"  # "w_only" or "wa"
    replace_linear_modules: bool = False
    enable_attention_quant: bool = False  # placeholder for future attention path

    # Weight quant params
    quant_type: str = "int"  # "int" or "fp"
    bit: int = 16
    w_group_size: int = -1  # block-wise quant group size; -1 means whole dim
    quant_dim: int = -1  # usually -1 for per-row grouping on linear weights
    strict_group_size: bool = True

    # Integer format controls
    clip_style: str = "sym"  # "sym" or "asym"

    # Float format controls (used when quant_type == "fp")
    e_bit: int = 4
    m_bit: int = 3

    # Scale precision controls (reuse Quant_Framework semantics)
    e8_scale: bool = False
    e8_scale_op: str = "ceil"  # "ceil", "floor", "round", "ocp"
    scale_quant: bool = False
    scale_quant_2: bool = False

    # Activation quant params (INT-only path for now)
    activation_enabled: bool = False
    act_bit: int = 16
    act_group_size: int = -1
    act_quant_dim: int = -1
    act_strict_group_size: bool = True
    act_clip_style: str = "sym"
    act_e8_scale: bool = False
    act_e8_scale_op: str = "ceil"
    act_scale_quant: bool = False
    act_scale_quant_2: bool = False

    # Attention quant params
    attn_quant_mode: str = "qkv"  # "qkv", "qkvo", "full"
    attn_bit: int = 16
    attn_group_size: int = -1
    attn_quant_dim: int = -1
    attn_strict_group_size: bool = True

    # Layer filtering
    include_modules: Optional[List[str]] = None
    exclude_modules: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.exclude_modules is None:
            self.exclude_modules = ["lm_head"]
        self.quant_type = str(self.quant_type).lower()
        self.linear_quant_mode = str(self.linear_quant_mode).lower()
        if self.activation_enabled and self.linear_quant_mode == "w_only":
            self.linear_quant_mode = "wa"
        if self.linear_quant_mode == "wa":
            self.replace_linear_modules = True
        self.attn_quant_mode = str(self.attn_quant_mode).lower()

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "QuantConfig":
        if cfg is None:
            return cls()
        cfg = dict(cfg)
        valid_keys = {f.name for f in fields(cls)}
        unknown_keys = sorted(k for k in cfg.keys() if k not in valid_keys)
        if unknown_keys:
            raise ValueError(f"Unknown quant config keys: {unknown_keys}")
        payload = {k: v for k, v in cfg.items() if k in valid_keys}
        return cls(**payload)

    def validate(self) -> None:
        if self.linear_quant_mode not in {"w_only", "wa"}:
            raise ValueError(
                f"linear_quant_mode must be 'w_only' or 'wa', got '{self.linear_quant_mode}'."
            )
        if self.quant_type not in {"int", "fp"}:
            raise ValueError(
                f"weight quant_type must be 'int' or 'fp', got '{self.quant_type}'."
            )
        if self.linear_quant_mode == "wa" and self.quant_type != "int":
            raise NotImplementedError(
                "Linear W+A quantization currently supports quant_type='int' only."
            )
        if self.linear_quant_mode == "wa" and self.act_bit >= 16:
            raise ValueError("linear_quant_mode='wa' requires act_bit < 16.")
        if self.bit < 2:
            raise ValueError(f"weight bit must be >= 2, got {self.bit}.")
        if self.w_group_size == 0:
            raise ValueError("weight w_group_size must be -1 or > 0.")
        if self.clip_style not in {"sym", "asym"}:
            raise ValueError(
                f"clip_style must be 'sym' or 'asym', got '{self.clip_style}'."
            )
        if self.e8_scale_op not in {"ceil", "floor", "round", "ocp"}:
            raise ValueError(
                f"e8_scale_op must be one of ['ceil', 'floor', 'round', 'ocp'], got '{self.e8_scale_op}'."
            )
        if self.e8_scale and self.scale_quant:
            raise ValueError("e8_scale and scale_quant cannot both be enabled.")
        if self.scale_quant and self.scale_quant_2:
            raise ValueError("scale_quant and scale_quant_2 cannot both be enabled.")
        if self.activation_enabled and self.act_bit < 2:
            raise ValueError(f"activation act_bit must be >= 2, got {self.act_bit}.")
        if self.act_group_size == 0:
            raise ValueError("activation act_group_size must be -1 or > 0.")
        if self.act_clip_style not in {"sym", "asym"}:
            raise ValueError(
                f"activation act_clip_style must be 'sym' or 'asym', got '{self.act_clip_style}'."
            )
        if self.act_e8_scale_op not in {"ceil", "floor", "round", "ocp"}:
            raise ValueError(
                "activation act_e8_scale_op must be one of ['ceil', 'floor', 'round', 'ocp'], "
                f"got '{self.act_e8_scale_op}'."
            )
        if self.act_e8_scale and self.act_scale_quant:
            raise ValueError("act_e8_scale and act_scale_quant cannot both be enabled.")
        if self.act_scale_quant and self.act_scale_quant_2:
            raise ValueError(
                "act_scale_quant and act_scale_quant_2 cannot both be enabled."
            )
        if self.attn_quant_mode not in {"qkv", "qkvo", "full"}:
            raise ValueError(
                f"attn_quant_mode must be one of ['qkv', 'qkvo', 'full'], got '{self.attn_quant_mode}'."
            )
        if self.attn_bit < 2:
            raise ValueError(f"attention attn_bit must be >= 2, got {self.attn_bit}.")
        if self.attn_group_size == 0:
            raise ValueError("attention attn_group_size must be -1 or > 0.")
        if self.include_modules is not None and not isinstance(
            self.include_modules, (list, tuple)
        ):
            raise ValueError(
                "include_modules must be a list/tuple of name patterns or null."
            )
        if self.exclude_modules is not None and not isinstance(
            self.exclude_modules, (list, tuple)
        ):
            raise ValueError(
                "exclude_modules must be a list/tuple of name patterns or null."
            )
        if self.quant_type == "fp" and self.bit < 16:
            expected = int(self.e_bit) + int(self.m_bit) + 1
            if int(self.bit) != expected:
                raise ValueError(
                    f"FP quant bit mismatch: bit={self.bit}, but e_bit+m_bit+1={expected}."
                )
        if (
            self.quant_type == "fp"
            and self.enable_attention_quant
            and self.attn_bit < 16
        ):
            expected = int(self.e_bit) + int(self.m_bit) + 1
            if int(self.attn_bit) != expected:
                raise ValueError(
                    f"Attention FP quant bit mismatch: attn_bit={self.attn_bit}, but e_bit+m_bit+1={expected}."
                )


def _resolve_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim = ndim + dim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"Invalid quant_dim={dim} for tensor with ndim={ndim}.")
    return dim


def _resolve_group_size(
    tensor: torch.Tensor,
    quant_dim: int,
    group_size: int,
    strict: bool,
    layer_name: str,
    tensor_kind: str,
) -> Optional[int]:
    dim_size = tensor.shape[quant_dim]
    resolved = int(group_size) if int(group_size) > 0 else int(dim_size)
    if dim_size % resolved != 0:
        msg = (
            f"Skipping {tensor_kind} quant for layer '{layer_name}': dim_size={dim_size} is not divisible by "
            f"group_size={resolved} (quant_dim={quant_dim})."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning)
        return None
    return resolved


def quantize_weight_tensor(
    weight: torch.Tensor,
    layer_name: str,
    cfg: QuantConfig,
    int_quant_fn: Callable,
    fp_quant_fn: Callable,
) -> Optional[torch.Tensor]:
    quant_dim = _resolve_dim(int(cfg.quant_dim), weight.ndim)
    resolved_group_size = _resolve_group_size(
        weight,
        quant_dim,
        int(cfg.w_group_size),
        bool(cfg.strict_group_size),
        layer_name,
        "weight",
    )
    if resolved_group_size is None:
        return None

    if int(cfg.bit) >= 16:
        return weight
    if cfg.quant_type == "int":
        return int_quant_fn(
            weight,
            bit=int(cfg.bit),
            dim=quant_dim,
            group_size=resolved_group_size,
            e8_scale=bool(cfg.e8_scale),
            e8_scale_op=cfg.e8_scale_op,
            clip_style=cfg.clip_style,
            scale_quant=bool(cfg.scale_quant),
            scale_quant_2=bool(cfg.scale_quant_2),
        )
    return fp_quant_fn(
        weight,
        bit=int(cfg.bit),
        e_bit=int(cfg.e_bit),
        m_bit=int(cfg.m_bit),
        dim=quant_dim,
        group_size=resolved_group_size,
        e8_scale=bool(cfg.e8_scale),
        e8_scale_op=cfg.e8_scale_op,
        scale_quant=bool(cfg.scale_quant),
        scale_quant_2=bool(cfg.scale_quant_2),
    )


def _quantize_activation_runtime(
    x: torch.Tensor,
    *,
    layer_name: str,
    int_quant_fn: Callable,
    act_bit: int,
    act_group_size: int,
    act_quant_dim: int,
    act_strict_group_size: bool,
    act_clip_style: str,
    act_e8_scale: bool,
    act_e8_scale_op: str,
    act_scale_quant: bool,
    act_scale_quant_2: bool,
) -> torch.Tensor:
    if act_bit >= 16:
        return x
    quant_dim = _resolve_dim(int(act_quant_dim), x.ndim)
    resolved_group_size = _resolve_group_size(
        x,
        quant_dim,
        int(act_group_size),
        bool(act_strict_group_size),
        layer_name,
        "activation",
    )
    if resolved_group_size is None:
        return x
    return int_quant_fn(
        x,
        bit=int(act_bit),
        dim=quant_dim,
        group_size=resolved_group_size,
        e8_scale=bool(act_e8_scale),
        e8_scale_op=act_e8_scale_op,
        clip_style=act_clip_style,
        scale_quant=bool(act_scale_quant),
        scale_quant_2=bool(act_scale_quant_2),
    )


class QuantLinearInference(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        dtype: torch.dtype,
        device: torch.device,
        layer_name: str,
        cfg: QuantConfig,
        int_quant_fn: Callable,
    ):
        super().__init__(
            in_features, out_features, bias=bias, dtype=dtype, device=device
        )
        self.layer_name = layer_name
        self.activation_enabled = bool(
            cfg.linear_quant_mode == "wa" or cfg.activation_enabled
        )
        self.act_bit = int(cfg.act_bit)
        self.act_group_size = int(cfg.act_group_size)
        self.act_quant_dim = int(cfg.act_quant_dim)
        self.act_strict_group_size = bool(cfg.act_strict_group_size)
        self.act_clip_style = cfg.act_clip_style
        self.act_e8_scale = bool(cfg.act_e8_scale)
        self.act_e8_scale_op = cfg.act_e8_scale_op
        self.act_scale_quant = bool(cfg.act_scale_quant)
        self.act_scale_quant_2 = bool(cfg.act_scale_quant_2)
        self._int_quant_fn = int_quant_fn

    @classmethod
    def from_linear(
        cls,
        original_module: nn.Linear,
        layer_name: str,
        cfg: QuantConfig,
        int_quant_fn: Callable,
        fp_quant_fn: Callable,
    ) -> Optional["QuantLinearInference"]:
        quantized_weight = quantize_weight_tensor(
            original_module.weight.data, layer_name, cfg, int_quant_fn, fp_quant_fn
        )
        if quantized_weight is None:
            return None

        new_module = cls(
            in_features=original_module.in_features,
            out_features=original_module.out_features,
            bias=original_module.bias is not None,
            dtype=original_module.weight.dtype,
            device=original_module.weight.device,
            layer_name=layer_name,
            cfg=cfg,
            int_quant_fn=int_quant_fn,
        )
        with torch.no_grad():
            new_module.weight.copy_(quantized_weight.to(dtype=new_module.weight.dtype))
            new_module.weight.requires_grad_(False)
            if new_module.bias is not None and original_module.bias is not None:
                new_module.bias.copy_(original_module.bias.data)
                new_module.bias.requires_grad_(False)
        return new_module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.activation_enabled:
            x = _quantize_activation_runtime(
                x,
                layer_name=self.layer_name,
                int_quant_fn=self._int_quant_fn,
                act_bit=self.act_bit,
                act_group_size=self.act_group_size,
                act_quant_dim=self.act_quant_dim,
                act_strict_group_size=self.act_strict_group_size,
                act_clip_style=self.act_clip_style,
                act_e8_scale=self.act_e8_scale,
                act_e8_scale_op=self.act_e8_scale_op,
                act_scale_quant=self.act_scale_quant,
                act_scale_quant_2=self.act_scale_quant_2,
            )
        return F.linear(x, self.weight, self.bias)


__all__ = ["QuantConfig", "QuantLinearInference", "quantize_weight_tensor"]
