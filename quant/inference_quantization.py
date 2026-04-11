import os
import warnings
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantConfig:
    enabled: bool = False
    linear_quant_mode: str = "w_only"  # "w_only", "wa", or "tiled_wa"
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

    # Tiled W+A linear path params
    n_tile: int = -1
    tiled_act_adaptive_enabled: bool = False
    tiled_act_refresh_interval: int = 1
    tiled_act_metric: str = "l1"  # "absmax", "l1", "l2", "exp_spread", "exp_concentration", or "exp_spread_nz_frac"
    tiled_act_int4_threshold: float = 0.25

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
        if self.linear_quant_mode in {"wa", "tiled_wa"}:
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
        if self.linear_quant_mode not in {"w_only", "wa", "tiled_wa"}:
            raise ValueError(
                "linear_quant_mode must be 'w_only', 'wa', or 'tiled_wa', "
                f"got '{self.linear_quant_mode}'."
            )
        if self.quant_type not in {"int", "fp"}:
            raise ValueError(
                f"weight quant_type must be 'int' or 'fp', got '{self.quant_type}'."
            )
        if self.linear_quant_mode in {"wa", "tiled_wa"} and self.quant_type != "int":
            raise NotImplementedError(
                "Linear W+A quantization currently supports quant_type='int' only."
            )
        if self.linear_quant_mode in {"wa", "tiled_wa"} and self.act_bit >= 16:
            raise ValueError(
                f"linear_quant_mode='{self.linear_quant_mode}' requires act_bit < 16."
            )
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
        if (
            self.activation_enabled or self.linear_quant_mode in {"wa", "tiled_wa"}
        ) and self.act_bit < 2:
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
        if self.linear_quant_mode == "tiled_wa":
            if self.w_group_size <= 0:
                raise ValueError("linear_quant_mode='tiled_wa' requires w_group_size > 0.")
            if self.act_group_size <= 0:
                raise ValueError("linear_quant_mode='tiled_wa' requires act_group_size > 0.")
            if int(self.w_group_size) != int(self.act_group_size):
                raise ValueError(
                    "linear_quant_mode='tiled_wa' requires w_group_size == act_group_size."
                )
            if int(self.n_tile) <= 0:
                raise ValueError("linear_quant_mode='tiled_wa' requires n_tile > 0.")
            if int(self.quant_dim) != -1:
                raise ValueError("linear_quant_mode='tiled_wa' requires quant_dim == -1.")
            if int(self.act_quant_dim) != -1:
                raise ValueError(
                    "linear_quant_mode='tiled_wa' requires act_quant_dim == -1."
                )
        if self.tiled_act_adaptive_enabled:
            if self.linear_quant_mode != "tiled_wa":
                raise ValueError(
                    "tiled_act_adaptive_enabled requires linear_quant_mode='tiled_wa'."
                )
            if int(self.tiled_act_refresh_interval) <= 0:
                raise ValueError(
                    "tiled_act_refresh_interval must be > 0 when adaptive tiled activation is enabled."
                )
            if self.tiled_act_metric not in {
                "absmax", "l1", "l2",
                "exp_spread", "exp_concentration", "exp_spread_nz_frac",
            }:
                raise ValueError(
                    "tiled_act_metric must be one of "
                    "['absmax', 'l1', 'l2', 'exp_spread', 'exp_concentration', 'exp_spread_nz_frac']."
                )
            threshold = float(self.tiled_act_int4_threshold)
            if threshold < 0.0 or threshold > 1.0:
                raise ValueError(
                    "tiled activation INT4 threshold must be in the range [0, 1]."
                )
            if int(self.act_bit) != 8:
                raise ValueError(
                    "Adaptive tiled activation currently assumes baseline act_bit == 8."
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


def _detach_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    return value


_EXP_CONCENTRATION_BAND = 4


def _extract_bf16_exponents(psum_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (exponent tensor, nonzero mask) from a float PSUM via BF16 bit extraction."""
    psum_bf16 = psum_f.bfloat16()
    bits = psum_bf16.view(torch.int16)
    exp = (bits >> 7) & 0xFF
    nonzero_mask = exp != 0
    return exp, nonzero_mask


def _compute_tiled_act_metric(psum: torch.Tensor, metric: str) -> torch.Tensor:
    psum_f = psum.float()
    if metric == "absmax":
        return psum_f.abs().amax()
    if metric == "l1":
        return psum_f.abs().sum()
    if metric == "l2":
        return torch.sqrt(torch.square(psum_f).sum())
    if metric == "exp_spread":
        exp, nonzero_mask = _extract_bf16_exponents(psum_f)
        if not nonzero_mask.any():
            return torch.tensor(0.0, device=psum.device)
        exp_nz = exp[nonzero_mask].float()
        return exp_nz.amax() - exp_nz.amin()
    if metric == "exp_concentration":
        exp, nonzero_mask = _extract_bf16_exponents(psum_f)
        if not nonzero_mask.any():
            return torch.tensor(0.0, device=psum.device)
        exp_nz = exp[nonzero_mask].float()
        max_exp = exp_nz.amax()
        concentrated = (exp_nz >= max_exp - _EXP_CONCENTRATION_BAND).sum()
        return 1.0 - (concentrated.float() / exp_nz.numel())
    if metric == "exp_spread_nz_frac":
        exp, nonzero_mask = _extract_bf16_exponents(psum_f)
        nz_count = nonzero_mask.sum()
        if nz_count == 0:
            return torch.tensor(0.0, device=psum.device)
        exp_nz = exp[nonzero_mask].float()
        spread = exp_nz.amax() - exp_nz.amin()
        nz_frac = nz_count.float() / exp.numel()
        return spread * nz_frac
    raise ValueError(f"Unsupported tiled activation metric '{metric}'.")


def _derive_tiled_act_quant_info(
    tile_metrics: torch.Tensor,
    *,
    baseline_bit: int,
    int4_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    tile_metrics = tile_metrics.float()
    ref = tile_metrics.amax()
    if float(ref.item()) <= 0.0:
        scores = torch.zeros_like(tile_metrics, dtype=torch.float32)
    else:
        scores = (tile_metrics / ref).clamp(0.0, 1.0)

    act_quant_info = torch.full(
        tile_metrics.shape,
        int(baseline_bit),
        dtype=torch.int8,
        device=tile_metrics.device,
    )
    act_quant_info = torch.where(
        scores < float(int4_threshold),
        torch.full_like(act_quant_info, 4),
        act_quant_info,
    )
    return act_quant_info, scores


class TiledWAProfiler:
    def __init__(
        self,
        save_dir: str,
        target_steps: Optional[List[int]] = None,
        target_modules: Optional[List[str]] = None,
        target_layer_indices: Optional[List[int]] = None,
    ):
        self.save_dir = save_dir
        self.target_steps = set(target_steps) if target_steps is not None else None
        self.target_modules = list(target_modules) if target_modules is not None else None
        self.target_layer_indices = (
            set(target_layer_indices) if target_layer_indices is not None else None
        )
        self.current_step = 0
        self.buffer: Dict[str, List[Dict[str, Any]]] = {}
        os.makedirs(self.save_dir, exist_ok=True)

    def _should_capture_layer(self, name: str) -> bool:
        if self.target_modules is not None and not any(
            name == token or name.endswith(token) for token in self.target_modules
        ):
            return False
        if self.target_layer_indices is None:
            return True
        parts = name.split(".")
        for idx, part in enumerate(parts):
            if part in {"layers", "blocks"} and idx + 1 < len(parts):
                try:
                    layer_idx = int(parts[idx + 1])
                except ValueError:
                    continue
                return layer_idx in self.target_layer_indices
        return False

    def should_capture(self, layer_name: str) -> bool:
        if self.target_steps is not None and self.current_step not in self.target_steps:
            return False
        return self._should_capture_layer(layer_name)

    def record(self, layer_name: str, payload: Dict[str, Any]) -> None:
        if not self.should_capture(layer_name):
            return
        saved_payload = dict(payload)
        saved_payload.setdefault("layer_name", layer_name)
        saved_payload.setdefault("step", int(self.current_step))
        self.buffer.setdefault(layer_name, []).append(_detach_to_cpu(saved_payload))

    def step(self, current_step_index: int) -> None:
        self.current_step = int(current_step_index)

    def save_buffer(self) -> None:
        if not self.buffer:
            return
        step_dir = os.path.join(self.save_dir, f"step_{self.current_step}")
        os.makedirs(step_dir, exist_ok=True)
        for layer_name, payloads in self.buffer.items():
            safe_name = layer_name.replace(".", "_")
            for payload_idx, payload in enumerate(payloads):
                file_name = f"{safe_name}.pt"
                if len(payloads) > 1:
                    file_name = f"{safe_name}__call_{payload_idx}.pt"
                torch.save(payload, os.path.join(step_dir, file_name))
        self.buffer = {}

    def clear(self) -> None:
        self.buffer = {}
        self.current_step = 0

    def get_collected_data(self) -> str:
        return self.save_dir


def _validate_tiled_wa_module(
    in_features: int,
    out_features: int,
    *,
    layer_name: str,
    cfg: QuantConfig,
) -> None:
    if cfg.linear_quant_mode != "tiled_wa":
        return
    k_tile = int(cfg.w_group_size)
    n_tile = int(cfg.n_tile)
    if in_features % k_tile != 0:
        raise ValueError(
            f"Layer '{layer_name}' requires in_features={in_features} to be divisible by "
            f"k_tile={k_tile} for linear_quant_mode='tiled_wa'."
        )
    if out_features % n_tile != 0:
        raise ValueError(
            f"Layer '{layer_name}' requires out_features={out_features} to be divisible by "
            f"n_tile={n_tile} for linear_quant_mode='tiled_wa'."
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
        self.linear_quant_mode = cfg.linear_quant_mode
        self.activation_enabled = bool(
            cfg.linear_quant_mode in {"wa", "tiled_wa"} or cfg.activation_enabled
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
        self.k_tile = int(cfg.w_group_size) if cfg.linear_quant_mode == "tiled_wa" else -1
        self.n_tile = int(cfg.n_tile) if cfg.linear_quant_mode == "tiled_wa" else -1
        self.tiled_act_adaptive_enabled = bool(
            cfg.linear_quant_mode == "tiled_wa" and cfg.tiled_act_adaptive_enabled
        )
        self.tiled_act_refresh_interval = int(cfg.tiled_act_refresh_interval)
        self.tiled_act_metric = cfg.tiled_act_metric
        self.tiled_act_int4_threshold = float(cfg.tiled_act_int4_threshold)
        self._tiled_wa_profiler: Optional[TiledWAProfiler] = None
        self._int_quant_fn = int_quant_fn
        self._tiled_forward_step = 0
        self._act_quant_info: Optional[torch.Tensor] = None
        self._last_act_scores: Optional[torch.Tensor] = None

    def set_tiled_wa_profiler(
        self, profiler: Optional[TiledWAProfiler]
    ) -> Optional[TiledWAProfiler]:
        old_profiler = self._tiled_wa_profiler
        self._tiled_wa_profiler = profiler
        return old_profiler

    def _should_refresh_tiled_act_quant(self) -> bool:
        if not self.tiled_act_adaptive_enabled:
            return False
        if self._act_quant_info is None:
            return True
        return (self._tiled_forward_step % self.tiled_act_refresh_interval) == 0

    def _quantize_activation_tile(
        self,
        x_tile: torch.Tensor,
        *,
        bit: int,
    ) -> torch.Tensor:
        if not self.activation_enabled or bit >= 16:
            return x_tile
        return _quantize_activation_runtime(
            x_tile,
            layer_name=self.layer_name,
            int_quant_fn=self._int_quant_fn,
            act_bit=int(bit),
            act_group_size=self.act_group_size,
            act_quant_dim=self.act_quant_dim,
            act_strict_group_size=self.act_strict_group_size,
            act_clip_style=self.act_clip_style,
            act_e8_scale=self.act_e8_scale,
            act_e8_scale_op=self.act_e8_scale_op,
            act_scale_quant=self.act_scale_quant,
            act_scale_quant_2=self.act_scale_quant_2,
        )

    def _forward_tiled_wa(self, x: torch.Tensor) -> torch.Tensor:
        n_tile_ranges = [
            (start, start + self.n_tile)
            for start in range(0, self.out_features, self.n_tile)
        ]
        k_tile_ranges = [
            (start, start + self.k_tile)
            for start in range(0, self.in_features, self.k_tile)
        ]
        capture_psums = (
            self._tiled_wa_profiler is not None
            and self._tiled_wa_profiler.should_capture(self.layer_name)
        )
        refresh_act_quant = self._should_refresh_tiled_act_quant()
        forward_step_idx = self._tiled_forward_step
        quantized_input_cache: Dict[tuple[int, int], torch.Tensor] = {}
        tile_metrics = None
        if self.tiled_act_adaptive_enabled and refresh_act_quant:
            tile_metrics = torch.zeros(
                (len(n_tile_ranges), len(k_tile_ranges)),
                dtype=torch.float32,
                device=x.device,
            )
        applied_act_quant_info = None
        if self.tiled_act_adaptive_enabled:
            if refresh_act_quant or self._act_quant_info is None:
                applied_act_quant_info = torch.full(
                    (len(n_tile_ranges), len(k_tile_ranges)),
                    int(self.act_bit),
                    dtype=torch.int8,
                )
            else:
                applied_act_quant_info = self._act_quant_info.clone()

        output_tiles: List[torch.Tensor] = []
        captured_psum_tiles: List[torch.Tensor] = []

        for n_idx, (n_start, n_end) in enumerate(n_tile_ranges):
            reduced_tile: Optional[torch.Tensor] = None
            per_k_tiles: List[torch.Tensor] = []
            for k_idx, (k_start, k_end) in enumerate(k_tile_ranges):
                tile_bit = int(self.act_bit)
                if (
                    self.tiled_act_adaptive_enabled
                    and not refresh_act_quant
                    and applied_act_quant_info is not None
                ):
                    tile_bit = int(applied_act_quant_info[n_idx, k_idx].item())

                cache_key = (k_idx, tile_bit)
                x_tile = quantized_input_cache.get(cache_key)
                if x_tile is None:
                    x_tile = self._quantize_activation_tile(
                        x[..., k_start:k_end],
                        bit=tile_bit,
                    )
                    quantized_input_cache[cache_key] = x_tile
                psum = F.linear(
                    x_tile,
                    self.weight[n_start:n_end, k_start:k_end],
                    None,
                )

                if tile_metrics is not None:
                    tile_metrics[n_idx, k_idx] = _compute_tiled_act_metric(
                        psum,
                        self.tiled_act_metric,
                    )
                if capture_psums:
                    per_k_tiles.append(psum.detach())
                reduced_tile = psum if reduced_tile is None else reduced_tile + psum
            if reduced_tile is None:
                raise RuntimeError(
                    f"Layer '{self.layer_name}' produced no tiled partial sums."
                )
            if self.bias is not None:
                reduced_tile = reduced_tile + self.bias[n_start:n_end]
            output_tiles.append(reduced_tile)
            if capture_psums:
                captured_psum_tiles.append(torch.stack(per_k_tiles, dim=0))

        output = torch.cat(output_tiles, dim=-1)
        next_act_quant_info = None
        next_act_scores = None
        if tile_metrics is not None:
            next_act_quant_info, next_act_scores = _derive_tiled_act_quant_info(
                tile_metrics,
                baseline_bit=int(self.act_bit),
                int4_threshold=self.tiled_act_int4_threshold,
            )
            self._act_quant_info = next_act_quant_info.detach().cpu()
            self._last_act_scores = next_act_scores.detach().cpu()
        elif self.tiled_act_adaptive_enabled and self._act_quant_info is not None:
            next_act_quant_info = self._act_quant_info.clone()

        if capture_psums and self._tiled_wa_profiler is not None:
            psum_tiles = torch.stack(captured_psum_tiles, dim=0)
            payload = {
                "linear_quant_mode": self.linear_quant_mode,
                "input_shape": tuple(x.shape),
                "weight_shape": tuple(self.weight.shape),
                "output_shape": tuple(output.shape),
                "k_tile": int(self.k_tile),
                "n_tile": int(self.n_tile),
                "num_k_tiles": len(k_tile_ranges),
                "num_n_tiles": len(n_tile_ranges),
                "act_bit": int(self.act_bit),
                "weight_bit": int(getattr(self, "weight_bit", -1)),
                "act_group_size": int(self.act_group_size),
                "w_group_size": int(self.k_tile),
                "forward_step_idx": int(forward_step_idx),
                "psum_tile_layout": [
                    "n_tile_index",
                    "k_tile_index",
                    "input_leading_dims",
                    "n_tile",
                ],
                "n_tile_ranges": n_tile_ranges,
                "k_tile_ranges": k_tile_ranges,
                "psum_tiles": psum_tiles,
            }
            if self.tiled_act_adaptive_enabled:
                payload.update(
                    {
                        "adaptive_act_enabled": True,
                        "adaptive_act_metric": self.tiled_act_metric,
                        "adaptive_act_refresh_interval": int(self.tiled_act_refresh_interval),
                        "adaptive_act_refresh_step": bool(refresh_act_quant),
                        "adaptive_act_applied_bits": applied_act_quant_info,
                        "adaptive_act_next_bits": next_act_quant_info,
                        "adaptive_act_scores": next_act_scores if refresh_act_quant else None,
                    }
                )
            self._tiled_wa_profiler.record(self.layer_name, payload)
        self._tiled_forward_step += 1
        return output

    @classmethod
    def from_linear(
        cls,
        original_module: nn.Linear,
        layer_name: str,
        cfg: QuantConfig,
        int_quant_fn: Callable,
        fp_quant_fn: Callable,
    ) -> Optional["QuantLinearInference"]:
        _validate_tiled_wa_module(
            original_module.in_features,
            original_module.out_features,
            layer_name=layer_name,
            cfg=cfg,
        )
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
            new_module.weight_bit = int(cfg.bit)
            if new_module.bias is not None and original_module.bias is not None:
                new_module.bias.copy_(original_module.bias.data)
                new_module.bias.requires_grad_(False)
        return new_module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.linear_quant_mode == "tiled_wa":
            return self._forward_tiled_wa(x)
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


def attach_tiled_wa_profiler(
    model: nn.Module, profiler: Optional[TiledWAProfiler]
) -> int:
    attached_modules = 0
    for module in model.modules():
        if not isinstance(module, QuantLinearInference):
            continue
        if module.linear_quant_mode != "tiled_wa":
            continue
        module.set_tiled_wa_profiler(profiler)
        attached_modules += 1
    return attached_modules


def detach_tiled_wa_profiler(model: nn.Module) -> int:
    return attach_tiled_wa_profiler(model, None)


__all__ = [
    "QuantConfig",
    "QuantLinearInference",
    "TiledWAProfiler",
    "attach_tiled_wa_profiler",
    "detach_tiled_wa_profiler",
    "quantize_weight_tensor",
]
