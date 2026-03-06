import math
import types
import warnings
from typing import Callable, Optional, Set, Tuple

import torch
import torch.nn as nn

from .inference_quantization import QuantConfig


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
    warned_group_mismatches: Set[Tuple[str, Tuple[int, ...], int, int]],
) -> Optional[int]:
    dim_size = tensor.shape[quant_dim]
    resolved = int(group_size) if int(group_size) > 0 else int(dim_size)
    if dim_size % resolved != 0:
        msg = (
            f"Skipping {tensor_kind} quant for attention layer '{layer_name}': dim_size={dim_size} is not divisible by "
            f"group_size={resolved} (quant_dim={quant_dim})."
        )
        if strict:
            raise ValueError(msg)
        warn_key = (tensor_kind, tuple(tensor.shape), quant_dim, resolved)
        if warn_key not in warned_group_mismatches:
            warnings.warn(msg, RuntimeWarning)
            warned_group_mismatches.add(warn_key)
        return None
    return resolved


def is_attention_quantizable_module(module: nn.Module) -> bool:
    return callable(getattr(module, "_scaled_dot_product_attention", None))


def patch_attention_sdpa_module(
    module: nn.Module,
    *,
    layer_name: str,
    cfg: QuantConfig,
    int_quant_fn: Callable,
    fp_quant_fn: Callable,
) -> bool:
    if not is_attention_quantizable_module(module):
        return False
    if hasattr(module, "_old_scaled_dot_product_attention_quant"):
        return False

    warned_group_mismatches: Set[Tuple[str, Tuple[int, ...], int, int]] = set()

    def _quantize_runtime_tensor(tensor: torch.Tensor, tensor_kind: str) -> torch.Tensor:
        if int(cfg.attn_bit) >= 16:
            return tensor

        quant_dim = _resolve_dim(int(cfg.attn_quant_dim), tensor.ndim)
        resolved_group_size = _resolve_group_size(
            tensor,
            quant_dim,
            int(cfg.attn_group_size),
            bool(cfg.attn_strict_group_size),
            layer_name,
            tensor_kind,
            warned_group_mismatches,
        )
        if resolved_group_size is None:
            return tensor

        if cfg.quant_type == "int":
            return int_quant_fn(
                tensor,
                bit=int(cfg.attn_bit),
                dim=quant_dim,
                group_size=resolved_group_size,
                e8_scale=bool(cfg.e8_scale),
                e8_scale_op=cfg.e8_scale_op,
                clip_style=cfg.clip_style,
                scale_quant=bool(cfg.scale_quant),
                scale_quant_2=bool(cfg.scale_quant_2),
            )
        return fp_quant_fn(
            tensor,
            bit=int(cfg.attn_bit),
            e_bit=int(cfg.e_bit),
            m_bit=int(cfg.m_bit),
            dim=quant_dim,
            group_size=resolved_group_size,
            e8_scale=bool(cfg.e8_scale),
            e8_scale_op=cfg.e8_scale_op,
            scale_quant=bool(cfg.scale_quant),
            scale_quant_2=bool(cfg.scale_quant_2),
        )

    def _quantized_scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        mode = cfg.attn_quant_mode

        q_q = _quantize_runtime_tensor(q, "attn_q")
        k_q = _quantize_runtime_tensor(k, "attn_k")
        v_q = _quantize_runtime_tensor(v, "attn_v")

        # Reuse original kernel to preserve backend behavior unless explicitly asking for full attention math quant.
        if mode in {"qkv", "qkvo"}:
            out = self._old_scaled_dot_product_attention_quant(q_q, k_q, v_q, attn_mask, dropout_p, is_causal)
            if mode == "qkvo":
                out = _quantize_runtime_tensor(out, "attn_out")
            return out

        # mode == "full": quantize score/probability path as well.
        assert mode == "full"
        assert k_q.size(1) == v_q.size(1)
        num_kv_heads = k_q.size(1)
        num_q_heads = q_q.size(1)
        if num_q_heads != num_kv_heads:
            if num_q_heads % num_kv_heads != 0:
                raise ValueError(
                    f"Attention head mismatch in layer '{layer_name}': q_heads={num_q_heads}, kv_heads={num_kv_heads}."
                )
            repeat_factor = num_q_heads // num_kv_heads
            k_q = k_q.repeat_interleave(repeat_factor, dim=1, output_size=num_q_heads)
            v_q = v_q.repeat_interleave(repeat_factor, dim=1, output_size=num_q_heads)

        q_fp = q_q.float()
        k_fp = k_q.float()
        v_fp = v_q.float()
        scale = 1.0 / math.sqrt(q_fp.size(-1))
        attn_scores = torch.matmul(q_fp, k_fp.transpose(-2, -1)) * scale

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~attn_mask, torch.finfo(attn_scores.dtype).min)
            else:
                mask = attn_mask
                if mask.device != attn_scores.device:
                    mask = mask.to(attn_scores.device)
                if mask.dtype != attn_scores.dtype:
                    mask = mask.to(attn_scores.dtype)
                attn_scores = attn_scores + mask

        if is_causal:
            query_len = attn_scores.size(-2)
            key_len = attn_scores.size(-1)
            causal_mask = torch.triu(
                torch.ones((query_len, key_len), dtype=torch.bool, device=attn_scores.device),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(causal_mask, torch.finfo(attn_scores.dtype).min)

        attn_scores = _quantize_runtime_tensor(attn_scores, "attn_scores")
        attn_probs = torch.softmax(attn_scores, dim=-1)
        if dropout_p > 0.0 and self.training:
            attn_probs = torch.dropout(attn_probs, dropout_p, train=True)
        attn_probs = _quantize_runtime_tensor(attn_probs, "attn_probs")

        out = torch.matmul(attn_probs, v_fp).to(dtype=q_q.dtype)
        out = _quantize_runtime_tensor(out, "attn_out")
        return out

    module._old_scaled_dot_product_attention_quant = module._scaled_dot_product_attention
    module._scaled_dot_product_attention = types.MethodType(_quantized_scaled_dot_product_attention, module)
    return True


def unpatch_attention_sdpa_module(module: nn.Module) -> bool:
    if not hasattr(module, "_old_scaled_dot_product_attention_quant"):
        return False
    module._scaled_dot_product_attention = module._old_scaled_dot_product_attention_quant
    del module._old_scaled_dot_product_attention_quant
    return True


__all__ = ["is_attention_quantizable_module", "patch_attention_sdpa_module", "unpatch_attention_sdpa_module"]
