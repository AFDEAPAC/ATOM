# SPDX-License-Identifier: MIT
"""gfx942 HIP sparse attention over MXFP8 planes (off: ATOM_DSV41_FP8_KV_HIP=0).

The split launches of `sparse_attn_v41_paged_2src_fp8kv` (decode) decode each
selected row to BF16 in registers before both dots, and full-width prefill
decodes the selected rows into BF16 scratch first. This kernel runs QK^T on
the fp8 MFMA over the raw key bytes (query split into fp8 hi + lo). Split
calls write the partials of
ATOM's split-combine kernel, which finishes as before; single-split calls
(prefill) finish with the sink in the kernel and write the BF16 output.
"""

import ctypes
import logging
import os
import subprocess
from functools import cache

import torch

from atom.model_ops import hip_gfx942

logger = logging.getLogger(__name__)

_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fp8_sparse_attention_gfx942.hip"
)
_D = 512
_MAX_HEADS = 16


class _Args(ctypes.Structure):
    _fields_ = (
        [
            (n, ctypes.c_void_p)
            for n in (
                "q",
                "p_data",
                "p_scale",
                "p_idx",
                "p_ptr",
                "e_data",
                "e_scale",
                "e_idx",
                "e_ptr",
                "part_acc",
                "part_ml",
                "zero",
                "out",
                "sink",
            )
        ]
        + [(n, ctypes.c_int64) for n in ("q_t", "q_h", "o_t", "o_h")]
        + [("sm_scale", ctypes.c_float)]
        + [(n, ctypes.c_int32) for n in ("T", "H", "S")]
    )


_SPLIT = "mxfp8_sparse_attn_split_gfx942"
_DIRECT = "mxfp8_sparse_attn_gfx942"


@cache
def _kernel():
    try:
        return hip_gfx942.load(_SRC, (_SPLIT, _DIRECT))
    except (OSError, subprocess.CalledProcessError, RuntimeError) as e:
        logger.warning("gfx942 HIP FP8-KV attention unavailable, keeping Triton: %s", e)
        return None


_zero = {}


def _device(device):
    device = torch.device(device)
    if device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())
    return device


def _zero_row(device):
    device = _device(device)
    z = _zero.get(device)
    if z is None:
        z = _zero[device] = torch.zeros(_D + _D // 32, dtype=torch.uint8, device=device)
    return z


def warm_up(device):
    """Build the kernels and this device's zero row before any graph capture."""
    if torch.device(device).type == "cuda" and _kernel() is not None:
        _zero_row(device)


def _ready(device):
    # A capture must not run hipcc, load a module, or allocate the zero row.
    if not torch.cuda.is_current_stream_capturing():
        return _kernel() is not None
    return (
        _kernel.cache_info().currsize > 0
        and _kernel() is not None
        and (_device(device) in _zero)
    )


def _bf16_rows(x):
    return (
        x.dtype == torch.bfloat16
        and x.stride(-1) == 1
        and not (x.stride(0) % 8 or x.stride(1) % 8 or x.data_ptr() % 16)
    )


def applies(query, pool, kv, out=None):
    _, heads, dim = query.shape
    if dim != _D or heads > _MAX_HEADS or not _bf16_rows(query):
        return False
    if out is not None and (out.shape != query.shape or not _bf16_rows(out)):
        return False
    for data, scale in (pool, kv):
        if data.shape[-1] != _D or scale.shape[-1] != _D // 32:
            return False
        if (
            data.stride(-2) != _D
            or scale.stride(-2) != _D // 32
            or data.data_ptr() % 16
            or scale.data_ptr() % 16
        ):
            return False
    return _ready(query.device)


def _launch(name, query, pool, lists, softmax_scale, num_splits, **outputs):
    if any(t.dtype != torch.int32 for t in lists):
        raise ValueError("HIP FP8-KV attention takes int32 CSR lists")
    hip, fns = _kernel()
    tokens, heads, _ = query.shape
    p_idx, p_ptr, e_idx, e_ptr = lists
    p_data, p_scale = (p.view(torch.uint8) for p in pool[0])
    e_data, e_scale = (p.view(torch.uint8) for p in pool[1])
    out = outputs.get("out")

    def ptr(key):
        return outputs[key].data_ptr() if key in outputs else None

    args = _Args(
        query.data_ptr(),
        p_data.data_ptr(),
        p_scale.data_ptr(),
        p_idx.data_ptr(),
        p_ptr.data_ptr(),
        e_data.data_ptr(),
        e_scale.data_ptr(),
        e_idx.data_ptr(),
        e_ptr.data_ptr(),
        ptr("part_acc"),
        ptr("part_ml"),
        _zero_row(query.device).data_ptr(),
        ptr("out"),
        ptr("sink"),
        query.stride(0),
        query.stride(1),
        out.stride(0) if out is not None else 0,
        out.stride(1) if out is not None else 0,
        float(softmax_scale),
        tokens,
        heads,
        num_splits,
    )
    hip.launch(fns[name], (tokens * num_splits + 1) // 2, args)


def split_partials(
    query,
    pool,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv,
    kv_indices_extend,
    kv_indptr_extend,
    softmax_scale,
    num_splits,
    part_acc,
    part_ml,
):
    """Fill `part_acc` [T * S * H, 512] and `part_ml` [T * S * H, 2] for the split combine."""
    lists = (kv_indices_prefix, kv_indptr_prefix, kv_indices_extend, kv_indptr_extend)
    _launch(
        _SPLIT,
        query,
        (pool, kv),
        lists,
        softmax_scale,
        num_splits,
        part_acc=part_acc,
        part_ml=part_ml,
    )


def attend(
    query,
    pool,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
    out,
):
    """One split: write the finished BF16 `out` [T, H, 512], sink included; `out` may be `query`."""
    if attn_sink.dtype != torch.float32 or not attn_sink.is_contiguous():
        raise ValueError("HIP FP8-KV attention takes a contiguous fp32 sink")
    lists = (kv_indices_prefix, kv_indptr_prefix, kv_indices_extend, kv_indptr_extend)
    _launch(
        _DIRECT, query, (pool, kv), lists, softmax_scale, 1, out=out, sink=attn_sink
    )
