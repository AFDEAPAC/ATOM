# SPDX-License-Identifier: MIT
"""gfx942 HIP split-KV sparse attention over MXFP8 planes (off: ATOM_DSV41_FP8_KV_HIP=0).

The split launches of `sparse_attn_v41_paged_2src_fp8kv` (decode) decode each
selected row to BF16 in registers before both dots. This kernel, adapted from
SGLang's sparse_mla_decode_gfx942.hip, runs QK^T on the fp8 MFMA over the raw
key bytes (query split into fp8 hi + lo) and writes the partials of ATOM's
split-combine kernel, which finishes as before.
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
            )
        ]
        + [(n, ctypes.c_int64) for n in ("q_t", "q_h")]
        + [("sm_scale", ctypes.c_float)]
        + [(n, ctypes.c_int32) for n in ("T", "H", "S")]
    )


@cache
def _kernel():
    name = "mxfp8_sparse_attn_split_gfx942"
    try:
        hip, fns = hip_gfx942.load(_SRC, (name,))
        return hip, fns[name]
    except (OSError, subprocess.CalledProcessError, RuntimeError) as e:
        logger.warning("gfx942 HIP FP8-KV attention unavailable, keeping Triton: %s", e)
        return None


_zero = {}


def _zero_row(device):
    z = _zero.get(device)
    if z is None:
        z = _zero[device] = torch.zeros(_D + _D // 32, dtype=torch.uint8, device=device)
    return z


def applies(query, pool, kv):
    _, heads, dim = query.shape
    if (
        dim != _D
        or heads > _MAX_HEADS
        or query.dtype != torch.bfloat16
        or query.stride(-1) != 1
    ):
        return False
    if query.stride(1) % 8 or query.stride(0) % 8 or query.data_ptr() % 16:
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
    return _kernel() is not None


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
    hip, fn = _kernel()
    tokens, heads, _ = query.shape
    if (
        kv_indices_prefix.dtype != torch.int32
        or kv_indices_extend.dtype != torch.int32
        or kv_indptr_prefix.dtype != torch.int32
        or kv_indptr_extend.dtype != torch.int32
    ):
        raise ValueError("HIP FP8-KV attention takes int32 CSR lists")
    p_data, p_scale = (p.view(torch.uint8) for p in pool)
    e_data, e_scale = (p.view(torch.uint8) for p in kv)
    args = _Args(
        query.data_ptr(),
        p_data.data_ptr(),
        p_scale.data_ptr(),
        kv_indices_prefix.data_ptr(),
        kv_indptr_prefix.data_ptr(),
        e_data.data_ptr(),
        e_scale.data_ptr(),
        kv_indices_extend.data_ptr(),
        kv_indptr_extend.data_ptr(),
        part_acc.data_ptr(),
        part_ml.data_ptr(),
        _zero_row(query.device).data_ptr(),
        query.stride(0),
        query.stride(1),
        float(softmax_scale),
        tokens,
        heads,
        num_splits,
    )
    hip.launch(fn, (tokens * num_splits + 1) // 2, args)
