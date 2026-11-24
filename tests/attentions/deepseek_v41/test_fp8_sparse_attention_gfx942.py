# SPDX-License-Identifier: MIT
"""The gfx942 HIP FP8-KV sparse attention: split partials (decode) and the
single-split kernel (prefill) against V4's reference on the dequantized rows
and against the Triton path, plus its fallbacks."""

import logging
import subprocess

import pytest
import torch

from atom.model_ops import hip_gfx942
from atom.model_ops.attentions.deepseek_v41 import fp8_attention, fp8_sparse_attention
from atom.model_ops.attentions.deepseek_v41 import (
    fp8_sparse_attention_gfx942 as hip,
)

DIM = 512
POOL_ROWS = 4096


def _gfx942():
    if not torch.cuda.is_available():
        return False
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx() == "gfx942"


gfx942 = pytest.mark.skipif(not _gfx942(), reason="gfx942 GPU required")


@pytest.fixture
def kernel():
    if hip._kernel() is None:
        pytest.skip("hipcc could not build the kernel")


@pytest.fixture
def launches(monkeypatch):
    names = []
    launch = hip._launch

    def spy(name, *args, **kwargs):
        names.append(name)
        return launch(name, *args, **kwargs)

    monkeypatch.setattr(hip, "_launch", spy)
    return names


def _rows(n):
    from atom.model_ops.blockscale import quantize_fp8

    values = quantize_fp8(torch.randn(n, DIM, device="cuda").bfloat16())
    return tuple(v.contiguous() for v in values)


def _case(tokens, heads, extend_rows):
    """Per-token CSA and SWA lists: empty, single, around the 16-key tile,
    the full top-k, and -1 holes."""
    prefix_cycle = [0, 1, 15, 16, 17, 37, 512, 300]
    extend_cycle = [128, 5, 0, 1, 128, 77, 33]
    prefix_counts = [prefix_cycle[t % len(prefix_cycle)] for t in range(tokens)]
    extend_counts = [extend_cycle[t % len(extend_cycle)] for t in range(tokens)]

    def lists(counts, rows):
        idx = torch.cat([torch.randperm(rows, device="cuda")[:n] for n in counts]).int()
        if idx.numel() > 3:
            idx[:: max(1, idx.numel() // 7)] = -1
        ptr = torch.tensor([0, *counts], device="cuda").cumsum(0).int()
        return idx, ptr

    pool, extend = _rows(POOL_ROWS), _rows(extend_rows)
    prefix, prefix_ptr = lists(prefix_counts, POOL_ROWS)
    second, second_ptr = lists(extend_counts, extend_rows)
    query = torch.randn(tokens, heads, DIM, device="cuda").bfloat16()
    sink = torch.randn(heads, device="cuda")
    return query, pool, prefix, prefix_ptr, extend, second, second_ptr, sink


def _reference(query, pool, prefix, prefix_ptr, extend, second, second_ptr, sink):
    from atom.model_ops.v4_kernels import sparse_attn_v4_paged_2src_reference

    return sparse_attn_v4_paged_2src_reference(
        query.float(),
        fp8_attention.dequantize_rows(*fp8_attention._split(pool, DIM)),
        prefix,
        prefix_ptr,
        fp8_attention.dequantize_rows(*fp8_attention._split(extend, DIM)),
        second,
        second_ptr,
        sink,
        DIM**-0.5,
    )


@gfx942
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize(
    "tokens, num_splits, kernel_name",
    [
        (6, None, hip._SPLIT),
        (8, 20, hip._SPLIT),
        (80, 2, hip._SPLIT),
        (80, 1, hip._DIRECT),
        (200, None, hip._DIRECT),
    ],
)
def test_hip_matches_reference_and_triton(
    monkeypatch, kernel, launches, heads, tokens, num_splits, kernel_name
):
    """Every split count the policy picks, a forced 20-way split that leaves
    splits without keys, and the single-split kernel."""
    torch.manual_seed(7)
    case = _case(tokens, heads, tokens + 128)
    args = (*case, DIM**-0.5)
    actual = fp8_sparse_attention.sparse_attn_v41_paged_2src_fp8kv(
        *args, num_splits=num_splits
    )
    assert launches == [kernel_name]
    expected = _reference(*case)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)

    monkeypatch.setenv("ATOM_DSV41_FP8_KV_HIP", "0")
    triton_out = fp8_sparse_attention.sparse_attn_v41_paged_2src_fp8kv(*args)
    assert launches == [kernel_name]
    torch.testing.assert_close(actual.float(), triton_out.float(), atol=2e-2, rtol=2e-2)


@gfx942
@pytest.mark.parametrize("heads", [8, 16])
def test_prefill_entry_runs_single_split_in_place(kernel, launches, heads):
    """The BF16 entry sends a full-width prefill (window source no larger
    than its selection) to the single-split kernel instead of the BF16
    scratch, writing over the query as the model does."""
    torch.manual_seed(11)
    tokens = 1024
    case = _case(tokens, heads, tokens + 128)
    expected = _reference(*case)
    query = case[0]
    out = fp8_attention.sparse_attn_v41_paged_2src_bf16(*case, DIM**-0.5, out=query)
    assert launches == [hip._DIRECT]
    assert out.data_ptr() == query.data_ptr()
    torch.testing.assert_close(out.float(), expected.float(), atol=2e-2, rtol=2e-2)


@gfx942
def test_more_heads_than_the_kernel_keep_triton(kernel, launches):
    torch.manual_seed(13)
    case = _case(200, 32, 328)
    fp8_attention.sparse_attn_v41_paged_2src_bf16(*case, DIM**-0.5)
    assert launches == []


def _cpu_call(dtype=torch.int32):
    query = torch.zeros(200, 16, DIM, dtype=torch.bfloat16)
    planes = (
        torch.zeros(8, DIM, dtype=torch.uint8),
        torch.zeros(8, 16, dtype=torch.uint8),
    )
    lists = [torch.zeros(4, dtype=dtype), torch.zeros(201, dtype=dtype)]
    return (query, planes, *lists, planes, *lists, torch.zeros(16))


def test_switch_off_keeps_triton(monkeypatch):
    monkeypatch.setenv("ATOM_DSV41_FP8_KV_HIP", "0")
    assert not fp8_sparse_attention.hip_applies(*_cpu_call())


def test_int64_lists_keep_triton(monkeypatch):
    monkeypatch.setenv("ATOM_DSV41_FP8_KV_HIP", "1")
    assert not fp8_sparse_attention.hip_applies(*_cpu_call(torch.int64))


@pytest.fixture
def fresh_kernel():
    hip._kernel.cache_clear()
    yield
    hip._kernel.cache_clear()


def test_failed_build_warns_and_keeps_triton(monkeypatch, caplog, fresh_kernel):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["hipcc"])

    monkeypatch.setattr(hip_gfx942, "load", fail)
    with caplog.at_level(logging.WARNING):
        assert hip._kernel() is None
    assert "keeping Triton" in caplog.text


def test_capture_never_builds(monkeypatch, fresh_kernel):
    def built_in_capture(*args, **kwargs):
        raise AssertionError("hipcc ran inside a graph capture")

    monkeypatch.setattr(hip_gfx942, "load", built_in_capture)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert not hip._ready("cuda:0")


def test_warm_up_skips_a_cpu_device(monkeypatch, fresh_kernel):
    def must_not_build(*args, **kwargs):
        raise AssertionError("built for a CPU device")

    monkeypatch.setattr(hip_gfx942, "load", must_not_build)
    hip.warm_up("cpu")
