# SPDX-License-Identifier: MIT
"""The gfx942 HIP FP8-KV split-KV sparse attention (decode) against V4's
reference on the dequantized rows and against the Triton path, plus its
dispatch policy and fallbacks."""

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
    "tokens, num_splits", [(6, None), (8, 20), (80, 2), (64, None)]
)
def test_hip_matches_reference_and_triton(
    monkeypatch, kernel, launches, heads, tokens, num_splits
):
    """Split counts the policy picks at both ends of the token range, and a
    forced 20-way split that leaves splits without keys."""
    torch.manual_seed(7)
    case = _case(tokens, heads, tokens + 128)
    args = (*case, DIM**-0.5)
    actual = fp8_sparse_attention.sparse_attn_v41_paged_2src_fp8kv(
        *args, num_splits=num_splits
    )
    assert launches == [hip._SPLIT]
    expected = _reference(*case)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)

    monkeypatch.setenv("ATOM_DSV41_FP8_KV_HIP", "0")
    triton_out = fp8_sparse_attention.sparse_attn_v41_paged_2src_fp8kv(*args)
    assert launches == [hip._SPLIT]
    torch.testing.assert_close(actual.float(), triton_out.float(), atol=2e-2, rtol=2e-2)


@gfx942
@pytest.mark.parametrize(
    "tokens, heads, num_splits",
    [
        (5, 16, None),
        (80, 16, 1),
        (200, 16, None),
        (1024, 16, None),
        (64, 32, None),
    ],
)
def test_outside_split_decode_keeps_triton(kernel, launches, tokens, heads, num_splits):
    """Below the token threshold, one split (prefill widths, as a captured
    token-graph bucket sends them here), or more heads than the kernel."""
    torch.manual_seed(11)
    case = _case(tokens, heads, tokens + 128)
    actual = fp8_sparse_attention.sparse_attn_v41_paged_2src_fp8kv(
        *case, DIM**-0.5, num_splits=num_splits
    )
    assert launches == []
    expected = _reference(*case)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@gfx942
def test_graph_replay_matches_eager(kernel, launches):
    torch.manual_seed(13)
    case = _case(8, 16, 136)
    args = (*case, DIM**-0.5)
    eager = fp8_sparse_attention.sparse_attn_v41_paged_2src_fp8kv(*args)
    out = torch.empty_like(eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fp8_sparse_attention.sparse_attn_v41_paged_2src_fp8kv(*args, out=out)
    graph.replay()
    torch.cuda.synchronize()
    assert launches == [hip._SPLIT, hip._SPLIT]
    torch.testing.assert_close(out, eager, atol=0, rtol=0)


def _cpu_call(dtype=torch.int32):
    query = torch.zeros(200, 16, DIM, dtype=torch.bfloat16)
    planes = (
        torch.zeros(8, DIM, dtype=torch.uint8),
        torch.zeros(8, 16, dtype=torch.uint8),
    )
    lists = [torch.zeros(4, dtype=dtype), torch.zeros(201, dtype=dtype)]
    return (query, planes, *lists, planes, *lists)


@pytest.mark.parametrize(
    "tokens, avg_keys, splits",
    [(6, 640, 20), (8, 640, 20), (16, 640, 10), (32, 640, 5), (64, 640, 2)]
    + [(80, 640, 2), (81, 640, 1), (1024, 640, 1), (8, 100, 3), (8, 40, 1)],
)
def test_hip_split_policy(tokens, avg_keys, splits):
    """About 160 (token, split) pairs, no split under 32 keys; one split
    means the call stays on Triton."""
    assert fp8_sparse_attention._hip_num_splits(tokens, avg_keys) == splits


@pytest.mark.parametrize(
    "switch, gfx, builds",
    [("1", "gfx942", True), ("0", "gfx942", False), ("1", "gfx950", False)],
)
def test_warm_up_only_on_gfx942_with_the_switch(monkeypatch, switch, gfx, builds):
    calls = []
    monkeypatch.setenv("ATOM_DSV41_FP8_KV_HIP", switch)
    monkeypatch.setattr(fp8_sparse_attention, "get_gfx", lambda: gfx)
    monkeypatch.setattr(hip, "warm_up", calls.append)
    fp8_sparse_attention.warm_up_hip("cuda:0")
    assert calls == (["cuda:0"] if builds else [])


def test_other_gfx_keeps_triton_without_building(monkeypatch):
    def must_not_build(*args, **kwargs):
        raise AssertionError("asked the gfx942 kernel off gfx942")

    monkeypatch.setenv("ATOM_DSV41_FP8_KV_HIP", "1")
    monkeypatch.setattr(fp8_sparse_attention, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(hip, "applies", must_not_build)
    assert not fp8_sparse_attention.hip_applies(*_cpu_call())


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


def test_code_object_built_once_per_source_and_flags(monkeypatch, tmp_path):
    builds = []

    def hipcc(cmd, **kwargs):
        builds.append(cmd)
        with open(cmd[cmd.index("-o") + 1], "wb") as f:
            f.write(b"co")

    src = tmp_path / "k.hip"
    src.write_text("// v1")
    monkeypatch.setattr(hip_gfx942.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(hip_gfx942.subprocess, "run", hipcc)
    first = hip_gfx942._code_object(str(src), ())
    assert hip_gfx942._code_object(str(src), ()) == first
    assert len(builds) == 1
    assert hip_gfx942._code_object(str(src), ("-DX",)) != first
    src.write_text("// v2")
    assert hip_gfx942._code_object(str(src), ()) != first
    assert len(builds) == 3
