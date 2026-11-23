// SPDX-License-Identifier: MIT
// gfx942 split-KV sparse attention over MXFP8 CSA / SWA planes (fp8_sparse_attention_gfx942.py). Adapted from
// SGLang's sparse_mla_decode_gfx942.hip (Apache-2.0) to ATOM's planes and CSR row lists:
//
// Keys: a plane pair per source, data [rows, 512] e4m3fnuz and scale [rows, 16] e8m0 (one code per 32 dims, the rope
// dims included). A token's keys are its prefix (CSA) list then its extend (SWA) list, each an int32 CSR span;
// positions of the prefix are padded to 16 so a 16-key tile lies in one source. -1 rows are skipped.
//
// QK^T on the fp8 MFMA without dequantizing K: q (bf16) = hi + lo, two e4m3fnuz values after a per-head power-of-two
// scale, one 16x16x32 MFMA pair per 32-dim group whose sum is multiplied by the key's group scale. PV on the bf16
// MFMA with p rounded to bf16 and V = fp8 * 2^e, exact in bf16. Lane l: MFMA column l % 16 (head for S, dim slot for
// O), group g = l / 16. A CTA is 4 waves: two (token, split) pairs x two halves of the 512 output dims; both halves
// of a pair compute the same QK^T and softmax state. Each wave loads the next tile's QK^T operands while it finishes
// the current tile.
//
// Output: the layout of ATOM's split-combine kernel, acc [T * S * H, 512] fp32 un-normalised and (m, l) [T * S * H, 2]
// in log2 units; a split without valid keys writes m = -FLT_MAX, l = 0, acc = 0.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef short i16x4 __attribute__((ext_vector_type(4)));
typedef unsigned short u16x2 __attribute__((ext_vector_type(2)));

constexpr int kD = 512;
constexpr int kG = kD / 32;

struct Args {
    const uint16_t* q;          // bf16 [T, H, 512]
    const uint8_t* p_data;      // CSA [rows, 512]
    const uint8_t* p_scale;     // CSA [rows, 16]
    const int32_t* p_idx;
    const int32_t* p_ptr;       // [T + 1]
    const uint8_t* e_data;      // SWA
    const uint8_t* e_scale;
    const int32_t* e_idx;
    const int32_t* e_ptr;
    float* part_acc;            // [T * S * H, 512]
    float* part_ml;             // [T * S * H, 2]
    const uint8_t* zero;        // 512 + 16 zero bytes: data and scales of an invalid key
    int64_t q_t, q_h;
    float sm_scale;
    int32_t T, H, S;
};

__device__ __forceinline__ float bf16_lo(uint32_t d) { return __uint_as_float(d << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t d) { return __uint_as_float(d & 0xffff0000u); }
__device__ __forceinline__ float pow2_e8m0(uint32_t e) { return __uint_as_float(e << 23); }
__device__ __forceinline__ uint64_t u64(uint32_t a, uint32_t b) { return (uint64_t)a | ((uint64_t)b << 32); }
__device__ __forceinline__ i16x4 bf16x4(uint32_t a, uint32_t b) { return __builtin_bit_cast(i16x4, u64(a, b)); }

__device__ __forceinline__ uint32_t rne_bf16_bits(float x) {
    const uint32_t u = __float_as_uint(x);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

// Eight bf16 q values (d holds dims 2w, 2w + 1 per dword) scaled by s -> hi and lo e4m3fnuz bytes, hi + lo = q * s.
__device__ __forceinline__ void q_split(u32x4 d, float s, uint64_t& hi, uint64_t& lo) {
    float x[8];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        x[2 * w] = bf16_lo(d[w]) * s;
        x[2 * w + 1] = bf16_hi(d[w]) * s;
    }
    uint32_t h[2], l[2];
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        int hv = __builtin_amdgcn_cvt_pk_fp8_f32(x[4 * w], x[4 * w + 1], 0, false);
        hv = __builtin_amdgcn_cvt_pk_fp8_f32(x[4 * w + 2], x[4 * w + 3], hv, true);
        const f32x2 h01 = __builtin_amdgcn_cvt_pk_f32_fp8(hv, false);
        const f32x2 h23 = __builtin_amdgcn_cvt_pk_f32_fp8(hv, true);
        int lv = __builtin_amdgcn_cvt_pk_fp8_f32(x[4 * w] - h01[0], x[4 * w + 1] - h01[1], 0, false);
        lv = __builtin_amdgcn_cvt_pk_fp8_f32(x[4 * w + 2] - h23[0], x[4 * w + 3] - h23[1], lv, true);
        h[w] = (uint32_t)hv;
        l[w] = (uint32_t)lv;
    }
    hi = u64(h[0], h[1]);
    lo = u64(l[0], l[1]);
}

// LDS query operands: group m, lane l -> (hi 8 bytes, lo 8 bytes) of head l % 16, dims 32m + 8(l / 16) .. +8.
struct QOps {
    u32x4 v[kG][64];
};

// A token's position space: prefix [0, p_pad), extend [p_pad, p_pad + e_len).
struct Lists {
    const int32_t* p_idx;
    const int32_t* e_idx;
    int p_len, p_pad, e_len;
};

struct Tile {
    const int32_t* idx;   // the tile's first index
    const uint8_t* data;
    const uint8_t* scale;
    int len;              // valid positions from the tile's first one (<= 0: none)
};

// A-operand key n (QK^T rows) and the C-row keys 4g .. 4g + 3 (scores, scales, PV rows) of a tile.
struct Keys {
    const uint8_t* adata;
    const uint8_t* cdata[4];
    const uint8_t* cscale[4];
    bool cvalid[4];
};

__device__ __forceinline__ Tile tile_at(const Args& a, const Lists& ls, int pos, int k1) {
    Tile tl;
    if (pos < ls.p_pad) {
        tl.idx = ls.p_idx + pos;
        tl.data = a.p_data;
        tl.scale = a.p_scale;
        tl.len = min(ls.p_len, k1) - pos;
    } else {
        const int e = pos - ls.p_pad;
        tl.idx = ls.e_idx + e;
        tl.data = a.e_data;
        tl.scale = a.e_scale;
        tl.len = min(ls.e_len, k1 - ls.p_pad) - e;
    }
    return tl;
}

__device__ __forceinline__ int idx_of(const Tile& tl, int j) { return j < tl.len ? tl.idx[j] : -1; }

__device__ __forceinline__ Keys keys_of(const Tile& tl, const uint8_t* zero, int ia, const int (&kc)[4]) {
    Keys k;
    k.adata = ia >= 0 ? tl.data + (size_t)ia * kD : zero;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        k.cvalid[i] = kc[i] >= 0;
        k.cdata[i] = k.cvalid[i] ? tl.data + (size_t)kc[i] * kD : zero;
        k.cscale[i] = k.cvalid[i] ? tl.scale + (size_t)kc[i] * kG : zero + kD;
    }
    return k;
}

__device__ __forceinline__ void load_qk(const Keys& k, int g, u32x2 (&kr)[kG], u32x4 (&sc)[4]) {
#pragma unroll
    for (int m = 0; m < kG; ++m) kr[m] = *(const u32x2*)(k.adata + 32 * m + 8 * g);
#pragma unroll
    for (int i = 0; i < 4; ++i) sc[i] = *(const u32x4*)k.cscale[i];
}

__device__ __forceinline__ uint32_t code_of(const u32x4& s, int m) { return (s[m >> 2] >> (8 * (m & 3))) & 0xffu; }

// PV of this half's 4 chunks of 64 dims into blocks 4cc + k: B row k = key 4g + k, column n = dim 64c + 4n + k.
template <int HALF>
__device__ __forceinline__ void pv_half(i16x4 pa4, const uint32_t (&vx)[4][4], const u32x4 (&sc)[4], int n,
                                        f32x4 (&o)[16]) {
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
        const int m = 2 * (4 * HALF + cc) + (n >> 3);
        float v[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float e = pow2_e8m0(code_of(sc[i], m));
            const f32x2 lo = __builtin_amdgcn_cvt_pk_f32_fp8((int)vx[i][cc], false) * e;
            const f32x2 hi = __builtin_amdgcn_cvt_pk_f32_fp8((int)vx[i][cc], true) * e;
            v[i][0] = lo[0];
            v[i][1] = lo[1];
            v[i][2] = hi[0];
            v[i][3] = hi[1];
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const i16x4 vb = bf16x4(__builtin_amdgcn_perm(__float_as_uint(v[1][k]), __float_as_uint(v[0][k]), 0x07060302u),
                                    __builtin_amdgcn_perm(__float_as_uint(v[3][k]), __float_as_uint(v[2][k]), 0x07060302u));
            o[4 * cc + k] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(pa4, vb, o[4 * cc + k], 0, 0, 0);
        }
    }
}

template <int HALF>
__device__ __forceinline__ void run(const Args& a, const QOps& s_q, const Lists& ls, int lane, int n, int g, int t,
                                    int split, int k0, int k1, float c_q) {
    f32x4 o[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i] = f32x4{0.f, 0.f, 0.f, 0.f};
    float m_run = -INFINITY, l_run = 0.f;

    Tile cur_t = tile_at(a, ls, k0, k1);
    Keys cur;
    u32x2 kr[kG];
    u32x4 sc[4];
    {
        const bool live = k0 < k1 && cur_t.len > 0;
        int kc[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) kc[i] = live ? idx_of(cur_t, 4 * g + i) : -1;
        cur = keys_of(cur_t, a.zero, live ? idx_of(cur_t, n) : -1, kc);
        load_qk(cur, g, kr, sc);
    }

    for (int pos = k0; pos < k1; pos += 16) {
        const Tile nxt_t = tile_at(a, ls, pos + 16, k1);
        const bool next_live = pos + 16 < k1 && nxt_t.len > 0;
        int ia_n = -1;
        int kc_n[4] = {-1, -1, -1, -1};
        if (next_live) {
            ia_n = idx_of(nxt_t, n);
#pragma unroll
            for (int i = 0; i < 4; ++i) kc_n[i] = idx_of(nxt_t, 4 * g + i);
        }
        uint32_t vx[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int cc = 0; cc < 4; ++cc) vx[i][cc] = *(const uint32_t*)(cur.cdata[i] + 256 * HALF + 64 * cc + 4 * n);

        // ---- QK^T: one fp8 MFMA pair (q hi, q lo) per 32-dim group, scaled by each C-row key's group code
        float acc_s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int m = 0; m < kG; ++m) {
            const u32x4 qv = s_q.v[m][lane];
            const long ka8 = (long)u64(kr[m][0], kr[m][1]);
            f32x4 ac = {0.f, 0.f, 0.f, 0.f};
            ac = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(ka8, (long)u64(qv[0], qv[1]), ac, 0, 0, 0);
            ac = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(ka8, (long)u64(qv[2], qv[3]), ac, 0, 0, 0);
#pragma unroll
            for (int i = 0; i < 4; ++i) acc_s[i] += pow2_e8m0(code_of(sc[i], m)) * ac[i];
        }
        float s[4];
        u32x4 sc_cur[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sc_cur[i] = sc[i];
            s[i] = cur.cvalid[i] ? c_q * acc_s[i] : -INFINITY;
        }
        __builtin_amdgcn_sched_barrier(0);

        // ---- next tile's addresses and QK^T operands, in flight during this tile's softmax and PV
        cur = keys_of(nxt_t, a.zero, ia_n, kc_n);
        load_qk(cur, g, kr, sc);

        // ---- online softmax over the tile's 16 keys, per head (column n)
        float tmax = fmaxf(fmaxf(s[0], s[1]), fmaxf(s[2], s[3]));
        tmax = fmaxf(tmax, __shfl_xor(tmax, 16));
        tmax = fmaxf(tmax, __shfl_xor(tmax, 32));
        const float m_new = fmaxf(m_run, tmax);
        const bool none = m_new == -INFINITY;
        const float alpha = none ? 1.f : exp2f(m_run - m_new);
        float p[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) p[i] = none ? 0.f : exp2f(s[i] - m_new);
        l_run = l_run * alpha + ((p[0] + p[1]) + (p[2] + p[3]));
        m_run = m_new;
        const i16x4 pa4 = bf16x4(rne_bf16_bits(p[0]) | (rne_bf16_bits(p[1]) << 16),
                                 rne_bf16_bits(p[2]) | (rne_bf16_bits(p[3]) << 16));
        // O rows are heads 4g + i, their alpha sits in lane 4g + i
        if (__any(alpha != 1.f)) {
            float al[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) al[i] = __shfl(alpha, 4 * g + i);
#pragma unroll
            for (int blk = 0; blk < 16; ++blk)
#pragma unroll
                for (int i = 0; i < 4; ++i) o[blk][i] *= al[i];
        }
        pv_half<HALF>(pa4, vx, sc_cur, n, o);
    }

    float l_tot = l_run + __shfl_xor(l_run, 16);
    l_tot += __shfl_xor(l_tot, 32);
    const size_t row0 = ((size_t)t * a.S + split) * a.H;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int h = 4 * g + i;
        if (h < a.H) {
            float* row = a.part_acc + (row0 + h) * kD;
#pragma unroll
            for (int cc = 0; cc < 4; ++cc) {
                const int dim = 256 * HALF + 64 * cc + 4 * n;
                *(f32x4*)(row + dim) = f32x4{o[4 * cc][i], o[4 * cc + 1][i], o[4 * cc + 2][i], o[4 * cc + 3][i]};
            }
        }
    }
    if (HALF == 0 && g == 0 && n < a.H) {
        const bool any = m_run != -INFINITY;
        a.part_ml[(row0 + n) * 2] = any ? m_run : -3.4028234663852886e38f;
        a.part_ml[(row0 + n) * 2 + 1] = any ? l_tot : 0.f;
    }
}

// A wave pair takes one (token, split) of the flattened T x S grid and builds its own query operands.
extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(2)))
mxfp8_sparse_attn_split_gfx942(Args a) {
    __shared__ QOps s_qp[2];
    __shared__ float s_cp[2][16];
    const int lane = threadIdx.x & 63;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int half = wave & 1;
    const int n = lane & 15, g = lane >> 4;
    const int pair = blockIdx.x * 2 + (wave >> 1);
    const int t = pair / a.S;
    const int split = pair - t * a.S;
    const bool live_pair = t < a.T;
    QOps& s_q = s_qp[wave >> 1];
    float* s_c = s_cp[wave >> 1];
    constexpr float LOG2E = 1.4426950408889634f;

    if (half == 0 && live_pair) {
        // column n is head n; lane group g holds dims 32m + 8g .. +8 of each group m
        const bool head_ok = n < a.H;
        const uint16_t* qh = a.q + (size_t)t * a.q_t + (size_t)min(n, a.H - 1) * a.q_h;
        u32x4 qraw[kG / 2];
        // qraw[j] = dims 64j + 8g .. +8 (group 2j) and 64j + 32 + 8g .. +8 (group 2j + 1), as two u32x2 halves
#pragma unroll
        for (int j = 0; j < kG / 2; ++j) {
            const u32x2 a0 = head_ok ? *(const u32x2*)(qh + 64 * j + 8 * g) : u32x2{0, 0};
            const u32x2 a1 = head_ok ? *(const u32x2*)(qh + 64 * j + 8 * g + 4) : u32x2{0, 0};
            const u32x2 b0 = head_ok ? *(const u32x2*)(qh + 64 * j + 32 + 8 * g) : u32x2{0, 0};
            const u32x2 b1 = head_ok ? *(const u32x2*)(qh + 64 * j + 32 + 8 * g + 4) : u32x2{0, 0};
            qraw[j] = u32x4{a0[0], a0[1], a1[0], a1[1]};
            s_q.v[2 * j + 1][lane] = u32x4{b0[0], b0[1], b1[0], b1[1]};
        }
        u16x2 mx = {0, 0};
#pragma unroll
        for (int j = 0; j < kG / 2; ++j) {
            const u32x4 b = s_q.v[2 * j + 1][lane];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                mx = __builtin_elementwise_max(mx, __builtin_bit_cast(u16x2, (uint32_t)(qraw[j][w] & 0x7fff7fffu)));
                mx = __builtin_elementwise_max(mx, __builtin_bit_cast(u16x2, (uint32_t)(b[w] & 0x7fff7fffu)));
            }
        }
        uint32_t amax = max((uint32_t)mx[0], (uint32_t)mx[1]);
        amax = max(amax, (uint32_t)__shfl_xor((int)amax, 16));
        amax = max(amax, (uint32_t)__shfl_xor((int)amax, 32));
        // |q| * 2^(6 - E) < 128 for the head's largest |q| = m * 2^E: below the e4m3fnuz maximum 240
        const int ef = (int)(amax >> 7);
        const float qs = __uint_as_float((uint32_t)min(max(260 - ef, 1), 254) << 23);
        const float qinv = __uint_as_float((uint32_t)min(max(ef - 6, 1), 254) << 23);
#pragma unroll
        for (int j = 0; j < kG / 2; ++j) {
            uint64_t hi0, lo0, hi1, lo1;
            const u32x4 b = s_q.v[2 * j + 1][lane];
            q_split(qraw[j], qs, hi0, lo0);
            q_split(b, qs, hi1, lo1);
            s_q.v[2 * j][lane] = u32x4{(uint32_t)hi0, (uint32_t)(hi0 >> 32), (uint32_t)lo0, (uint32_t)(lo0 >> 32)};
            s_q.v[2 * j + 1][lane] = u32x4{(uint32_t)hi1, (uint32_t)(hi1 >> 32), (uint32_t)lo1, (uint32_t)(lo1 >> 32)};
        }
        if (g == 0) s_c[n] = a.sm_scale * LOG2E * (ef > 0 ? qinv : 0.f);
    }
    __syncthreads();
    if (!live_pair) return;
    const float c_q = s_c[n];

    Lists ls;
    const int p_start = a.p_ptr[t], e_start = a.e_ptr[t];
    ls.p_len = a.p_ptr[t + 1] - p_start;
    ls.e_len = a.e_ptr[t + 1] - e_start;
    ls.p_pad = (ls.p_len + 15) & ~15;
    ls.p_idx = a.p_idx + p_start;
    ls.e_idx = a.e_idx + e_start;
    const int total = ls.p_pad + ls.e_len;
    const int chunk = ((total + a.S - 1) / a.S + 15) & ~15;
    const int k0 = split * chunk;
    const int k1 = min(k0 + chunk, total);

    if (half == 0)
        run<0>(a, s_q, ls, lane, n, g, t, split, k0, k1, c_q);
    else
        run<1>(a, s_q, ls, lane, n, g, t, split, k0, k1, c_q);
}
