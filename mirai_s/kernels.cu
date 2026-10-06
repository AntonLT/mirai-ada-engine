// Mirai S kernels, compiled at runtime with NVRTC (torch.cuda._utils._nvrtc_compile): no CUDA toolkit needed.
//
// A linear layer y = W x runs in the rotated domain. With R the saved rotation (Walsh-Hadamard (x) small_q) and S the
// saved signs, W = S R W_rot row by row, so y = W_rot x_rot with x_rot = (H (x) Q) (S x):
//   transform                        x_rot ~ s * (q0 + q1 / 254), two int8 planes per token, plus per-token sums
//   gemv (1 token)                   decode the trellis, two dp4a per state
//   mma (2-384 tokens)               decode once per 64 tokens, int8 tensor-core MMAs
//   levels + cuBLAS int8 GEMM + prefill_output (longer)
// All three compute the same two exact int32 dot products per (token, row) and finish them in output_value, so any
// token's output is bit-identical whichever path its batch takes.
//
// The codebook is computed, not stored: a 16-bit trellis state gives V weights through
//   h = fmix32(state * A + B),  weight_j = c * level(byte j of h) + d_j,
//   level(b) = 8 * pairs(b) + ((3 * (b & 15)) & 15) - 54,  pairs(b) = sum of b's four 2-bit fields.
// c is one constant per codebook and d_j one per column position j (V4: 4, V2: 2), so level stays an integer.
//
// Tape layout (repacked once at conversion): per 64-column packet (128 for V2 T6) an entry state and STEPS symbols of T
// bits packed LSB-first into WORDS 16-byte words; every step replays as state = (state << T | symbol) & 0xFFFF.
// Rows are grouped by 32 and interleaved, so lane = row and a warp's packet load is one contiguous 512-byte read.

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef signed char s8;

#define WARPS 16  // column slices per 32-row CTA

__device__ __forceinline__ u32 fmix_hash(u32 state) {
    u32 x = state * 0xCFCCB83Fu + 0x584B4AA3u;
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    return x ^ (x >> 16);
}

// level(b) + 54 for each byte of h, in [0, 111].
__device__ __forceinline__ u32 levels_plus_54(u32 h) {
    const u32 nibble_pairs = (h & 0x33333333u) + ((h >> 2) & 0x33333333u);
    const u32 pairs = (nibble_pairs + (nibble_pairs >> 4)) & 0x0F0F0F0Fu;
    return (pairs << 3) + (((h & 0x0F0F0F0Fu) * 3u) & 0x0F0F0F0Fu);
}

// acc + sum_i a.u8[i] * b.s8[i]
__device__ __forceinline__ int dp4a_us(u32 a, u32 b, int acc) {
    int result;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(acc));
    return result;
}

template <int T, int WORDS>
__device__ __forceinline__ u32 symbol_at(const u32 (&bits)[4 * WORDS], int index) {
    const int bit = index * T;
    const int word = bit / 32, shift = bit % 32;
    if (shift + T <= 32) return (bits[word] >> shift) & ((1u << T) - 1);
    return __funnelshift_r(bits[word], bits[word + 1], shift) & ((1u << T) - 1);
}

template <int WORDS>
__device__ __forceinline__ void load_packet(const uint4* packets, size_t slot, int lane, u32 (&bits)[4 * WORDS]) {
#pragma unroll
    for (int word = 0; word < WORDS; ++word) {
        const uint4 chunk = __ldg(packets + (slot * WORDS + word) * 32 + lane);
        bits[4 * word] = chunk.x;
        bits[4 * word + 1] = chunk.y;
        bits[4 * word + 2] = chunk.z;
        bits[4 * word + 3] = chunk.w;
    }
}

__device__ __forceinline__ float bf16_to_float(u16 value) { return __int_as_float(static_cast<u32>(value) << 16); }

__device__ __forceinline__ u16 float_to_bf16(float value) {
    u16 bits;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(value));
    return bits;
}

// Replays one packet of a lane's row, handing each 4-column word of levels (level + 54 per byte, columns 4g .. 4g+3 of
// the packet) to use(word, g). Loops are fully unrolled, so g is a compile-time constant inside `use`.
template <int V, int T, int STEPS, int WORDS, typename Use>
__device__ __forceinline__ void decode_packet(const u32 (&bits)[4 * WORDS], u32 state, Use&& use) {
    if constexpr (V == 4) {
#pragma unroll
        for (int step = 0; step < STEPS; ++step) {
            state = ((state << T) | symbol_at<T, WORDS>(bits, step)) & 0xFFFFu;
            use(levels_plus_54(fmix_hash(state)), step);
        }
    } else {  // two V=2 steps make one word: bytes 0-1 of each step's hash
#pragma unroll
        for (int step = 0; step < STEPS; step += 2) {
            state = ((state << T) | symbol_at<T, WORDS>(bits, step)) & 0xFFFFu;
            const u32 first = fmix_hash(state);
            state = ((state << T) | symbol_at<T, WORDS>(bits, step + 1)) & 0xFFFFu;
            use(levels_plus_54(__byte_perm(first, fmix_hash(state), 0x5410)), step / 2);
        }
    }
}

// y = rowscale * (c * s * ((coarse - 54 sum_q0) + (fine - 54 sum_q1) / 254) + sum_r d_r S_r), from the exact dot
// products coarse = sum (level + 54) q0 and fine = sum (level + 54) q1. token = {s, sum q0, sum q1, S_0 .. S_3, 0}.
__device__ __forceinline__ u16 output_value(int coarse, int fine, const float* __restrict__ token,
                                            const float* __restrict__ codebook, float rowscale) {
    coarse -= 54 * __float2int_rn(token[1]);
    fine -= 54 * __float2int_rn(token[2]);
    const float offsets =
        codebook[1] * token[3] + codebook[2] * token[4] + codebook[3] * token[5] + codebook[4] * token[6];
    const float dot =
        codebook[0] * token[0] * (static_cast<float>(coarse) + static_cast<float>(fine) * (1.0f / 254.0f));
    return float_to_bf16(rowscale * (dot + offsets));
}

// ---------------------------------------------------------------------------------------------------------------
// Rotation and quantization, specialized per input shape (WIDTH x ORDER = 1024 x 5, 2048 x 3, 1024 x 17):
//   s = max |x_rot| / 127, q0 = round(x / s), q1 = round((x / s - q0) * 254).
// Up to 64 tokens the Walsh-Hadamard, independent per small_q column, runs as ORDER blocks per token:
//   rotate_columns_N  grid (ORDER, tokens): column o's small_q mix, its WIDTH-point Walsh-Hadamard, the column max
//   quantize_rows_N   grid (tokens): q[m, g] a uint2 (plane 0, plane 1) of the four int8 values of columns 4g .. 4g+3
// Longer inputs run transform_N below, one CTA per token, writing the int8 GEMM's planes instead.
// stats[m] = {s, sum q0, sum q1, residue 0..3, 0} with residue r = s * (sum q0 + sum q1 / 254) over columns = r mod 4:
// sums of exact integers, so both paths give every token the same stats.

__device__ __forceinline__ void write_stats(float* out, float step, const int (&coarse)[4], const int (&fine)[4]) {
    out[0] = step;
    out[1] = static_cast<float>(coarse[0] + coarse[1] + coarse[2] + coarse[3]);
    out[2] = static_cast<float>(fine[0] + fine[1] + fine[2] + fine[3]);
#pragma unroll
    for (int r = 0; r < 4; ++r)
        out[3 + r] = step * (static_cast<float>(coarse[r]) + static_cast<float>(fine[r]) / 254.0f);
    out[7] = 0.0f;
}

__device__ __forceinline__ float block_max(float value, float* shared) {
    for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    value = lane < warps ? shared[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    return value;
}

template <int WIDTH, int ORDER>
__device__ __forceinline__ void columns_body(const u16* __restrict__ x, const float* __restrict__ signs,
                                             const float* __restrict__ small_q, float* __restrict__ rotated,
                                             float* __restrict__ column_max) {
    constexpr int N = WIDTH * ORDER;
    __shared__ float column[WIDTH];
    __shared__ float mix[ORDER];
    __shared__ float shared[32];
    // The token's S x, staged with coalesced reads: the mix below walks it at a stride of ORDER, which straight from
    // global memory cost every CTA ORDER uncoalesced passes over the row. signs are exactly +-1, so flipping the bf16
    // sign bit is exact and the sums below are the ones bf16_to_float(x) * sign * mix gives.
    __shared__ alignas(16) u32 flipped[N / 2];
    const int out = blockIdx.x;
    const size_t base = static_cast<size_t>(blockIdx.y) * N;
    if (threadIdx.x < ORDER) mix[threadIdx.x] = small_q[out * ORDER + threadIdx.x];
    // 16-byte loads (8 values) keep a few independent reads in flight per thread instead of a serial chain.
    const uint4* chunks = reinterpret_cast<const uint4*>(x + base);
    const float4* sign_quads = reinterpret_cast<const float4*>(signs);
#pragma unroll 4
    for (int i = threadIdx.x; i < N / 8; i += blockDim.x) {
        const uint4 chunk = __ldg(chunks + i);
        const float4 low = __ldg(sign_quads + 2 * i), high = __ldg(sign_quads + 2 * i + 1);
        const auto flip = [](float a, float b) { return (a < 0.0f ? 0x8000u : 0u) | (b < 0.0f ? 0x80000000u : 0u); };
        reinterpret_cast<uint4*>(flipped)[i] = make_uint4(chunk.x ^ flip(low.x, low.y), chunk.y ^ flip(low.z, low.w),
                                                          chunk.z ^ flip(high.x, high.y), chunk.w ^ flip(high.z, high.w));
    }
    __syncthreads();
    const u16* row = reinterpret_cast<const u16*>(flipped);
    for (int w = threadIdx.x; w < WIDTH; w += blockDim.x) {
        float value = 0.0f;
#pragma unroll
        for (int c = 0; c < ORDER; ++c) value += bf16_to_float(row[w * ORDER + c]) * mix[c];
        column[w] = value;
    }
    __syncthreads();
#pragma unroll
    for (int stride = 1; stride < WIDTH; stride <<= 1) {
        for (int pair = threadIdx.x; pair < WIDTH / 2; pair += blockDim.x) {
            const int low = (pair / stride) * 2 * stride + pair % stride, high = low + stride;
            const float a = column[low], b = column[high];
            column[low] = a + b;
            column[high] = a - b;
        }
        __syncthreads();
    }
    const float normalization = rsqrtf(static_cast<float>(WIDTH));
    float maximum = 0.0f;
    for (int w = threadIdx.x; w < WIDTH; w += blockDim.x) {
        const float value = column[w] * normalization;
        rotated[base + w * ORDER + out] = value;
        maximum = fmaxf(maximum, fabsf(value));
    }
    maximum = block_max(maximum, shared);
    if (threadIdx.x == 0) column_max[blockIdx.y * ORDER + out] = maximum;
}

template <int WIDTH, int ORDER>
__device__ __forceinline__ void quantize_rows_body(const float* __restrict__ rotated,
                                                   const float* __restrict__ column_max, uint2* __restrict__ q,
                                                   float* __restrict__ stats) {
    constexpr int N = WIDTH * ORDER;
    __shared__ int sums[32][8];
    const size_t base = static_cast<size_t>(blockIdx.x) * N;
    float maximum = 0.0f;
#pragma unroll
    for (int c = 0; c < ORDER; ++c) maximum = fmaxf(maximum, column_max[blockIdx.x * ORDER + c]);
    const float step = maximum > 0.0f ? maximum / 127.0f : 1.0f, inverse = 1.0f / step;
    int local[8] = {};  // per residue class: sum q0 (0..3), sum q1 (4..7)
    for (int group = threadIdx.x; group < N / 4; group += blockDim.x) {
        const float4 values = reinterpret_cast<const float4*>(rotated + base)[group];
        const float v[4] = {values.x, values.y, values.z, values.w};
        u32 plane0 = 0, plane1 = 0;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float scaled = v[r] * inverse;
            const int coarse = min(127, max(-127, __float2int_rn(scaled)));
            const int fine = min(127, max(-127, __float2int_rn((scaled - coarse) * 254.0f)));
            plane0 |= (static_cast<u32>(coarse) & 0xFFu) << (8 * r);
            plane1 |= (static_cast<u32>(fine) & 0xFFu) << (8 * r);
            local[r] += coarse;
            local[4 + r] += fine;
        }
        q[static_cast<size_t>(blockIdx.x) * (N / 4) + group] = make_uint2(plane0, plane1);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
        for (int offset = 16; offset > 0; offset >>= 1) local[i] += __shfl_xor_sync(0xffffffffu, local[i], offset);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    if (lane == 0)
#pragma unroll
        for (int i = 0; i < 8; ++i) sums[warp][i] = local[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        int coarse[4] = {}, fine[4] = {};
        for (int w = 0; w < warps; ++w)
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                coarse[r] += sums[w][r];
                fine[r] += sums[w][4 + r];
            }
        write_stats(stats + static_cast<size_t>(blockIdx.x) * 8, step, coarse, fine);
    }
}

#define SPLIT_ROTATE(N, WIDTH, ORDER)                                                                               \
    extern "C" __global__ void rotate_columns_##N(const u16* x, const float* signs, const float* small_q,            \
                                                  float* rotated, float* column_max) {                               \
        columns_body<WIDTH, ORDER>(x, signs, small_q, rotated, column_max);                                           \
    }                                                                                                                \
    extern "C" __global__ void quantize_rows_##N(const float* rotated, const float* column_max, uint2* q,            \
                                                 float* stats) {                                                     \
        quantize_rows_body<WIDTH, ORDER>(rotated, column_max, q, stats);                                              \
    }

SPLIT_ROTATE(5120, 1024, 5)
SPLIT_ROTATE(6144, 2048, 3)
SPLIT_ROTATE(17408, 1024, 17)

// Butterfly at lane distance `mask`: the lower index of each pair keeps a + b, the upper a - b.
__device__ __forceinline__ float butterfly(float value, int lane, int mask) {
    const float other = __shfl_xor_sync(0xffffffffu, value, mask);
    return (lane & mask) ? other - value : value + other;
}

// transform_N: grid (tokens), 512 threads. Every input is read once and kept in registers; for each small_q column the
// thread mixes its points, the Walsh-Hadamard runs as warp shuffles around one shared-memory transpose, and the values
// stay in registers for the max and the quantization. Mixing sums c in ascending order and the butterflies run at
// strides 1, 2, 4, ..., as rotate_columns does, so q and s are the same. Output: the int8 GEMM's planes, rows
// [0, tokens) plane 0 and rows [tokens, 2 * tokens) plane 1; transform_words_N writes q's words for the mma kernels
// instead, faster than the split kernels from 17 tokens on.
template <int WIDTH, int ORDER, bool WORDS>
__device__ __forceinline__ void transform_body(const u16* __restrict__ x, const float* __restrict__ signs,
                                               const float* __restrict__ small_q, s8* __restrict__ planes,
                                               float* __restrict__ stats, int tokens) {
    constexpr int PER = WIDTH / 512, N = WIDTH * ORDER;  // points per thread per column
    __shared__ float values[WIDTH];
    __shared__ alignas(16) s8 staged[2][N];  // the token's two plane rows, written out as 16-byte stores
    __shared__ float mix[ORDER * ORDER];
    __shared__ int partial[16][8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t token = blockIdx.x;
    for (int i = threadIdx.x; i < ORDER * ORDER; i += 512) mix[i] = small_q[i];

    // phase 1: the thread's points are h = 32 * (warp + 16 * i) + lane
    float inputs[PER][ORDER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int h = 32 * (warp + 16 * i) + lane;
#pragma unroll
        for (int c = 0; c < ORDER; ++c)
            inputs[i][c] = bf16_to_float(x[token * N + h * ORDER + c]) * signs[h * ORDER + c];
    }
    __syncthreads();

    // phase 2: WIDTH 1024 holds h = 32 * lane + warp + 16 * i; WIDTH 2048 holds h = 32 * (lane + 32 * (i % 2)) +
    // warp + 16 * (i / 2)
    const auto phase2_point = [&](int i) {
        return WIDTH == 1024 ? 32 * lane + warp + 16 * i : 32 * (lane + 32 * (i % 2)) + warp + 16 * (i / 2);
    };
    const float normalization = rsqrtf(static_cast<float>(WIDTH));
    float out[ORDER][PER];
    float maximum = 0.0f;
#pragma unroll
    for (int o = 0; o < ORDER; ++o) {
        float element[PER];
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            float value = 0.0f;
#pragma unroll
            for (int c = 0; c < ORDER; ++c) value += inputs[i][c] * mix[o * ORDER + c];
            element[i] = value;
        }
#pragma unroll
        for (int mask = 1; mask <= 16; mask <<= 1)  // strides 1..16: lanes hold consecutive h
#pragma unroll
            for (int i = 0; i < PER; ++i) element[i] = butterfly(element[i], lane, mask);
        __syncthreads();  // the previous column is done reading values
#pragma unroll
        for (int i = 0; i < PER; ++i) values[32 * (warp + 16 * i) + lane] = element[i];
        __syncthreads();
#pragma unroll
        for (int i = 0; i < PER; ++i) element[i] = values[phase2_point(i)];
#pragma unroll
        for (int mask = 1; mask <= 16; mask <<= 1)  // strides 32..512: the lane bits of h
#pragma unroll
            for (int i = 0; i < PER; ++i) element[i] = butterfly(element[i], lane, mask);
        if constexpr (WIDTH == 2048) {  // stride 1024: the register pairs (i, i + 1)
#pragma unroll
            for (int i = 0; i < PER; i += 2) {
                const float low = element[i], high = element[i + 1];
                element[i] = low + high;
                element[i + 1] = low - high;
            }
        }
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            out[o][i] = element[i] * normalization;
            maximum = fmaxf(maximum, fabsf(out[o][i]));
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
    __syncthreads();  // values is reused for the max
    if (lane == 0) values[warp] = maximum;
    __syncthreads();
    maximum = 0.0f;
#pragma unroll
    for (int w = 0; w < 16; ++w) maximum = fmaxf(maximum, values[w]);
    const float step = maximum > 0.0f ? maximum / 127.0f : 1.0f, inverse = 1.0f / step;

    int local[8] = {};  // per residue class: sum q0 (0..3), sum q1 (4..7)
#pragma unroll
    for (int o = 0; o < ORDER; ++o)
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int column = phase2_point(i) * ORDER + o;
            const float scaled = out[o][i] * inverse;
            const int coarse = min(127, max(-127, __float2int_rn(scaled)));
            const int fine = min(127, max(-127, __float2int_rn((scaled - coarse) * 254.0f)));
            staged[0][column] = static_cast<s8>(coarse);
            staged[1][column] = static_cast<s8>(fine);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                local[r] += column % 4 == r ? coarse : 0;
                local[4 + r] += column % 4 == r ? fine : 0;
            }
        }
#pragma unroll
    for (int k = 0; k < 8; ++k)
        for (int offset = 16; offset > 0; offset >>= 1) local[k] += __shfl_xor_sync(0xffffffffu, local[k], offset);
    if (lane == 0)
#pragma unroll
        for (int k = 0; k < 8; ++k) partial[warp][k] = local[k];
    __syncthreads();
    if constexpr (WORDS) {
        uint2* destination = reinterpret_cast<uint2*>(planes) + token * (N / 4);
        for (int i = threadIdx.x; i < N / 4; i += 512)
            destination[i] = make_uint2(reinterpret_cast<const u32*>(staged[0])[i],
                                        reinterpret_cast<const u32*>(staged[1])[i]);
    } else {
#pragma unroll
        for (int plane = 0; plane < 2; ++plane) {
            uint4* destination = reinterpret_cast<uint4*>(planes + (plane * static_cast<size_t>(tokens) + token) * N);
            for (int i = threadIdx.x; i < N / 16; i += 512)
                destination[i] = reinterpret_cast<const uint4*>(staged[plane])[i];
        }
    }
    if (threadIdx.x == 0) {
        int coarse[4] = {}, fine[4] = {};
        for (int w = 0; w < 16; ++w)
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                coarse[r] += partial[w][r];
                fine[r] += partial[w][4 + r];
            }
        write_stats(stats + token * 8, step, coarse, fine);
    }
}

#define TRANSFORM(N, WIDTH, ORDER)                                                                                  \
    extern "C" __global__ void __launch_bounds__(512)                                                                \
        transform_##N(const u16* x, const float* signs, const float* small_q, s8* planes, float* stats,          \
                      int tokens) {                                                                                  \
        transform_body<WIDTH, ORDER, false>(x, signs, small_q, planes, stats, tokens);                                \
    }                                                                                                                \
    extern "C" __global__ void __launch_bounds__(512)                                                                \
        transform_words_##N(const u16* x, const float* signs, const float* small_q, s8* q, float* stats) {           \
        transform_body<WIDTH, ORDER, true>(x, signs, small_q, q, stats, 0);                                           \
    }

TRANSFORM(5120, 1024, 5)
TRANSFORM(6144, 2048, 3)
TRANSFORM(17408, 1024, 17)

// ---------------------------------------------------------------------------------------------------------------
// gemv: one token. One CTA per 32-row group, WARPS column slices; each state feeds two dp4a.

template <int V, int T, int STEPS, int WORDS, typename Entry>
__device__ __forceinline__ void gemv_body(const uint4* __restrict__ packets, const Entry* __restrict__ entries,
                                          int packets_per_row, const float* __restrict__ rowscale,
                                          const uint2* __restrict__ q, const float* __restrict__ stats,
                                          const float* __restrict__ codebook, u16* __restrict__ y,
                                          const int* __restrict__ row_map) {
    __shared__ int partial[2][WARPS][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t group = blockIdx.x;

    int coarse = 0, fine = 0;
    for (int packet = warp; packet < packets_per_row; packet += WARPS) {
        const size_t slot = group * packets_per_row + packet;
        u32 bits[4 * WORDS];
        load_packet<WORDS>(packets, slot, lane, bits);
        const uint2* x = q + packet * STEPS * V / 4;
        decode_packet<V, T, STEPS, WORDS>(bits, __ldg(entries + slot * 32 + lane), [&](u32 levels, int g) {
            const uint2 value = __ldg(x + g);
            coarse = dp4a_us(levels, value.x, coarse);
            fine = dp4a_us(levels, value.y, fine);
        });
    }
    partial[0][warp][lane] = coarse;
    partial[1][warp][lane] = fine;
    __syncthreads();
    const size_t row = group * 32 + lane;
    const int column = row_map[row];
    if (warp != 0 || column < 0) return;
    coarse = fine = 0;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        coarse += partial[0][w][lane];
        fine += partial[1][w][lane];
    }
    y[column] = output_value(coarse, fine, stats, codebook, rowscale[row]);
}

#define GEMV(NAME, V, T, STEPS, WORDS, ENTRY)                                                                         \
    extern "C" __global__ void __launch_bounds__(32 * WARPS)                                                          \
        NAME(const uint4* packets, const ENTRY* entries, int packets_per_row, const float* rowscale, const uint2* q,    \
             const float* stats, const float* codebook, u16* y, const int* row_map) {                                 \
        gemv_body<V, T, STEPS, WORDS, ENTRY>(packets, entries, packets_per_row, rowscale, q, stats, codebook, y,       \
                                             row_map);                                                                \
    }

GEMV(gemv_v4t8, 4, 8, 16, 1, u8)
GEMV(gemv_v2t4, 2, 4, 32, 1, u16)
GEMV(gemv_v2t6, 2, 6, 64, 3, u16)

// ---------------------------------------------------------------------------------------------------------------
// mma: up to NT tokens per CTA (grid.y tiles the batch), where 2 dp4a per token per state would cost too much.
// Same CTA shape and decode as gemv (lane = row, MMA_WARPS column slices), but each warp parks 64 decoded columns of its
// 32 rows in a shared-memory slab and reads them back as m16n8k32 A fragments (ldmatrix). One u8 x s8 MMA then takes 16
// rows x 32 columns against 8 tokens of one activation plane; the B fragments come straight from q, zero past `tokens`.
// NT = 64 CTAs take 64 rows, two row groups of 4 warps each walking the same packets, so the second warp to load an
// activation finds it in L1. Warps 2i and 2i + 1 multiply both of their slabs, one against tokens 0-31 and the other
// against 32-63: every weight is decoded once for all 64 tokens while each warp keeps the accumulators of 32.

#define MMA_WARPS 8
#define SLAB 20  // u32 per slab row: 16 words of levels + 4 of padding, so ldmatrix and 16-byte stores avoid bank conflicts

__device__ __forceinline__ void mma_u8s8(int (&c)[4], const u32 (&a)[4], u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A fragment of rows 0-15 x 32 columns (bytes) at `tile`: lane i addresses row i % 16, bytes 16 * (i / 16) of it.
__device__ __forceinline__ void load_a(u32 (&a)[4], const u32* tile, int lane) {
    const u32 address = static_cast<u32>(__cvta_generic_to_shared(tile + (lane & 15) * SLAB + 4 * (lane >> 4)));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(address));
}

template <int V, int T, int STEPS, int WORDS, typename Entry, int NT>
__device__ __forceinline__ void mma_body(const uint4* __restrict__ packets, const Entry* __restrict__ entries,
                                         int packets_per_row, const float* __restrict__ rowscale,
                                         const uint2* __restrict__ q, int groups_per_token, int tokens,
                                         const float* __restrict__ stats, const float* __restrict__ codebook,
                                         u16* __restrict__ y, int y_stride, const int* __restrict__ row_map,
                                         int* __restrict__ split_sums) {
    constexpr bool PAIRED = NT == 64;
    constexpr int GROUPS = PAIRED ? 2 : 1, SLICES = MMA_WARPS / GROUPS;  // row groups per CTA, warps per row group
    constexpr int TILES = (PAIRED ? 32 : NT) / 8, PACKET_WORDS = STEPS * V / 4;  // a V2 T6 packet is two slabs
    __shared__ alignas(16) u32 slabs[MMA_WARPS][32 * SLAB];
    __shared__ int sums[2][NT][32];  // [plane][token][row], one row group at a time
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, row_group = warp / SLICES;
    const size_t group = blockIdx.x * GROUPS + row_group;
    const int first_token = blockIdx.y * NT, warp_token = PAIRED ? 32 * (warp & 1) : 0;  // this warp's first, in the CTA
    u32* slab = slabs[warp];
    for (int i = threadIdx.x; i < 2 * NT * 32; i += blockDim.x) (&sums[0][0][0])[i] = 0;

    // acc[half][tile][plane]: C fragment of rows 16 * half .. + 15 x tokens warp_token + 8 * tile .. + 7
    int acc[2][TILES][2][4] = {};
    const auto multiply = [&](const u32* levels, int first_group) {
#pragma unroll
        for (int chunk = 0; chunk < 2; ++chunk) {
            u32 a[2][4];
#pragma unroll
            for (int half = 0; half < 2; ++half) load_a(a[half], levels + 16 * half * SLAB + 8 * chunk, lane);
#pragma unroll
            for (int tile = 0; tile < TILES; ++tile) {
                const int token = first_token + warp_token + 8 * tile + (lane >> 2);
                uint2 low = make_uint2(0, 0), high = make_uint2(0, 0);
                if (token < tokens) {
                    const uint2* x = q + static_cast<size_t>(token) * groups_per_token + first_group + 8 * chunk + (lane & 3);
                    low = __ldg(x);
                    high = __ldg(x + 4);
                }
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    mma_u8s8(acc[half][tile][0], a[half], low.x, high.x);
                    mma_u8s8(acc[half][tile][1], a[half], low.y, high.y);
                }
            }
        }
    };

    // Split-K (grid.z > 1, NT < 64): CTA z takes every gridDim.z-th run of SLICES packets and adds its exact int32
    // sums into split_sums ([2 * tokens][rows], prefill_output's layout), which prefill_output then finishes.
    for (int packet = blockIdx.z * SLICES + warp % SLICES; packet < packets_per_row; packet += SLICES * gridDim.z) {
        const size_t slot = group * packets_per_row + packet;
        u32 bits[4 * WORDS];
        load_packet<WORDS>(packets, slot, lane, bits);
        u32 quad[4];
        decode_packet<V, T, STEPS, WORDS>(bits, __ldg(entries + slot * 32 + lane), [&](u32 word, int g) {
            quad[g % 4] = word;
            if (g % 4 == 3)
                reinterpret_cast<uint4*>(slab + lane * SLAB)[(g % 16) / 4] = make_uint4(quad[0], quad[1], quad[2], quad[3]);
            if (g % 16 != 15) return;
            // slab full: multiply, then let every lane finish reading before it is overwritten
            if constexpr (PAIRED) {  // the same for both warps of the pair; packet ^ 1 is the partner's
                asm volatile("bar.sync %0, 64;" ::"r"(1 + warp / 2) : "memory");
                multiply(slab, packet * PACKET_WORDS + g - 15);
                multiply(slabs[warp ^ 1], (packet ^ 1) * PACKET_WORDS + g - 15);
                asm volatile("bar.sync %0, 64;" ::"r"(1 + warp / 2) : "memory");
            } else {
                __syncwarp();
                multiply(slab, packet * PACKET_WORDS + g - 15);
                __syncwarp();
            }
        });
    }

    // One row group at a time: C fragment f holds row lane / 4 (+ 8 if f >= 2), token 2 * (lane % 4) + f % 2.
    for (int round = 0; round < GROUPS; ++round) {
        __syncthreads();
        if (row_group == round)
#pragma unroll
            for (int half = 0; half < 2; ++half)
#pragma unroll
                for (int tile = 0; tile < TILES; ++tile)
#pragma unroll
                    for (int plane = 0; plane < 2; ++plane)
#pragma unroll
                        for (int f = 0; f < 4; ++f)
                            atomicAdd(&sums[plane][warp_token + 8 * tile + 2 * (lane & 3) + (f & 1)]
                                          [16 * half + (lane >> 2) + 8 * (f >> 1)],
                                      acc[half][tile][plane][f]);
        __syncthreads();
        for (int i = threadIdx.x; i < NT * 32; i += blockDim.x) {
            const int t = i / 32, r = i % 32, token = first_token + t;
            const size_t row = (blockIdx.x * GROUPS + round) * 32 + r;
            const int column = row_map[row];
            if (token < tokens && column >= 0) {
                if (split_sums != nullptr) {
                    const size_t rows = static_cast<size_t>(gridDim.x) * GROUPS * 32;
                    atomicAdd(split_sums + static_cast<size_t>(token) * rows + row, sums[0][t][r]);
                    atomicAdd(split_sums + static_cast<size_t>(tokens + token) * rows + row, sums[1][t][r]);
                } else {
                    y[static_cast<size_t>(token) * y_stride + column] =
                        output_value(sums[0][t][r], sums[1][t][r], stats + token * 8, codebook, rowscale[row]);
                }
            }
            sums[0][t][r] = sums[1][t][r] = 0;
        }
    }
}

#define MMA(NAME, V, T, STEPS, WORDS, ENTRY, NT)                                                                      \
    extern "C" __global__ void __launch_bounds__(32 * MMA_WARPS, 2)                                                   \
        NAME(const uint4* packets, const ENTRY* entries, int packets_per_row, const float* rowscale, const uint2* q,    \
             int groups_per_token, int tokens, const float* stats, const float* codebook, u16* y, int y_stride,       \
             const int* row_map, int* split_sums) {                                                                   \
        mma_body<V, T, STEPS, WORDS, ENTRY, NT>(packets, entries, packets_per_row, rowscale, q, groups_per_token,      \
                                                tokens, stats, codebook, y, y_stride, row_map, split_sums);           \
    }
#define MMA_ALL_NT(PREFIX, V, T, STEPS, WORDS, ENTRY) \
    MMA(PREFIX##_n8, V, T, STEPS, WORDS, ENTRY, 8)    \
    MMA(PREFIX##_n16, V, T, STEPS, WORDS, ENTRY, 16)  \
    MMA(PREFIX##_n32, V, T, STEPS, WORDS, ENTRY, 32)  \
    MMA(PREFIX##_n64, V, T, STEPS, WORDS, ENTRY, 64)

MMA_ALL_NT(mma_v4t8, 4, 8, 16, 1, u8)
MMA_ALL_NT(mma_v2t4, 2, 4, 32, 1, u16)
MMA_ALL_NT(mma_v2t6, 2, 6, 64, 3, u16)

// ---------------------------------------------------------------------------------------------------------------
// Long prompts: levels writes W_rot as level + 54 bytes ([rows][columns], all in 0..111, so also valid int8) for one
// cuBLAS int8 GEMM against both activation planes; prefill_output finishes its int32 products like the kernels above.

template <int V, int T, int STEPS, int WORDS, typename Entry>
__device__ __forceinline__ void levels_body(const uint4* __restrict__ packets, const Entry* __restrict__ entries,
                                            int packets_per_row, u32* __restrict__ w, int groups_per_row) {
    const int lane = threadIdx.x & 31;
    const size_t group = blockIdx.x, row = group * 32 + lane;
    for (int packet = threadIdx.x >> 5; packet < packets_per_row; packet += blockDim.x >> 5) {
        const size_t slot = group * packets_per_row + packet;
        u32 bits[4 * WORDS];
        load_packet<WORDS>(packets, slot, lane, bits);
        uint4* out = reinterpret_cast<uint4*>(w + row * groups_per_row + packet * (STEPS * V / 4));
        u32 quad[4];
        decode_packet<V, T, STEPS, WORDS>(bits, __ldg(entries + slot * 32 + lane), [&](u32 word, int g) {
            quad[g % 4] = word;
            if (g % 4 == 3) out[g / 4] = make_uint4(quad[0], quad[1], quad[2], quad[3]);
        });
    }
}

#define LEVELS(NAME, V, T, STEPS, WORDS, ENTRY)                                                                       \
    extern "C" __global__ void NAME(const uint4* packets, const ENTRY* entries, int packets_per_row, u32* w,          \
                                    int groups_per_row) {                                                             \
        levels_body<V, T, STEPS, WORDS, ENTRY>(packets, entries, packets_per_row, w, groups_per_row);                 \
    }

LEVELS(levels_v4t8, 4, 8, 16, 1, u8)
LEVELS(levels_v2t4, 2, 4, 32, 1, u16)
LEVELS(levels_v2t6, 2, 6, 64, 3, u16)

// products: [2 * tokens][rows] int32, plane 0 of every token first, then plane 1. Grid (rows / 256, tokens).
extern "C" __global__ void prefill_output(const int* __restrict__ products, int tokens, int rows,
                                          const float* __restrict__ rowscale, const float* __restrict__ stats,
                                          const float* __restrict__ codebook, u16* __restrict__ y, int y_stride,
                                          const int* __restrict__ row_map) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x, token = blockIdx.y;
    if (row >= rows) return;
    const int column = row_map[row];
    if (column < 0) return;
    const size_t index = static_cast<size_t>(token) * rows + row;
    y[static_cast<size_t>(token) * y_stride + column] = output_value(
        products[index], products[static_cast<size_t>(tokens) * rows + index], stats + token * 8, codebook, rowscale[row]);
}

// ---------------------------------------------------------------------------------------------------------------
// The MTP drafter's layers: int8 weights W [rows][columns] with a float32 scale per row, run as one cuBLAS int8 GEMM
// against two activation planes like the long-prompt path above, but without a rotation: per token s = max |x| / 127,
// q0 = round(x / s), q1 = round((x / s - q0) * 254), and then y[m, r] = scale[r] * s * (W q0 + W q1 / 254).
// w8_quantize: grid (tokens), 256 threads. Plane 0 of token m is planes row m, plane 1 row plane_rows + m.

extern "C" __global__ void __launch_bounds__(256)
    w8_quantize(const u16* __restrict__ x, int columns, int plane_rows, s8* __restrict__ planes,
                float* __restrict__ x_scale) {
    __shared__ float warp_max[8];
    const int token = blockIdx.x, lane = threadIdx.x & 31;
    const uint4* row = reinterpret_cast<const uint4*>(x + static_cast<size_t>(token) * columns);
    float largest = 0.0f;
    for (int i = threadIdx.x; i < columns / 8; i += 256) {
        const uint4 packed = row[i];
        const u32 pairs[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
        for (int j = 0; j < 4; ++j)
            largest = fmaxf(largest, fmaxf(fabsf(__int_as_float(pairs[j] << 16)),
                                           fabsf(__int_as_float(pairs[j] & 0xFFFF0000u))));
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, offset));
    if (lane == 0) warp_max[threadIdx.x >> 5] = largest;
    __syncthreads();
    largest = warp_max[0];
    for (int w = 1; w < 8; ++w) largest = fmaxf(largest, warp_max[w]);
    const float inverse = largest > 0.0f ? 127.0f / largest : 0.0f;  // CUDA graphs pad the batch with zero rows
    if (threadIdx.x == 0) x_scale[token] = largest / 127.0f;

    uint2* coarse = reinterpret_cast<uint2*>(planes + static_cast<size_t>(token) * columns);
    uint2* fine = reinterpret_cast<uint2*>(planes + static_cast<size_t>(plane_rows + token) * columns);
    for (int i = threadIdx.x; i < columns / 8; i += 256) {
        const uint4 packed = row[i];
        const u32 pairs[4] = {packed.x, packed.y, packed.z, packed.w};
        u32 q0[2] = {}, q1[2] = {};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float value = __int_as_float(j % 2 ? pairs[j / 2] & 0xFFFF0000u : pairs[j / 2] << 16) * inverse;
            const float whole = rintf(value);
            q0[j / 4] |= (static_cast<u32>(static_cast<int>(whole)) & 0xFFu) << (8 * (j % 4));
            q1[j / 4] |= (static_cast<u32>(__float2int_rn((value - whole) * 254.0f)) & 0xFFu) << (8 * (j % 4));
        }
        coarse[i] = make_uint2(q0[0], q0[1]);
        fine[i] = make_uint2(q1[0], q1[1]);
    }
}

// products: [2 * plane_rows][rows] int32. Grid (rows / 256, tokens).
extern "C" __global__ void w8_output(const int* __restrict__ products, int plane_rows, int rows,
                                     const float* __restrict__ scale, const float* __restrict__ x_scale,
                                     u16* __restrict__ y) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x, token = blockIdx.y;
    if (row >= rows) return;
    const size_t index = static_cast<size_t>(token) * rows + row;
    const float dot = static_cast<float>(products[index]) +
                      static_cast<float>(products[static_cast<size_t>(plane_rows) * rows + index]) * (1.0f / 254.0f);
    y[index] = float_to_bf16(scale[row] * x_scale[token] * dot);
}

// ---------------------------------------------------------------------------------------------------------------
// Vocabulary surfaces: the input embedding (D4) and the output head (I3), stored per vocabulary row as
//   W[v, :] = signs * H32(values * row_scale[v] * ladder[group])      (groups of 64 columns, H32 per 32 columns)
// with H32 the normalized Walsh-Hadamard on 32 consecutive columns. D4 values are table[code byte][column % 4];
// I3 values are 2c - 7 for 3-bit codes c, packed LSB-first. Ladder indices are 4 bits per group, low nibble first.

#define HIDDEN 5120
#define FULL 0xffffffffu

// In-warp Walsh-Hadamard of one 32-column block (lane = column), in lalamo's butterfly order and rounding.
__device__ __forceinline__ float hadamard32(float x, int lane) {
#pragma unroll
    for (int half = 1; half < 32; half <<= 1) {
        const float other = __shfl_xor_sync(FULL, x, half);
        x = (lane & half) ? other - x : x + other;
    }
    return x / sqrtf(32.0f);
}

__device__ __forceinline__ float ladder_step(const u8* __restrict__ ladder_indices, const float* __restrict__ ladder,
                                             size_t row, int group) {
    const u8 packed = ladder_indices[row * (HIDDEN / 128) + group / 2];
    return ladder[(group & 1) ? packed >> 4 : packed & 15];
}

// embed_rows: grid (tokens), 8 warps; out[m] = W[ids[m], :] as bf16.
extern "C" __global__ void __launch_bounds__(256)
    embed_rows(const int* __restrict__ ids, const u8* __restrict__ codes, const float* __restrict__ row_scales,
               const float* __restrict__ ladder, const u8* __restrict__ ladder_indices,
               const signed char* __restrict__ table, const float* __restrict__ signs, u16* __restrict__ out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t row = ids[blockIdx.x];
    for (int block = warp; block < HIDDEN / 32; block += 8) {
        const int column = 32 * block + lane;
        const float value = table[4 * codes[row * (HIDDEN / 4) + column / 4] + column % 4];
        const float scale = row_scales[row] * ladder_step(ladder_indices, ladder, row, column / 64);
        const float weight = hadamard32(value * scale, lane) * signs[column];
        out[static_cast<size_t>(blockIdx.x) * HIDDEN + column] = float_to_bf16(weight);
    }
}

// head_input: x_rot = H32(signs * x) per token, as fp16. Then logits[v] = row_scale[v] * sum_j W'[v, j] x_rot[j] with
// W'[v, j] = (2c - 7) * ladder[group], because W = signs * H32(W') and H32 is symmetric.
extern "C" __global__ void __launch_bounds__(256)
    head_input(const u16* __restrict__ x, const float* __restrict__ signs, u16* __restrict__ x_rot) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t base = static_cast<size_t>(blockIdx.x) * HIDDEN;
    for (int block = warp; block < HIDDEN / 32; block += 8) {
        const int column = 32 * block + lane;
        u16 bits;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(hadamard32(bf16_to_float(x[base + column]) * signs[column], lane)));
        x_rot[base + column] = bits;
    }
}

// head_mma: logits for up to NT tokens per CTA (grid.y tiles the batch) on fp16 tensor cores. Each warp owns 32
// vocabulary rows, whose codes and ladder indices are interleaved (Sidecar.surface) so a warp's loads are contiguous.
// Per 64-column group, lane = row decodes (2c - 7) * ladder (exact in fp32, one rounding to fp16) into a shared-memory
// slab; the warp reads it back as 16 x 16 A fragments (ldmatrix) and runs m16n8k16 MMAs against x_rot
// with fp32 accumulation. The row scale is applied at the end.
#define HEAD_WARPS 8
#define HEAD_SLAB 72  // halves per slab row: 64 columns + 8 padding, so ldmatrix rows land in distinct banks

__device__ __forceinline__ void mma_f16(float (&c)[4], const u32 (&a)[4], u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two I3 weights as fp16, (2c - 7) * step: `bits` holds c0 in bits 0-2 and c1 in bits 3-5. x * 16386 puts 2 c0 at bit
// 1 and 2 c1 at bit 17, so with 0x6400 each half reads 1024 + 2c exactly; minus 1031 is 2c - 7, and the multiply by
// the ladder step is the one rounding.
__device__ __forceinline__ u32 weight_pair(u32 bits, u32 step2) {
    u32 pair = ((bits & 63u) * 16386u & 0x000E000Eu) | 0x64006400u, value;
    asm("add.rn.f16x2 %0, %1, %2;" : "=r"(value) : "r"(pair), "r"(0xE407E407u));  // 0xE407 = -1031
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(value) : "r"(value), "r"(step2));
    return value;
}

template <int NT>
__device__ __forceinline__ void head_body(const u16* __restrict__ x_rot, int tokens, const u32* __restrict__ codes,
                                          const float* __restrict__ row_scales, const float* __restrict__ ladder,
                                          const u8* __restrict__ ladder_indices, float* __restrict__ logits,
                                          int vocab) {
    constexpr int TILES = NT / 8;
    __shared__ alignas(16) u16 slabs[HEAD_WARPS][32 * HEAD_SLAB];
    __shared__ float steps[16];
    if (threadIdx.x < 16) steps[threadIdx.x] = ladder[threadIdx.x];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t block = static_cast<size_t>(blockIdx.x) * HEAD_WARPS + warp, first_row = block * 32;
    constexpr int PAIRS = HIDDEN / 128;  // groups come in pairs: 48 code bytes and one ladder byte, low nibble first
    // this lane's row within the interleaved layout: codes [pair][part][lane], ladder bytes [pair][lane]
    const uint4* row_codes = reinterpret_cast<const uint4*>(codes) + block * PAIRS * 3 * 32 + lane;
    const u8* row_ladder = ladder_indices + block * PAIRS * 32 + lane;
    const int first_token = blockIdx.y * NT;
    u16* slab = slabs[warp];
    u32 carry[6];  // the second group of a loaded pair

    // acc[half][tile]: C fragment of rows 16 * half .. + 15 x tokens 8 * tile .. + 7
    float acc[2][TILES][4] = {};
    uint4 next[3];  // the codes of the next two groups (48 bytes) and their ladder byte, loaded one step ahead
    u8 next_ladder = row_ladder[0], ladder_byte = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) next[i] = __ldg(row_codes + 32 * i);
    for (int group = 0; group < HIDDEN / 64; ++group) {
        u32 words[6];
        if (group % 2 == 0) {
            const uint4 chunk[3] = {next[0], next[1], next[2]};
            ladder_byte = next_ladder;
            if (group + 2 < HIDDEN / 64) {
#pragma unroll
                for (int i = 0; i < 3; ++i) next[i] = __ldg(row_codes + 32 * (3 * (group / 2 + 1) + i));
                next_ladder = row_ladder[32 * (group / 2 + 1)];
            }
            const u32* flat = reinterpret_cast<const u32*>(chunk);
#pragma unroll
            for (int i = 0; i < 6; ++i) words[i] = flat[i];
            carry[0] = chunk[1].z, carry[1] = chunk[1].w, carry[2] = chunk[2].x;
            carry[3] = chunk[2].y, carry[4] = chunk[2].z, carry[5] = chunk[2].w;
        } else {
#pragma unroll
            for (int i = 0; i < 6; ++i) words[i] = carry[i];
        }
        u16 step_half;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(step_half) : "f"(steps[group % 2 ? ladder_byte >> 4 : ladder_byte & 15]));
        const u32 step2 = static_cast<u32>(step_half) * 0x10001u;
        u32 packed[32];
#pragma unroll
        for (int j = 0; j < 64; j += 2) {
            const int bit = 3 * j;
            packed[j / 2] = weight_pair(__funnelshift_r(words[bit / 32], words[min(bit / 32 + 1, 5)], bit % 32), step2);
        }
        __syncwarp();  // the previous group's ldmatrix reads are done
        uint4* out = reinterpret_cast<uint4*>(slab + lane * HEAD_SLAB);
#pragma unroll
        for (int q = 0; q < 8; ++q) out[q] = make_uint4(packed[4 * q], packed[4 * q + 1], packed[4 * q + 2], packed[4 * q + 3]);
        __syncwarp();
#pragma unroll
        for (int kstep = 0; kstep < 4; ++kstep) {
            u32 a[2][4];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const u16* tile = slab + (16 * half + (lane & 15)) * HEAD_SLAB + 16 * kstep + 8 * (lane >> 4);
                const u32 address = static_cast<u32>(__cvta_generic_to_shared(tile));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                             : "=r"(a[half][0]), "=r"(a[half][1]), "=r"(a[half][2]), "=r"(a[half][3])
                             : "r"(address));
            }
            const int column = 64 * group + 16 * kstep + 2 * (lane & 3);
#pragma unroll
            for (int t = 0; t < TILES; ++t) {  // B fragment: token lane / 4 of the tile, columns 2 (lane % 4) + {0, 1, 8, 9}
                const int token = first_token + 8 * t + (lane >> 2);
                u32 b0 = 0, b1 = 0;
                if (token < tokens) {
                    const u16* x = x_rot + static_cast<size_t>(token) * HIDDEN + column;
                    b0 = *reinterpret_cast<const u32*>(x);
                    b1 = *reinterpret_cast<const u32*>(x + 8);
                }
#pragma unroll
                for (int half = 0; half < 2; ++half) mma_f16(acc[half][t], a[half], b0, b1);
            }
        }
    }

#pragma unroll
    for (int half = 0; half < 2; ++half)
#pragma unroll
        for (int t = 0; t < TILES; ++t)
#pragma unroll
            for (int f = 0; f < 4; ++f) {  // C fragment: rows lane / 4 (+ 8 for f >= 2), tokens 2 * (lane % 4) + f % 2
                const int token = first_token + 8 * t + 2 * (lane & 3) + (f & 1);
                const size_t r = first_row + 16 * half + (lane >> 2) + 8 * (f >> 1);
                if (token < tokens) logits[static_cast<size_t>(token) * vocab + r] = acc[half][t][f] * row_scales[r];
            }
}

#define HEAD(NT)                                                                                                     \
    extern "C" __global__ void __launch_bounds__(32 * HEAD_WARPS)                                                    \
        head_mma_n##NT(const u16* x_rot, int tokens, const u32* codes, const float* row_scales, const float* ladder, \
                       const u8* ladder_indices, float* logits, int vocab) {                                        \
        head_body<NT>(x_rot, tokens, codes, row_scales, ladder, ladder_indices, logits, vocab);                       \
    }
HEAD(8)
HEAD(16)
HEAD(32)
