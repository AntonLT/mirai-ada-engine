"""Mirai S linear layers on CUDA: runtime-compiled kernels (NVRTC) behind one torch custom op.

A layer is a list of blocks sharing one input. Each block is one trellis matrix in a single tape format; its
`row_map` sends each of its rows to an output column (or -1 for padding), so kernels write straight into the output
in the order the caller expects.

Per call the input is rotated and quantized to two int8 planes once, then each block runs one of (see kernels.cu):
  1 token  dp4a GEMV;  <= MMA_TOKENS  int8 tensor-core GEMV;  longer  int8 levels + cuBLAS int8 GEMM.
All three give bit-identical outputs per token.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch

SOURCE = (Path(__file__).parent / "kernels.cu").read_text()
GEMV_THREADS, MMA_THREADS = 32 * 16, 32 * 8  # the kernels' WARPS and MMA_WARPS
FORMATS = {"v4t8": (4, 8, 16, 1), "v2t4": (2, 4, 32, 1), "v2t6": (2, 6, 64, 3)}  # V, T, steps, words per packet
MMA_TOKENS = 384  # mma_<format>_n8 / n16 / n32 / n64; beyond ~512 tokens levels + cuBLAS is faster (RTX 3090)
HIDDEN = 5120  # the vocabulary surfaces' width (kernels.cu HIDDEN)


@cache
def module() -> "torch.cuda._utils._CudaModule":
    """All kernels, compiled once for this GPU (NVRTC builds the whole file whichever kernel is named)."""
    binary, _ = torch.cuda._utils._nvrtc_compile(SOURCE, "gemv_v4t8", nvcc_options=["-std=c++17"])
    return torch.cuda._utils._cuda_load_module(binary)


@cache
def kernel(name: str) -> "torch.cuda._utils._CudaKernel":
    return getattr(module(), name)


def compile_all() -> None:
    """Compile and look up every kernel up front, so nothing compiles during CUDA graph capture."""
    kinds = ("rotate_columns", "quantize_rows", "transform", "transform_words")
    names = [f"{kind}_{n}" for kind in kinds for n in (5120, 6144, 17408)]
    for fmt in FORMATS:
        names += [f"gemv_{fmt}", f"levels_{fmt}"] + [f"mma_{fmt}_n{n}" for n in (8, 16, 32, 64)]
    names += ["prefill_output", "embed_rows", "head_input", "w8_quantize", "w8_output"]
    names += [f"head_mma_n{n}" for n in (8, 16, 32)]
    for name in names:
        kernel(name)


@dataclass(frozen=True)
class Block:
    format: str
    packets: torch.Tensor  # uint8, groups x packets_per_row x words x 32 x 16 bytes
    entries: torch.Tensor  # uint8 (v4t8) or int16, groups x packets_per_row x 32
    rowscale: torch.Tensor  # float32, rows padded to a multiple of 32
    codebook: torch.Tensor  # float32 [c, d0, d1, d2, d3]
    columns: int
    row_map: torch.Tensor  # int32, output column of each of this block's rows, -1 for padding

    def __post_init__(self) -> None:
        assert self.packets_per_row % 8 == 0, "mma n64 pairs warps 2i and 2i + 1, so each warp needs as many packets"

    @property
    def rows(self) -> int:
        return self.rowscale.numel()

    @property
    def packets_per_row(self) -> int:
        width, _, steps, _ = FORMATS[self.format]
        return self.columns // (steps * width)


@dataclass(frozen=True)
class Layer:
    blocks: list[Block]
    signs: torch.Tensor
    small_q: torch.Tensor
    out_features: int


def quantize(x: torch.Tensor, layer: Layer) -> tuple[torch.Tensor, torch.Tensor]:
    """x_rot as two int8 planes and per-token stats [tokens, 8]. Up to MMA_TOKENS tokens the planes are q [tokens,
    columns / 4, 2] int32 words (plane 0, plane 1 of four columns) for the gemv and mma kernels; longer inputs get the
    int8 GEMM's [2 * tokens, columns] (plane 0's rows, then plane 1's). Up to 16 tokens the split kernels fill the GPU
    better than one CTA per token."""
    tokens, columns = x.shape
    stats = torch.empty(tokens, 8, dtype=torch.float32, device=x.device)
    if tokens > MMA_TOKENS:
        planes = torch.empty(2 * tokens, columns, dtype=torch.int8, device=x.device)
        kernel(f"transform_{columns}")(grid=(tokens, 1, 1), block=(512, 1, 1),
                                       args=[x, layer.signs, layer.small_q, planes, stats, tokens])
        return planes, stats
    q = torch.empty(tokens, columns // 4, 2, dtype=torch.int32, device=x.device)
    if tokens > 16:
        kernel(f"transform_words_{columns}")(grid=(tokens, 1, 1), block=(512, 1, 1),
                                             args=[x, layer.signs, layer.small_q, q, stats])
        return q, stats
    order = layer.small_q.shape[0]
    rotated = torch.empty(tokens, columns, dtype=torch.float32, device=x.device)
    column_max = torch.empty(tokens, order, dtype=torch.float32, device=x.device)
    width = columns // order
    kernel(f"rotate_columns_{columns}")(grid=(order, tokens, 1), block=(min(1024, width // 2), 1, 1),
                                        args=[x, layer.signs, layer.small_q, rotated, column_max])
    kernel(f"quantize_rows_{columns}")(grid=(tokens, 1, 1), block=(1024, 1, 1), args=[rotated, column_max, q, stats])
    return q, stats


# Layers fed the very same input (vLLM's in_proj_qkvz and in_proj_ba) share one quantization. Holding the input keeps
# its identity unique, so a new tensor can never be mistaken for it.
_last_quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None


def quantized_input(x: torch.Tensor, layer: Layer) -> tuple[torch.Tensor, torch.Tensor]:
    global _last_quantized
    if _last_quantized is not None and _last_quantized[0] is x and _last_quantized[1] is layer.signs:
        return _last_quantized[2], _last_quantized[3]
    q, stats = quantize(x, layer)
    _last_quantized = (x, layer.signs, q, stats)
    return q, stats


def linear(x: torch.Tensor, layer: Layer) -> torch.Tensor:
    assert x.dtype == torch.bfloat16 and x.is_contiguous(), "vLLM hands Mirai S layers contiguous bf16 activations"
    tokens = x.shape[0]
    out = torch.empty(tokens, layer.out_features, dtype=torch.bfloat16, device=x.device)
    q, stats = quantized_input(x, layer)  # longer inputs: q is the planes matrix
    for block in layer.blocks:
        tape = [block.packets, block.entries, block.packets_per_row]
        epilogue = [stats, block.codebook, out, out.stride(0), block.row_map]
        if tokens == 1:
            kernel(f"gemv_{block.format}")(grid=(block.rows // 32, 1, 1), block=(GEMV_THREADS, 1, 1),
                                           args=tape + [block.rowscale, q, stats, block.codebook, out, block.row_map])
        elif tokens <= MMA_TOKENS:  # an n64 CTA takes 64 rows: in_proj_ba's 96 get two 32-token tiles instead
            tile = 8 if tokens <= 8 else 16 if tokens <= 16 else 32 if tokens <= 32 or block.rows % 64 else 64
            rows_per_cta = 64 if tile == 64 else 32
            kernel(f"mma_{block.format}_n{tile}")(grid=(block.rows // rows_per_cta, -(-tokens // tile), 1),
                                                  block=(MMA_THREADS, 1, 1),
                                                  args=tape + [block.rowscale, q, block.columns // 4, tokens] + epilogue)
        else:
            levels = torch.empty(block.rows, block.columns, dtype=torch.int8, device=x.device)
            kernel(f"levels_{block.format}")(grid=(block.rows // 32, 1, 1), block=(GEMV_THREADS, 1, 1),
                                             args=tape + [levels, block.columns // 4])
            products = torch._int_mm(q, levels.t())
            kernel("prefill_output")(grid=(-(-block.rows // 256), tokens, 1), block=(256, 1, 1),
                                     args=[products, tokens, block.rows, block.rowscale] + epilogue)
    return out


# torch.compile and CUDA graphs see one opaque op; the layers live in a registry indexed by an int.
LAYERS: list[Layer] = []


def register_layer(layer: Layer) -> int:
    LAYERS.append(layer)
    return len(LAYERS) - 1


@torch.library.custom_op("mirai_s::linear", mutates_args=())
def linear_op(x: torch.Tensor, layer: int) -> torch.Tensor:
    return linear(x, LAYERS[layer])


@linear_op.register_fake
def _(x: torch.Tensor, layer: int) -> torch.Tensor:
    return x.new_empty(x.shape[0], LAYERS[layer].out_features, dtype=torch.bfloat16)


# The MTP drafter's layers, indexed like LAYERS: int8 weights [rows, columns] and a float32 scale per row.
DRAFTER: list[tuple[torch.Tensor, torch.Tensor]] = []


def register_drafter_layer(weight: torch.Tensor, scale: torch.Tensor) -> int:
    DRAFTER.append((weight, scale))
    return len(DRAFTER) - 1


@torch.library.custom_op("mirai_s::drafter_linear", mutates_args=())
def drafter_linear_op(x: torch.Tensor, layer: int) -> torch.Tensor:
    assert x.dtype == torch.bfloat16 and x.is_contiguous(), "vLLM hands the drafter contiguous bf16 activations"
    weight, scale = DRAFTER[layer]
    (tokens, columns), rows = x.shape, weight.shape[0]
    plane_rows = max(tokens, 9)  # torch._int_mm takes more than 16 rows
    planes = torch.empty(2 * plane_rows, columns, dtype=torch.int8, device=x.device)
    x_scale = torch.empty(tokens, dtype=torch.float32, device=x.device)
    kernel("w8_quantize")(grid=(tokens, 1, 1), block=(256, 1, 1), args=[x, columns, plane_rows, planes, x_scale])
    products = torch._int_mm(planes, weight.t())
    out = torch.empty(tokens, rows, dtype=torch.bfloat16, device=x.device)
    kernel("w8_output")(grid=(-(-rows // 256), tokens, 1), block=(256, 1, 1),
                        args=[products, plane_rows, rows, scale, x_scale, out])
    return out


@drafter_linear_op.register_fake
def _(x: torch.Tensor, layer: int) -> torch.Tensor:
    return x.new_empty(x.shape[0], DRAFTER[layer][0].shape[0], dtype=torch.bfloat16)


# Vocabulary surfaces (kernels.cu): stateless ops, the stored tensors are arguments.
@torch.library.custom_op("mirai_s::embed", mutates_args=())
def embed_op(ids: torch.Tensor, codes: torch.Tensor, row_scales: torch.Tensor, ladder: torch.Tensor,
             ladder_indices: torch.Tensor, table: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    ids = ids.to(torch.int32)
    out = torch.empty(ids.shape[0], HIDDEN, dtype=torch.bfloat16, device=ids.device)
    kernel("embed_rows")(grid=(ids.shape[0], 1, 1), block=(256, 1, 1),
                         args=[ids, codes, row_scales, ladder, ladder_indices, table, signs, out])
    return out


@embed_op.register_fake
def _(ids, codes, row_scales, ladder, ladder_indices, table, signs):
    return ids.new_empty(ids.shape[0], HIDDEN, dtype=torch.bfloat16)


def head_logits(x: torch.Tensor, codes: torch.Tensor, row_scales: torch.Tensor, ladder: torch.Tensor,
                ladder_indices: torch.Tensor, signs: torch.Tensor, row_ranges: list[tuple[int, int]],
                out: torch.Tensor) -> torch.Tensor:
    """Writes the fp32 logits of bf16 hidden states x [tokens, HIDDEN] for the head rows [first, first + rows) of each
    range into out [tokens, vocab]; codes and ladder_indices are interleaved per 32 rows, so ranges start on those."""
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    tokens, vocab = out.shape
    x_rot = torch.empty(tokens, HIDDEN, dtype=torch.float16, device=x.device)
    kernel("head_input")(grid=(tokens, 1, 1), block=(256, 1, 1), args=[x, signs, x_rot])
    tile = 8 if tokens <= 8 else 16 if tokens <= 16 else 32
    for first, rows in row_ranges:
        assert first % 256 == 0 and rows % 256 == 0, "head_mma takes 8 warps x 32 rows per CTA"
        kernel(f"head_mma_n{tile}")(grid=(rows // 256, -(-tokens // tile), 1), block=(256, 1, 1),
                                    args=[x_rot, tokens, codes[first // 32:], row_scales[first:], ladder,
                                          ladder_indices[first // 32:], out[:, first:], vocab])
    return out


@torch.library.custom_op("mirai_s::logits", mutates_args=())
def logits_op(x: torch.Tensor, codes: torch.Tensor, row_scales: torch.Tensor, ladder: torch.Tensor,
              ladder_indices: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    vocab = row_scales.shape[0]
    out = torch.empty(x.shape[0], vocab, dtype=torch.float32, device=x.device)
    return head_logits(x, codes, row_scales, ladder, ladder_indices, signs, [(0, vocab)], out)


@logits_op.register_fake
def _(x, codes, row_scales, ladder, ladder_indices, signs):
    return x.new_empty(x.shape[0], codes.shape[0], dtype=torch.float32)


# The MTP drafter's head: ids below DRAFT_VOCAB, 99.8% of generated tokens in English prose, code and reasoning, for a
# third of the full head's time. The other logits are -inf: greedy drafts never pick them and sampled drafts give them
# probability 0, so verification stays exact and only the acceptance rate can move (2.848 -> 2.845 tokens per step).
# The chat control tokens (<|im_end|>, </think>, <tool_call>) sit at the top of the vocabulary: each costs one draft
# position, while a second launch for them would cost as much as this one.
DRAFT_VOCAB = 98304


@torch.library.custom_op("mirai_s::drafter_logits", mutates_args=())
def drafter_logits_op(x: torch.Tensor, codes: torch.Tensor, row_scales: torch.Tensor, ladder: torch.Tensor,
                      ladder_indices: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    vocab = row_scales.shape[0]
    out = torch.full((x.shape[0], vocab), float("-inf"), dtype=torch.float32, device=x.device)
    return head_logits(x, codes, row_scales, ladder, ladder_indices, signs, [(0, DRAFT_VOCAB)], out)


@drafter_logits_op.register_fake
def _(x, codes, row_scales, ladder, ladder_indices, signs):
    return x.new_empty(x.shape[0], codes.shape[0], dtype=torch.float32)
