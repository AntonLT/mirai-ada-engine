"""Every activation quantization and linear output must match the shipped kernels (git HEAD~N's kernels.cu, given as
a path) bit for bit. Runs both kernel builds in one process by swapping the NVRTC source.

    python engine/test_against_shipped.py shipped_kernels.cu [sidecar]
"""
import sys, torch
from mirai_s.quant import open_sidecar
from mirai_s import ops

shipped = open(sys.argv[1]).read()
side = open_sidecar(sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-s/vllm/trellis.mirai")
dev = torch.device("cuda")
current = ops.SOURCE


def use(source):
    ops.SOURCE = source
    ops.module.cache_clear()
    ops.kernel.cache_clear()
    ops.resident_ctas.cache_clear() if hasattr(ops, "resident_ctas") else None
    ops._last_quantized = None
    ops.compile_all()


cases, seen = [], set()
torch.manual_seed(2)
for key in side.layers:
    role = key.split(".", 2)[2]
    if role in seen:
        continue
    seen.add(role)
    layer = side.layer(key, dev)
    for tokens in (1, 2, 8, 11, 16, 32, 64):
        cases.append((role, tokens, layer, 3 * torch.randn(tokens, side.layers[key]["in_features"], dtype=torch.bfloat16, device=dev)))


def outputs():
    results = []
    for role, tokens, layer, x in cases:
        ops._last_quantized = None
        results.append(ops.linear(x, layer).clone())
    return results


# The shipped kernels have no split-K (they ignore grid.z), so the reference runs unsplit.
waves = ops.SPLIT_WAVES
ops.SPLIT_WAVES = 0
use(shipped)
reference = outputs()
ops.SPLIT_WAVES = waves
use(current)
for (role, tokens, _, _), a, b in zip(cases, reference, outputs()):
    assert torch.equal(a, b), f"{role} tokens={tokens}: output differs from the shipped kernels"
print(f"bit-identical to the shipped kernels: {len(cases)} (role, tokens) cases")
