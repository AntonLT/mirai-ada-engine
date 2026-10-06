"""Split-K must not change a single bit: every role's first layer at decode-verify token counts, split vs unsplit.

    python engine/test_exact.py [sidecar]
"""
import sys, torch
from mirai_s.quant import open_sidecar
from mirai_s import ops

side = open_sidecar(sys.argv[1] if len(sys.argv) > 1 else "qwen3.8-s/vllm/trellis.mirai")
dev = torch.device("cuda")
ops.compile_all()
torch.manual_seed(0)
seen, checked = set(), 0
for key in side.layers:
    role = key.split(".", 2)[2]
    if role in seen:
        continue
    seen.add(role)
    layer = side.layer(key, dev)
    for tokens in (2, 5, 8, 11, 16, 24, 32, 64, 100):
        x = torch.randn(tokens, side.layers[key]["in_features"], dtype=torch.bfloat16, device=dev)
        waves = ops.SPLIT_WAVES
        ops.SPLIT_WAVES = 0
        reference = ops.linear(x, layer)
        ops.SPLIT_WAVES = waves
        ops._last_quantized = None
        split = ops.linear(x, layer)
        ops._last_quantized = None
        assert torch.equal(reference, split), f"{role} tokens={tokens}: split output differs"
        checked += 1
print(f"bit-identical: {checked} (role, tokens) cases")
