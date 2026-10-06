"""Time the activation rotation + quantization per linear call (split kernels vs one CTA per token)."""
import torch
from mirai_s.quant import open_sidecar
from mirai_s import ops
side = open_sidecar("qwen3.8-s/vllm/trellis.mirai"); ops.compile_all(); dev = torch.device("cuda")
layers = {}
for k in side.layers:
    c = side.layers[k]["in_features"]
    if c not in layers: layers[c] = side.layer(k, dev)
def timeit(fn, n=200):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n): fn()
    g.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); g.replay(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / n
for c, layer in layers.items():
    for t in (1, 8, 16):
        x = torch.randn(t, c, dtype=torch.bfloat16, device=dev)
        split = timeit(lambda: ops.quantize(x, layer))
        q = torch.empty(t, c // 4, 2, dtype=torch.int32, device=dev); st = torch.empty(t, 8, dtype=torch.float32, device=dev)
        one = timeit(lambda: ops.kernel(f"transform_words_{c}")(grid=(t, 1, 1), block=(512, 1, 1), args=[x, layer.signs, layer.small_q, q, st]))
        print(f"columns {c:5d} t={t:2d}: split kernels {split:5.1f} us   one CTA/token {one:5.1f} us")
print("--- split kernel breakdown")
for c, layer in layers.items():
    for t in (8, 16):
        x = torch.randn(t, c, dtype=torch.bfloat16, device=dev)
        order = layer.small_q.shape[0]; width = c // order
        rot = torch.empty(t, c, dtype=torch.float32, device=dev); cm = torch.empty(t, order, dtype=torch.float32, device=dev)
        q = torch.empty(t, c // 4, 2, dtype=torch.int32, device=dev); st = torch.empty(t, 8, dtype=torch.float32, device=dev)
        r = timeit(lambda: ops.kernel(f"rotate_columns_{c}")(grid=(order, t, 1), block=(min(1024, width // 2), 1, 1), args=[x, layer.signs, layer.small_q, rot, cm]))
        qq = timeit(lambda: ops.kernel(f"quantize_rows_{c}")(grid=(t, 1, 1), block=(1024, 1, 1), args=[rot, cm, q, st]))
        print(f"columns {c:5d} t={t:2d}: rotate_columns {r:5.1f} us  quantize_rows {qq:5.1f} us")
