"""Microbenchmark of the Mirai S linear kernels on this GPU, every layer of a role in turn (as a forward pass reads
them, so L2 cannot serve repeats): time per layer and effective weight bandwidth, at decode-verify token counts.

    python engine/microbench.py [sidecar] [tokens,...] [role-substring]
"""
import sys, torch
from mirai_s.quant import open_sidecar
from mirai_s import ops

path = sys.argv[1] if len(sys.argv) > 1 else "qwen3.8-s/vllm/trellis.mirai"
tokens_list = [int(t) for t in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1", "8", "16"])]
only = sys.argv[3] if len(sys.argv) > 3 else ""
side = open_sidecar(path)
dev = torch.device("cuda")
ops.compile_all()
roles = ["linear_attn.in_proj_qkvz", "linear_attn.in_proj_ba", "linear_attn.out_proj", "self_attn.qkv_proj",
         "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"]
total = {t: 0.0 for t in tokens_list}
for role in roles:
    if only not in role:
        continue
    keys = [k for k in side.layers if k.split(".", 2)[2] == role]
    layers = [side.layer(k, dev) for k in keys]
    nbytes = sum(b.packets.numel() + b.entries.numel() * b.entries.element_size() for b in layers[0].blocks)
    columns = side.layers[keys[0]]["in_features"]
    for t in tokens_list:
        xs = [torch.randn(t, columns, dtype=torch.bfloat16, device=dev) for _ in layers]
        for x, layer in zip(xs, layers):
            ops.linear(x, layer)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for x, layer in zip(xs, layers):
                ops.linear(x, layer)
        g.replay(); torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            g.replay()
        end.record(); torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000 / (10 * len(layers))
        total[t] += us * len(layers)
        print(f"{role:26s} x{len(layers):2d} {nbytes/1e6:6.1f} MB  t={t:3d}  {us:7.1f} us  {nbytes/us/1e3:6.0f} GB/s", flush=True)
    del layers, xs
    torch.cuda.empty_cache()
for t in tokens_list:
    print(f"quantized linears, t={t}: {total[t]/1000:.2f} ms per forward")
