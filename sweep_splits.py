"""Per-role split-K sweep at decode-verify token counts: median time of all layers of a role per split factor."""
import statistics, sys, torch
from mirai_s.quant import open_sidecar
from mirai_s import ops

side = open_sidecar("qwen3.8-s/vllm/trellis.mirai"); ops.compile_all(); dev = torch.device("cuda")
tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 8
forced = {"value": 1}
original = ops.split_count
ops.split_count = lambda block, tile, token_tiles, device: min(forced["value"], max(1, block.packets_per_row // 8))
for role in ["linear_attn.in_proj_ba", "linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj",
             "self_attn.qkv_proj", "linear_attn.in_proj_qkvz", "mlp.gate_up_proj"]:
    keys = [k for k in side.layers if k.split(".", 2)[2] == role]
    layers = [side.layer(k, dev) for k in keys]
    xs = [torch.randn(tokens, side.layers[k]["in_features"], dtype=torch.bfloat16, device=dev) for k in keys]
    row = []
    for splits in (1, 2, 3, 4, 6, 8, 12):
        forced["value"] = splits
        for x, layer in zip(xs, layers):
            ops._last_quantized = None; ops.linear(x, layer)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for x, layer in zip(xs, layers):
                ops._last_quantized = None; ops.linear(x, layer)
        times = []
        for _ in range(7):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(5): g.replay()
            e.record(); torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1000 / (5 * len(layers)))
        row.append((splits, statistics.median(times)))
        del g
    best = min(row, key=lambda r: r[1])
    print(f"{role:26s} t={tokens} " + "  ".join(f"s{s}:{t:5.1f}" for s, t in row) + f"   best s{best[0]}", flush=True)
    del layers, xs; torch.cuda.empty_cache()
