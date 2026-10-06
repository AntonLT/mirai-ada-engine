"""Every linear output must match the shipped plugin bit for bit, at decode-verify and prefill token counts.

    python engine/test_against_shipped.py SHIPPED_DIR [sidecar]

SHIPPED_DIR holds the shipped `mirai_s` package (e.g. `git archive <first commit> mirai_s | tar -x -C DIR`). It runs
in a subprocess on the same seeded inputs; this process runs the working tree's package and compares.
"""
import os, subprocess, sys, tempfile, torch

SIDECAR = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-s/vllm/trellis.mirai"
TOKENS = (1, 2, 8, 11, 16, 32, 64, 400)


def run(out_path: str) -> None:
    from mirai_s.quant import open_sidecar
    from mirai_s import ops

    side, dev, seen, results = open_sidecar(SIDECAR), torch.device("cuda"), set(), []
    ops.compile_all()
    generator = torch.Generator(device="cpu").manual_seed(2)
    for key in side.layers:
        role = key.split(".", 2)[2]
        if role in seen:
            continue
        seen.add(role)
        layer = side.layer(key, dev)
        for tokens in TOKENS:
            x = 3 * torch.randn(tokens, side.layers[key]["in_features"], generator=generator).to(torch.bfloat16)
            ops._last_quantized = None
            results.append((role, tokens, ops.linear(x.to(dev), layer).cpu()))
    torch.save(results, out_path)


if __name__ == "__main__" and os.environ.get("MIRAI_REFERENCE_OUT"):
    run(os.environ["MIRAI_REFERENCE_OUT"])
elif __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        shipped, current = os.path.join(tmp, "shipped.pt"), os.path.join(tmp, "current.pt")
        env = dict(os.environ, PYTHONPATH=os.path.abspath(sys.argv[1]), MIRAI_REFERENCE_OUT=shipped)
        subprocess.run([sys.executable, os.path.abspath(__file__), *sys.argv[1:]], env=env, check=True)
        run(current)
        reference, ours = torch.load(shipped), torch.load(current)
        for (role, tokens, a), (_, _, b) in zip(reference, ours, strict=True):
            assert torch.equal(a, b), f"{role} tokens={tokens}: output differs from the shipped plugin"
        print(f"bit-identical to the shipped plugin: {len(ours)} (role, tokens) cases")
