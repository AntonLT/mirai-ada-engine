# mirai_s (Ada fork)

A fork of the `mirai_s` 0.2.1 vLLM plugin shipped with
[trymirai/Qwen3.8-27B-S-experimental](https://huggingface.co/trymirai/Qwen3.8-27B-S-experimental). It is tuned on an
RTX 4070 Ti SUPER (Ada, sm_89, 66 SMs, 16 GB) and runs the model as a fast local coding sidekick. The first commit is
the shipped plugin; each later commit explains one change.

- **DFlash 2 drafter** (`z-lab/Qwen3.8-27B-DFlash2`):
  - The drafter shares the target's compressed embedding and head, with no bf16 vocabulary copies.
  - Its linears are staged in host memory and kept on the GPU as int8 (`"quantization": "mirai_s_w8"`) or int4
    Marlin (`"mirai_s_w4"`).
  - Its candidates come from the 98k-row draft vocabulary.
- **KV-cache groups:** the group size with the least padding for hybrid target + drafter layer counts.
- **Split-K** for mma launches smaller than one wave on the GPU. The int32 sums are exact.
- **Activation rotation:** each row is staged in shared memory with coalesced reads, and the ±1 signs are read as a
  bitmask.

Outputs are bit-identical to the shipped plugin (`test_against_shipped.py`). Measured with
`qwen3.8-s/speedcheck.py`: 242 tok/s decoding code and 108 on prose, against 155 / 88 for the shipped plugin with the
model card's MTP command.

```bash
uv pip install -e .   # into the vLLM 0.30 environment
vllm serve qwen3.8-s/vllm --max-num-seqs 2 --max-num-batched-tokens 1024 \
  --speculative-config '{"method":"dflash","model":"<DFlash2 dir>","num_speculative_tokens":7,"quantization":"mirai_s_w4"}'
```

Tools (run from the directory holding `qwen3.8-s/`):
- `microbench.py`: per-role kernel times.
- `sweep_splits.py`: split-K factors per role.
- `bench_quantize.py`: rotation and quantization.
- `test_exact.py`: split vs unsplit.
- `test_against_shipped.py SHIPPED_DIR`: comparison with the shipped plugin.
