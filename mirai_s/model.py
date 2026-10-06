"""Qwen3.5-architecture models whose input embedding is the S package's compressed D4 surface.

vLLM builds `embed_tokens` without a quantization config, so a bf16 table (2.5 GB for Qwen3.8's vocabulary) would be
allocated whatever the checkpoint holds. These subclasses build the model with `VocabParallelEmbedding` swapped for
`Embedding` in the one module that constructs it, for the length of the constructor only. The MTP drafter gets the same
treatment: vLLM hands it the target's embedding afterwards, but it must not allocate its own copy first.
"""

from contextlib import contextmanager

import torch
from vllm.model_executor.models import qwen3_5, qwen3_5_mtp, qwen3_dflash, qwen3_dflash2

from mirai_s.ops import HIDDEN, drafter_logits_op, embed_op
from mirai_s.quant import MiraiSConfig, open_sidecar


class Embedding(torch.nn.Module):
    def __init__(self, sidecar_path: str) -> None:
        super().__init__()
        surface = open_sidecar(sidecar_path).surface("embedding", torch.device("cuda", torch.cuda.current_device()))
        self.tensors = [surface[name] for name in ("codes", "row_scales", "ladder", "ladder_indices", "table", "signs")]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return embed_op(ids, *self.tensors)


@contextmanager
def compressed_embedding(module, vllm_config):
    """Inside the block, `module.VocabParallelEmbedding(vocab, hidden)` builds an `Embedding`."""
    config = vllm_config.quant_config
    if not isinstance(config, MiraiSConfig):
        yield
        return
    original = module.VocabParallelEmbedding

    def build(num_embeddings: int, embedding_dim: int) -> Embedding:
        assert embedding_dim == HIDDEN, embedding_dim
        embedding = Embedding(config.sidecar_path)
        assert embedding.tensors[0].shape[0] == num_embeddings, "vocabulary size differs from the sidecar"
        return embedding

    module.VocabParallelEmbedding = build
    try:
        yield
    finally:
        module.VocabParallelEmbedding = original


class MiraiSQwen3_5ForCausalLM(qwen3_5.Qwen3_5ForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        with compressed_embedding(qwen3_5, vllm_config):
            super().__init__(vllm_config=vllm_config, prefix=prefix)


class MiraiSQwen3_5MTP(qwen3_5_mtp.Qwen3_5MTP):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        with compressed_embedding(qwen3_5_mtp, vllm_config):
            super().__init__(vllm_config=vllm_config, prefix=prefix)

    def compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0) -> torch.Tensor:
        """The target's head, shared with the drafter, over the drafter's rows only (ops.DRAFT_VOCAB)."""
        logits = drafter_logits_op(hidden_states, *self.lm_head.mirai_head)
        return logits[:, : self.logits_processor.org_vocab_size]


DRAFTER_METHODS = ("mirai_s_w8", "mirai_s_w4")  # quant.py: the host-staged drafter linears


class Shared(torch.nn.Module):
    """Stands in for a drafter's embedding and head until vLLM replaces them with the target's. `weight` is None so
    vLLM's sharing checks, which compare weights only when both are tensors, pass over it."""

    weight = None


@contextmanager
def shared_vocabulary(module):
    """Inside the block, the module's `VocabParallelEmbedding` and `ParallelLMHead` build `Shared` placeholders."""
    names = ("VocabParallelEmbedding", "ParallelLMHead")
    originals = [getattr(module, name) for name in names]
    for name in names:
        setattr(module, name, lambda *args, **kwargs: Shared())
    try:
        yield
    finally:
        for name, original in zip(names, originals):
            setattr(module, name, original)


class MiraiSDFlash2(qwen3_dflash2.DFlash2Qwen3ForCausalLM):
    """DFlash 2 drafter for a Mirai S target: the checkpoint carries neither embedding nor head (it shares the
    target's), so with host-staged linears (`"quantization"` of DRAFTER_METHODS) it builds no bf16 vocabulary table at all.
    Any other configuration behaves exactly like vLLM's class."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        if vllm_config.speculative_config.draft_model_config.quantization not in DRAFTER_METHODS:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            return
        with shared_vocabulary(qwen3_dflash):
            super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        if self.model.quant_config is not None and self.model.quant_config.get_name() in DRAFTER_METHODS:
            # vLLM fuses every layer's K/V projection into one bf16 GEMM over the context. It is built from the
            # weights as loaded, which the int8 method stages in host memory; that small fused copy belongs on the GPU.
            self.model._fused_kv_weight = self.model._fused_kv_weight.to(torch.device("cuda", torch.cuda.current_device()))
        return loaded
