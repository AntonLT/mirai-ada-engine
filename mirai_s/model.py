"""Qwen3.5-architecture models whose input embedding is the S package's compressed D4 surface.

vLLM builds `embed_tokens` without a quantization config, so a bf16 table (2.5 GB for Qwen3.8's vocabulary) would be
allocated whatever the checkpoint holds. These subclasses build the model with `VocabParallelEmbedding` swapped for
`Embedding` in the one module that constructs it, for the length of the constructor only. The MTP drafter gets the same
treatment: vLLM hands it the target's embedding afterwards, but it must not allocate its own copy first.
"""

from contextlib import contextmanager

import torch
from vllm.model_executor.models import qwen3_5, qwen3_5_mtp

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
