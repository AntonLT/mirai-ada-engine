"""vLLM quantization method "mirai_s": Mirai S trellis linears and the I3 output head, loaded from the model's
`trellis.mirai` sidecar.

The checkpoint's own safetensors hold only the small dense tensors (norms, convolutions, the MTP layer), which vLLM
loads as usual. Each quantized layer reads its blocks from the sidecar after loading, keyed by `layers.<n>.<role>`,
so vLLM's per-shard weight loaders never see the trellis format. The input embedding is handled in model.py.
"""

import json
import re
from functools import cache
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from mirai_s.ops import (Block, Layer, compile_all, drafter_linear_op, linear_op, logits_op, register_drafter_layer,
                         register_layer)

SIDECAR = "trellis.mirai"
ROLES = (
    "linear_attn.in_proj_qkvz",
    "linear_attn.in_proj_ba",
    "linear_attn.out_proj",
    "self_attn.qkv_proj",
    "self_attn.o_proj",
    "mlp.gate_up_proj",
    "mlp.down_proj",
)
# Main-model linears only: the MTP drafter's `mtp.layers.0...` linears are dense bf16 in the checkpoint.
LAYER = re.compile(r"^model\.layers\.(\d+)\.(" + "|".join(re.escape(role) for role in ROLES) + r")$")


def resolve_sidecar(model: str, revision: str | None) -> Path:
    local = Path(model) / SIDECAR
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model, SIDECAR, revision=revision))


class Sidecar:
    """GPU tensors of the sidecar, shared rotations and codebooks loaded once."""

    def __init__(self, path: Path) -> None:
        self.file = safe_open(str(path), framework="pt", device="cpu")
        self.layers = json.loads(self.file.metadata()["layers"])

    @cache
    def tensor(self, name: str, device: torch.device) -> torch.Tensor:
        return self.file.get_tensor(name).to(device)

    def surface(self, name: str, device: torch.device) -> dict[str, torch.Tensor]:
        """The `embedding` or `head` vocabulary surface; ladder as float32 (exact from fp16). The head's codes and ladder
        indices are interleaved per 32 rows, [row block][group pair][part][row][16 bytes] and [row block][group pair]
        [row], so a warp of head_mma (lane = row) loads contiguous bytes."""
        parts = ("codes", "row_scales", "ladder", "ladder_indices", "signs") + (("table",) if name == "embedding" else ())
        surface = {part: self.file.get_tensor(f"{name}.{part}") for part in parts}
        surface["ladder"] = surface["ladder"].float()
        if name == "head":  # interleaved on the CPU, so the GPU never holds a second copy
            blocks, pairs = surface["codes"].shape[0] // 32, surface["ladder_indices"].shape[1]
            surface["codes"] = surface["codes"].view(blocks, 32, pairs, 3, 16).permute(0, 2, 3, 1, 4).contiguous()
            surface["ladder_indices"] = surface["ladder_indices"].view(blocks, 32, pairs).permute(0, 2, 1).contiguous()
        return {part: tensor.to(device) for part, tensor in surface.items()}

    def layer(self, key: str, device: torch.device) -> Layer:
        meta = self.layers[key]
        columns, out_features = meta["in_features"], meta["out_features"]
        rowscales = [self.tensor(f"{key}.{index}.rowscale", device) for index in range(len(meta["blocks"]))]
        internal_rows = sum(rowscale.numel() for rowscale in rowscales)
        # row_map[internal row] = output column; perm lists the internal row of every output column.
        row_map = torch.full((internal_rows,), -1, dtype=torch.int32)
        sources = self.file.get_tensor(f"{key}.perm").long() if meta["has_perm"] else torch.arange(out_features)
        row_map[sources] = torch.arange(out_features, dtype=torch.int32)
        row_map = row_map.to(device)
        blocks = [
            Block(
                format=block["format"],
                packets=self.tensor(f"{key}.{index}.packets", device),
                entries=self.tensor(f"{key}.{index}.entries", device),
                rowscale=rowscales[index],
                codebook=self.tensor(f"codebook.{block['format'][:2]}", device),
                columns=columns,
                row_map=row_map[block["offset"] : block["offset"] + rowscales[index].numel()],
            )
            for index, block in enumerate(meta["blocks"])
        ]
        return Layer(blocks, self.tensor(f"rotation.signs_{columns}", device),
                     self.tensor(f"rotation.q_{columns}", device), out_features)


@cache
def open_sidecar(path: str) -> Sidecar:
    """One open sidecar per process; the config itself only carries the path, since vLLM pickles it."""
    return Sidecar(Path(path))


# The checkpoint says quant_method "mirai_s"; the plugin runs it as "mirai_s_2". vLLM keys its compile cache on this name
# and not on plugin code, and the graphs 0.1 cached hold the MTP drafter's bf16 weights, which are int8 since 0.2.
@register_quantization_config("mirai_s_2")
class MiraiSConfig(QuantizationConfig):
    def __init__(self) -> None:
        super().__init__()
        self.sidecar_path: str | None = None

    def __repr__(self) -> str:
        return "MiraiSConfig()"

    @classmethod
    def get_name(cls) -> str:
        return "mirai_s_2"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg: dict, user_quant: str | None, hf_config=None) -> str | None:
        return "mirai_s_2" if hf_quant_cfg["quant_method"] == "mirai_s" else None

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "MiraiSConfig":
        return cls()

    def maybe_update_config(self, model_name: str, hf_config=None, revision: str | None = None) -> None:
        self.sidecar_path = str(resolve_sidecar(model_name, revision))

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, ParallelLMHead):  # the target's and the MTP drafter's: both read the one sidecar head
            return MiraiSHeadMethod(self)
        if not isinstance(layer, LinearBase):
            return None
        match = LAYER.match(prefix)
        if match is None and prefix.startswith("mtp."):
            return MiraiSDrafterMethod()
        if match is None:
            return UnquantizedLinearMethod()
        return MiraiSLinearMethod(self, f"layers.{match[1]}.{match[2]}")


class MiraiSLinearMethod(LinearMethodBase):
    def __init__(self, config: MiraiSConfig, key: str) -> None:
        self.config = config
        self.key = key

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs) -> None:
        assert input_size_per_partition == input_size, "Mirai S layers run with tensor parallel size 1"
        layer.mirai_out_features = sum(output_partition_sizes)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert self.config.sidecar_path is not None, "the sidecar is located in maybe_update_config"
        device = torch.device("cuda", torch.cuda.current_device())
        mirai = open_sidecar(self.config.sidecar_path).layer(self.key, device)
        assert mirai.out_features == layer.mirai_out_features, f"{self.key}: sidecar rows do not match the layer"
        layer.mirai_layer = register_layer(mirai)
        compile_all()

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        y = linear_op(x.reshape(-1, x.shape[-1]), layer.mirai_layer)
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], layer.mirai_out_features)


class MiraiSDrafterMethod(UnquantizedLinearMethod):
    """The MTP drafter's bf16 layers as int8 weights with a per-row scale: half the memory and half the bytes each draft
    step reads, and int8 tensor cores. Drafts are verified, so this can only move the acceptance rate."""

    def create_weights(self, layer, *args, **kwargs) -> None:
        """The bf16 weights are staged in host memory and only their int8 form reaches the GPU, so loading a drafter
        never needs room for its bf16 copy (1.9B parameters for DFlash 2: 3.8 GB, more than a 16 GB card has left)."""
        with torch.device("cpu"):
            super().create_weights(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data.to(torch.device("cuda", torch.cuda.current_device())).float()
        rows, columns = weight.shape
        assert rows % 8 == 0 and columns % 8 == 0, "torch._int_mm and w8_quantize take multiples of 8"
        scale = weight.abs().amax(dim=1).clamp_min(1e-12) / 127
        layer.drafter_layer = register_drafter_layer(torch.round(weight / scale[:, None]).to(torch.int8), scale)
        del layer.weight
        compile_all()

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        y = drafter_linear_op(x.reshape(-1, x.shape[-1]), layer.drafter_layer)
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], y.shape[-1])


class MiraiSHeadMethod(QuantizeMethodBase):
    def __init__(self, config: MiraiSConfig) -> None:
        self.config = config

    def create_weights(self, layer: torch.nn.Module, *args, **kwargs) -> None:
        pass

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        head = open_sidecar(self.config.sidecar_path).surface("head", device)
        layer.mirai_head = [head[name] for name in ("codes", "row_scales", "ladder", "ladder_indices", "signs")]
        compile_all()

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        assert bias is None
        return logits_op(x.reshape(-1, x.shape[-1]), *layer.mirai_head).reshape(*x.shape[:-1], -1)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("the head is never used as an input embedding (Qwen3.8 is untied)")


# Drafters loaded as their own checkpoint (DFlash 2) get the MTP drafter's int8 linears through this method name:
# `--speculative-config '{"method": "dflash", "model": ..., "quantization": "mirai_s_w8"}'`.
@register_quantization_config("mirai_s_w8")
class MiraiSW8Config(QuantizationConfig):
    def __repr__(self) -> str:
        return "MiraiSW8Config()"

    @classmethod
    def get_name(cls) -> str:
        return "mirai_s_w8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "MiraiSW8Config":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        return MiraiSDrafterMethod() if isinstance(layer, LinearBase) else None


class MiraiSDrafterW4Method(UnquantizedLinearMethod):
    """A separately loaded drafter's linears as 4-bit weights (round-to-nearest, groups of 128) run by vLLM's Marlin
    W4A16 kernel: a quarter of the bf16 bytes each draft step reads. Drafts are verified, so this can only move the
    acceptance rate."""

    def create_weights(self, layer, *args, **kwargs) -> None:
        with torch.device("cpu"):  # see MiraiSDrafterMethod.create_weights
            super().create_weights(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (marlin_make_empty,
                                                                                marlin_make_workspace_new)
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
        from vllm.scalar_type import scalar_types

        device = torch.device("cuda", torch.cuda.current_device())
        weight = layer.weight.data.to(device)
        layer.w4_out, layer.w4_in = weight.shape
        _, layer.w4_weight, layer.w4_scale = marlin_quantize(weight.t().contiguous(), scalar_types.uint4b8, 128)
        layer.w4_zp = marlin_make_empty(device)
        layer.w4_workspace = marlin_make_workspace_new(device)
        del layer.weight

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import apply_gptq_marlin_linear
        from vllm.scalar_type import scalar_types

        return apply_gptq_marlin_linear(x, layer.w4_weight, layer.w4_scale, layer.w4_zp, layer.w4_workspace,
                                        scalar_types.uint4b8, layer.w4_out, layer.w4_in, bias=bias)


@register_quantization_config("mirai_s_w4")
class MiraiSW4Config(MiraiSW8Config):
    def __repr__(self) -> str:
        return "MiraiSW4Config()"

    @classmethod
    def get_name(cls) -> str:
        return "mirai_s_w4"

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        return MiraiSDrafterW4Method() if isinstance(layer, LinearBase) else None
