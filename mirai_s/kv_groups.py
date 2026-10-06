"""KV cache grouping for a hybrid target plus a separately loaded drafter.

vLLM splits each layer type into groups of one size and pads the last group of every type. It takes the size from
the smallest type, so Qwen3.8's 48 recurrent and 16 full-attention layers next to a 5-layer DFlash drafter get groups
of 5: 16 full-attention layers padded to 20, a quarter of the full-attention KV cache allocated and never used. This
picks the group size with the least padding instead (8 here: only the drafter's sliding-window group is padded, and a
sliding window holds few blocks). The rest of vLLM's function is unchanged; the patch is applied by rewriting its
source, and fails loudly if vLLM's code no longer matches.
"""

import inspect
import textwrap

from vllm.v1.core import kv_cache_utils

ORIGINAL = """    min_num_layers = min([len(layers) for layers in layer_buckets])
    group_size = min_num_layers
"""
REPLACEMENT = """    min_num_layers = min([len(layers) for layers in layer_buckets])
    group_size = _least_padding_group_size([len(layers) for layers in layer_buckets])
"""


def least_padding_group_size(counts: list[int]) -> int:
    """Group size from half the smallest type upward with the fewest padding layers, larger on ties (fewer groups).
    vLLM's own rule (smallest type, or the largest when all types are close) still applies when nothing pads less."""
    smallest, largest = min(counts), max(counts)
    default = largest if largest < smallest * 1.5 else smallest
    def padding(size: int) -> int:
        return sum(-count % size for count in counts)
    best = min(range(max(1, (smallest + 1) // 2), largest + 1), key=lambda size: (padding(size), -size))
    return best if padding(best) < padding(default) else default


def apply() -> None:
    function = kv_cache_utils._get_kv_cache_groups_uniform_page_size
    if getattr(function, "_mirai_patched", False):
        return
    source = textwrap.dedent(inspect.getsource(function))
    assert ORIGINAL in source, "vLLM's KV cache grouping changed; update mirai_s/kv_groups.py"
    namespace = dict(vars(kv_cache_utils), _least_padding_group_size=least_padding_group_size)
    exec(compile(source.replace(ORIGINAL, REPLACEMENT), inspect.getsourcefile(function), "exec"), namespace)
    patched = namespace[function.__name__]
    patched._mirai_patched = True
    kv_cache_utils._get_kv_cache_groups_uniform_page_size = patched
