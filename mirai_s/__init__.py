def register() -> None:
    """vLLM plugin entry point: the "mirai_s" quantization method and the models that load its compressed embedding."""
    from vllm import ModelRegistry

    import mirai_s.quant  # noqa: F401
    from mirai_s import kv_groups

    kv_groups.apply()

    ModelRegistry.register_model("MiraiSQwen3_5ForCausalLM", "mirai_s.model:MiraiSQwen3_5ForCausalLM")
    # The MTP drafter class is looked up by this name; ours behaves exactly like vLLM's unless the model is mirai_s.
    ModelRegistry.register_model("Qwen3_5MTP", "mirai_s.model:MiraiSQwen3_5MTP")
    ModelRegistry.register_model("DFlash2DraftModel", "mirai_s.model:MiraiSDFlash2")
