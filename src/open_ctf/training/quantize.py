"""Post-load 4-bit quantization for MoE expert parameters.

BitsAndBytes only quantizes ``nn.Linear`` (2D weight matrices).
Transformers v5 MoE models store expert weights as fused 3D
``nn.Parameter`` tensors (shape: ``[num_experts, dim1, dim2]``) which
BnB silently leaves in BF16.

For MoE models (e.g. GLM-4.7-Flash) this means BnB ``load_in_4bit``
only saves ~3 GB because 90% of parameters are routed experts.

This module provides post-load quantization via
``bitsandbytes.nn.parametrize.replace_parameter_4bit`` (requires
bitsandbytes >= 0.48.0) to quantize those expert tensors too.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def find_moe_expert_param_names(model) -> list:
    """Detect 3D+ nn.Parameter tensors that BnB skips during quantization.

    Returns deduplicated parameter path suffixes, e.g.::

        ["mlp.experts.down_proj", "mlp.experts.gate_up_proj"]
    """
    seen_suffixes = set()
    for name, param in model.named_parameters():
        if param.ndim >= 3 and any(
            kw in name for kw in ("experts", "gate_up_proj", "down_proj")
        ):
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part.isdigit():
                    suffix = ".".join(parts[i + 1:])
                    seen_suffixes.add(suffix)
                    break
    result = sorted(seen_suffixes)
    if result:
        logger.info("Detected MoE expert parameters: %s", result)
    return result


def quantize_moe_expert_params(model, quant_type=None,
                                compress_statistics=None):
    """Quantize 3D MoE expert nn.Parameter tensors that BnB skipped.

    Uses ``bitsandbytes.nn.parametrize.replace_parameter_4bit`` to apply
    NF4 quantization to fused expert weights post-load.

    For large MoE models (e.g. 46 MoE layers x 64 experts):
      BF16 experts: ~52 GB  ->  4-bit experts: ~6.5 GB  =  ~45 GB saved

    Returns True if any parameters were quantized.
    """
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn.parametrize import replace_parameter_4bit
    except (ImportError, AttributeError):
        logger.warning(
            "bitsandbytes.nn.parametrize not available (need bnb >= 0.48.0). "
            "MoE expert tensors will remain in BF16."
        )
        return False

    # Find unquantized 3D+ expert params (skip already-quantized Linear4bit)
    params_to_quantize = []
    for _, module in model.named_modules():
        if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
            continue
        for param_name, param in module.named_parameters(recurse=False):
            if param.ndim >= 3 and any(
                kw in param_name
                for kw in ("experts", "gate_up_proj", "down_proj")
            ):
                params_to_quantize.append((module, param_name))

    if not params_to_quantize:
        return False

    # Derive settings from model's BnB config for consistency
    if quant_type is None or compress_statistics is None:
        bnb_config = getattr(model.config, "quantization_config", None)
        if bnb_config is not None:
            if quant_type is None:
                quant_type = getattr(bnb_config, "bnb_4bit_quant_type", "nf4")
            if compress_statistics is None:
                compress_statistics = getattr(
                    bnb_config, "bnb_4bit_use_double_quant", True
                )
    quant_type = quant_type or "nf4"
    if compress_statistics is None:
        compress_statistics = True

    # Log memory before quantization
    mem_before = None
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / 1e9
        logger.info("GPU memory before MoE expert quantization: %.2f GB", mem_before)

    count = 0
    for module, param_name in params_to_quantize:
        param = getattr(module, param_name)
        if count == 0:
            logger.info(
                "First expert param: %s, shape=%s, dtype=%s",
                param_name, list(param.shape), param.dtype,
            )
        replace_parameter_4bit(
            module,
            param_name,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
        )
        count += 1

    torch.cuda.empty_cache()

    if torch.cuda.is_available() and mem_before is not None:
        mem_after = torch.cuda.memory_allocated() / 1e9
        logger.info(
            "GPU memory after MoE expert quantization: %.2f GB (saved %.2f GB)",
            mem_after, mem_before - mem_after,
        )

    logger.info(
        "Quantized %d MoE expert parameters to 4-bit "
        "(quant_type=%s, double_quant=%s)",
        count, quant_type, compress_statistics,
    )
    return True
