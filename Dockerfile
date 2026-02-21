# Open CTF Environment - Online RL Training Container
# Base: vllm-node-tf5 (built from eugr/spark-vllm-docker with --pre-tf)
#
# Prerequisites — build the vLLM base image first:
#   git clone https://github.com/eugr/spark-vllm-docker.git
#   cd spark-vllm-docker && ./build-and-copy.sh --pre-tf -t vllm-node-tf5
#
# This base provides:
#   - vLLM (latest main) compiled for Blackwell GB10 (sm_121a)
#   - FlashInfer (prebuilt wheels or compiled from source)
#   - transformers>=5.0 (glm4_moe_lite architecture support)
#   - --tool-call-parser glm47 for native GLM-4.7-Flash tool calling
#   - --reasoning-parser glm45 for thinking token preservation
#   - NGC PyTorch 26.01 (CUDA 13.1, Python 3.12)
#
# Two training modes:
#   SFT:  Unsloth fast kernels + MoE grouped_mm backend
#   GRPO: OPEN_CTF_NO_UNSLOTH=1 + TRL 0.28 tools= + vLLM colocate
#
# Build:
#   docker build -t open-ctf-env:latest .
#
# Run SFT:
#   docker run --gpus all --name sft \
#       -v ./data:/workspace/data -v ./outputs:/workspace/outputs \
#       open-ctf-env:latest open-ctf-train sft \
#         --model unsloth/GLM-4.7-Flash \
#         --data /workspace/data/sft.jsonl \
#         --output /workspace/outputs/sft \
#         --config /workspace/configs/training_dgx.yaml
#
# Run GRPO (online RL with live tool execution):
#   docker run --gpus all --name grpo \
#       -e OPEN_CTF_NO_UNSLOTH=1 \
#       -e OPEN_CTF_ENV_URL=http://localhost:8100 \
#       -v ./data:/workspace/data -v ./outputs:/workspace/outputs \
#       open-ctf-env:latest open-ctf-train grpo \
#         --model /workspace/outputs/sft-merged \
#         --data /workspace/data/grpo.jsonl \
#         --output /workspace/outputs/grpo \
#         --config /workspace/configs/training_dgx.yaml

FROM vllm-node-tf5

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace/open-ctf-env

# --------------------------------------------------------------------------
# 1. System packages
# --------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# 2. Python ML stack
# --------------------------------------------------------------------------
# TRL 0.28+ for:
#   - tools= parameter (online RL with live tool execution, added 0.26)
#   - max_tool_calling_iterations (added 0.27)
#   - async tool calls + parquet reward logging (added 0.28)
#   - use_vllm colocate mode for fast generation
#
# Install TRL WITHOUT [vllm] extra: TRL 0.28 pins vllm<0.13, but our base
# image ships vLLM 0.14+. The pre-installed vLLM works fine — TRL's vLLM
# integration is API-level (colocate or server mode), not version-locked.
#
# jmespath: required by TRL for tools= JSON schema generation.
# accelerate>=1.4.0: required by TRL 0.28 (bumped from 1.2.0).
# datasets>=3.0.0: required by TRL 0.28 (bumped from 2.x).
# bitsandbytes>=0.48.0: 8-bit AdamW + replace_parameter_4bit for MoE expert
#   3D tensors (post-load quantization of nn.Parameter that BnB skips).
RUN pip install --no-cache-dir \
    "trl>=0.28.0,<0.29.0" \
    "peft>=0.14.0" \
    "accelerate>=1.4.0" \
    "bitsandbytes>=0.48.0" \
    "datasets>=3.0.0" \
    "jmespath" \
    "pyyaml" "rich" "jsonlines" "requests" \
    "uvicorn[standard]" "wandb" "colorama" \
    "ray[default]" "gymnasium"

# --------------------------------------------------------------------------
# 3. Unsloth (for SFT pipeline only)
# --------------------------------------------------------------------------
# Unsloth is used exclusively for SFT training (fast kernels, MoE support).
# GRPO uses OPEN_CTF_NO_UNSLOTH=1 to bypass Unsloth's GRPOTrainer patches
# which are incompatible with GLM-4.7-Flash's shared_head architecture
# (Bug 2: shape mismatch in efficient_log_softmax).
RUN pip install --upgrade --force-reinstall --no-cache-dir --no-deps \
    "unsloth @ git+https://github.com/unslothai/unsloth.git" \
    unsloth_zoo

# --------------------------------------------------------------------------
# 4. transformers 5.0.x (required for glm4_moe_lite architecture)
# --------------------------------------------------------------------------
# Skip 5.1.0 (DeepSpeed incompatibility). 5.0.x and 5.2.x are fine.
# The base image may ship 5.2+ from --pre-tf build; Unsloth's --no-deps
# install may have shifted versions. Re-pin explicitly.
RUN pip install --no-cache-dir "transformers>=5.0.0,!=5.1.0"

# --------------------------------------------------------------------------
# 5. TRL compatibility patches (SFT pipeline — Unsloth token injection)
# --------------------------------------------------------------------------
# NOTE: Patch 5a (prefix-preserving chat template check) is REMOVED from the
# Dockerfile. It is now handled by an in-memory monkey-patch in grpo.py
# (_patch_trl_prefix_check) which is more version-resilient. The in-memory
# patch catches the ValueError from get_training_chat_template() regardless
# of the exact error message text across TRL versions.

# 5b. eos_token fallback for Unsloth compatibility.
# Unsloth's _backwards_compatible_trainer re-creates SFTConfig and injects
# <EOS_TOKEN> as default, which doesn't exist in GLM-4.7-Flash's vocabulary.
# Patch SFTTrainer.__init__ to fall back to tokenizer.eos_token gracefully.
# If the pattern doesn't match (TRL 0.28 may have changed it), the in-memory
# fallback in sft.py will handle it at runtime.
RUN python3 << 'PATCH_EOS'
import glob
paths = glob.glob("/usr/local/lib/python*/dist-packages/trl/trainer/sft_trainer.py")
if paths:
    p = paths[0]
    s = open(p).read()
    old = '''        if args.eos_token is not None:
            eos_token = args.eos_token
            eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
            if eos_token_id is None:
                raise ValueError(
                    f"The specified `eos_token` ('{eos_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `eos_token` exists "
                    "in the vocabulary before using it as an EOS token."
                )
            tokenizer.eos_token_id = eos_token_id'''
    new = '''        if args.eos_token is not None:
            eos_token = args.eos_token
            eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
            if eos_token_id is None:
                # Patched: fall back to tokenizer's own eos_token instead of crashing.
                # Unsloth's _backwards_compatible_trainer can inject <EOS_TOKEN> default
                # that doesn't exist in all model vocabularies (e.g. GLM-4.7-Flash).
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "eos_token '%s' not in vocab, falling back to tokenizer.eos_token='%s'",
                    eos_token, tokenizer.eos_token,
                )
                eos_token = tokenizer.eos_token
                eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
            tokenizer.eos_token_id = eos_token_id'''
    if old in s:
        s = s.replace(old, new)
        open(p, "w").write(s)
        print(f"Patched {p}: eos_token fallback for Unsloth compatibility")
    else:
        print("Pattern not found in sft_trainer.py (TRL 0.28 may use different text). "
              "In-memory fallback in sft.py will handle this at runtime.")
else:
    print("WARNING: Could not locate TRL SFT Trainer.")
PATCH_EOS

# 5c. pad_token fallback for Unsloth compatibility.
# Same issue: Unsloth injects <PAD_TOKEN> which doesn't exist in
# GLM-4.7-Flash (uses [MASK] as pad_token).
RUN python3 << 'PATCH_PAD'
import glob
paths = glob.glob("/usr/local/lib/python*/dist-packages/trl/trainer/sft_trainer.py")
if paths:
    p = paths[0]
    s = open(p).read()
    old = '''            pad_token = args.pad_token or tokenizer.pad_token or tokenizer.eos_token
            pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
            if pad_token_id is None:
                raise ValueError(
                    f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` exists "
                    "in the vocabulary before using it as a padding token."
                )'''
    new = '''            pad_token = args.pad_token or tokenizer.pad_token or tokenizer.eos_token
            pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
            if pad_token_id is None:
                # Patched: fall back to tokenizer's own pad_token instead of crashing.
                # Unsloth's _backwards_compatible_trainer can inject <PAD_TOKEN> default
                # that doesn't exist in all model vocabularies (e.g. GLM-4.7-Flash).
                import logging as _logging
                _fallback = tokenizer.pad_token or tokenizer.eos_token
                _logging.getLogger(__name__).warning(
                    "pad_token '%s' not in vocab, falling back to '%s'",
                    pad_token, _fallback,
                )
                pad_token = _fallback
                pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)'''
    if old in s:
        s = s.replace(old, new)
        open(p, "w").write(s)
        print(f"Patched {p}: pad_token fallback for Unsloth compatibility")
    else:
        print("Pattern not found in sft_trainer.py (TRL 0.28 may use different text). "
              "In-memory fallback in sft.py will handle this at runtime.")
else:
    print("WARNING: Could not locate TRL SFT Trainer.")
PATCH_PAD

# --------------------------------------------------------------------------
# 6. Project source
# --------------------------------------------------------------------------
COPY . /workspace/open-ctf-env/
RUN pip install --no-cache-dir -e ".[dev,train]"

# --------------------------------------------------------------------------
# 7. Environment variables — GB10 Blackwell (sm_121) optimizations
# --------------------------------------------------------------------------
ENV PYTHONPATH=/workspace/open-ctf-env/src:$PYTHONPATH \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    # --- bitsandbytes: no CUDA 13.1 binary yet, 13.0 is forward-compatible ---
    BNB_CUDA_VERSION=130 \
    # --- Unsloth MoE: use torch._grouped_mm instead of Triton MoE kernels ---
    # GB10 has 99KB shared mem per SM; Triton MoE kernels need 104-147KB.
    UNSLOTH_MOE_BACKEND=grouped_mm \
    # --- PyTorch inductor: prune kernel configs exceeding shared memory ------
    # Catches FlexAttention backward OOM (136KB required > 101KB limit).
    TORCHINDUCTOR_MAX_AUTOTUNE_PRUNE_CHOICES_BASED_ON_SHARED_MEM=1 \
    # --- Unsloth compiler: disable torch.compile on GB10 --------------------
    # FlexAttention backward kernels exceed GB10 shared memory limits.
    # Disabling Unsloth's compiler avoids the fused_flex_attention_backward
    # crash while keeping all other Unsloth optimizations (MoE kernels,
    # gradient checkpointing, sequence packing, logit chunking).
    UNSLOTH_COMPILE_DISABLE=1

CMD ["open-ctf-train", "--help"]
