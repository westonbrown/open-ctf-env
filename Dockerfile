# Open CTF Environment - Unified Training Container
# Base: scitrera/dgx-spark-vllm:0.14.1-t5 (GB10/CUDA 13.1 compatibility)
#
# Fully leverages Unsloth on DGX Spark (Blackwell GB10, sm_121):
#   - grouped_mm MoE backend (bypasses Triton shared memory limit)
#   - Inductor auto-prune for shared memory (catches FlexAttention backward OOM)
#   - transformers>=5.0 for GLM-4.7-Flash (glm4_moe_lite architecture)
#   - BNB_CUDA_VERSION=130 (no CUDA 13.1 binary yet, 13.0 is forward-compatible)
#
# Build:
#   docker build -t open-ctf-env:latest .
#
# Run SFT:
#   docker run --gpus all --name sft \
#       -v ./data:/workspace/data -v ./outputs:/workspace/outputs \
#       open-ctf-env:latest open-ctf-train sft \
#         --model unsloth/GLM-4.7-Flash \
#         --data /workspace/data/sft_general_tactics.jsonl \
#         --output /workspace/outputs/sft \
#         --config /workspace/configs/training_dgx.yaml
#
# Run GRPO:
#   docker run --gpus all --name grpo \
#       -e OPEN_CTF_NO_UNSLOTH=1 \
#       -v ./data:/workspace/data -v ./outputs:/workspace/outputs \
#       open-ctf-env:latest open-ctf-train grpo \
#         --model /workspace/outputs/sft/final \
#         --data /workspace/data/grpo_general_failures.jsonl \
#         --output /workspace/outputs/grpo \
#         --config /workspace/configs/training_dgx.yaml

FROM scitrera/dgx-spark-vllm:0.14.1-t5

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
# Pin TRL 0.24.x (validated with Unsloth), datasets 4.3.0 (Unsloth compat).
# bitsandbytes for optimizer (8-bit AdamW); 4-bit QLoRA NOT used for MoE.
RUN pip install --no-cache-dir \
    "trl>=0.22.0,<=0.24.0" \
    "peft>=0.14.0" \
    "accelerate>=1.2.0" \
    "bitsandbytes>=0.45.0" \
    "datasets==4.3.0" \
    "pyyaml" "rich" "jsonlines" "requests" "jmespath" \
    "uvicorn[standard]" "wandb" "colorama" \
    "ray[default]" "gymnasium"

# --------------------------------------------------------------------------
# 3. Unsloth (latest from GitHub for GLM-4.7-Flash MoE support)
# --------------------------------------------------------------------------
RUN pip install --upgrade --force-reinstall --no-cache-dir --no-deps \
    "unsloth @ git+https://github.com/unslothai/unsloth.git" \
    unsloth_zoo

# --------------------------------------------------------------------------
# 4. transformers >= 5.0 (required for glm4_moe_lite architecture)
# --------------------------------------------------------------------------
# The base image ships transformers 4.57.x which does not recognize
# GLM-4.7-Flash's glm4_moe_lite model type. Upgrade explicitly.
# This must run AFTER Unsloth install (which uses --no-deps).
RUN pip install --no-cache-dir "transformers>=5.0.0"

# --------------------------------------------------------------------------
# 5. TRL compatibility patch
# --------------------------------------------------------------------------
# GLM-4.7-Flash chat template is not prefix-preserving (<think>/<|observation|>
# tags change token prefix layout). Convert TRL's hard error to a soft pass.
RUN python3 << 'PATCH'
import glob
paths = glob.glob("/usr/local/lib/python*/dist-packages/trl/trainer/grpo_trainer.py")
if paths:
    p = paths[0]
    s = open(p).read()
    old = '''raise ValueError(
                        "The chat template is not prefix-preserving. Please update it to use a prefix-preserving "
                        "format."
                    )'''
    if old in s:
        s = s.replace(old, 'pass  # patched: prefix check disabled for GLM-4.7-Flash')
        open(p, "w").write(s)
        print(f"Patched {p}: prefix-preserving -> pass")
    else:
        print("Pattern not found (TRL version may differ or already patched).")
else:
    print("WARNING: Could not locate TRL GRPO Trainer.")
PATCH

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
