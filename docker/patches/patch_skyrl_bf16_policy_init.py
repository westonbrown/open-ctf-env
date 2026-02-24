#!/usr/bin/env python3
"""Patch SkyRL fsdp_worker.py to use configurable bf16 for policy model init.

Problem: FSDPPolicyWorkerBase.init_model() hardcodes bf16=False for the policy model,
forcing fp32 initialization (~12GB for a 3B model vs ~6GB in bf16). Combined with
FSDP state dict copies and vLLM allocation, this causes OOM on memory-constrained
systems like DGX Spark GB10.

Fix: Change `bf16=False` to `bf16=self.cfg.trainer.bf16` for the policy model init.
The ref model (line 363) already uses configurable bf16.
"""
import pathlib
import sys

WORKER_PATH = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/skyrl_train/workers/fsdp/fsdp_worker.py"
)

def main():
    if not WORKER_PATH.exists():
        print(f"ERROR: {WORKER_PATH} not found")
        sys.exit(1)

    content = WORKER_PATH.read_text()
    lines = content.split("\n")

    # Find the FIRST occurrence of bf16=False (policy model, around line 130)
    patched = False
    for i, line in enumerate(lines):
        if "bf16=False" in line and not patched:
            # Only patch the first occurrence (policy model)
            lines[i] = line.replace("bf16=False", "bf16=self.cfg.trainer.bf16")
            patched = True
            print(f"   Patch (bf16 policy init): APPLIED at line {i+1}")
            break

    if patched:
        WORKER_PATH.write_text("\n".join(lines))
    else:
        # Check if already patched
        if lines[129] and "bf16=self.cfg.trainer.bf16" in lines[129]:
            print("   Patch (bf16 policy init): already applied")
        else:
            print("   Patch (bf16 policy init): no matching pattern found")
            sys.exit(1)


if __name__ == "__main__":
    main()
