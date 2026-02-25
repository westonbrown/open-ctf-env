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

WORKER_PATH = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/skyrl_train/workers/fsdp/fsdp_worker.py"
)

def main():
    if not WORKER_PATH.exists():
        print(f"   Patch (bf16 policy init): SKIP - {WORKER_PATH} not found")
        return

    content = WORKER_PATH.read_text()
    lines = content.splitlines()

    if "bf16=self.cfg.trainer.bf16" in content:
        print("   Patch (bf16 policy init): already applied")
        return

    # Patch only the first policy-model occurrence to avoid changing ref-model behavior.
    for i, line in enumerate(lines):
        if "bf16=False" in line:
            lines[i] = line.replace("bf16=False", "bf16=self.cfg.trainer.bf16", 1)
            WORKER_PATH.write_text("\n".join(lines) + "\n")
            print(f"   Patch (bf16 policy init): APPLIED at line {i + 1}")
            return

    # Upstream may have already refactored this callsite.
    print("   Patch (bf16 policy init): no matching pattern found, skipping")


if __name__ == "__main__":
    main()
