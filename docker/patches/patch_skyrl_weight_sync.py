#!/usr/bin/env python3
"""Patch SkyRL worker.py to skip NCCL weight sync when using LoRA + remote engines.

Problem: init_weight_sync_state() creates an NCCL process group that requires both
the trainer (rank 0) and inference engine (rank 1) to join simultaneously. With remote
engines that don't have WorkerWrap, the inference side never joins, causing a deadlock.

Fix: When LoRA is configured and engines are remote, skip the NCCL init entirely.
LoRA uses file-based sync (save adapters → HTTP /load_lora), not NCCL broadcast.
"""
import pathlib
import re
import sys

WORKER_PATH = pathlib.Path("/usr/local/lib/python3.12/dist-packages/skyrl_train/workers/worker.py")

def main():
    if not WORKER_PATH.exists():
        print(f"ERROR: {WORKER_PATH} not found")
        sys.exit(1)

    content = WORKER_PATH.read_text()

    # Check if already patched
    if "Skip NCCL weight sync init" in content:
        print("   Patch (LoRA weight sync skip): already applied")
        return

    # Find the init_weight_sync_state method and add the skip logic
    # We insert right after "assert inference_engine_client is not None"
    old_pattern = "        assert inference_engine_client is not None\n\n        # Create init info on all ranks"

    new_code = """        assert inference_engine_client is not None

        # PATCH: Skip NCCL weight sync init when using LoRA + remote engines.
        # LoRA uses file-based sync (save adapters -> HTTP /load_lora), not NCCL broadcast.
        # Without this, create_sender blocks forever waiting for inference engine to join NCCL group.
        _lora_rank = getattr(self.cfg.trainer.policy.model.lora, "rank", 0)
        _run_locally = getattr(self.cfg.generator, "run_engines_locally", True)
        if _lora_rank > 0 and not _run_locally:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Skipping NCCL weight sync init (LoRA rank=%d, remote engines). Using file-based sync.",
                _lora_rank,
            )
            self._weight_transfer_sender = None
            return

        # Create init info on all ranks"""

    if old_pattern in content:
        content = content.replace(old_pattern, new_code, 1)
        WORKER_PATH.write_text(content)
        print("   Patch (LoRA weight sync skip): APPLIED")
    else:
        # Try a more flexible match
        pattern = re.compile(
            r"(        assert inference_engine_client is not None\s*\n)"
            r"(\s*# Create init info on all ranks)"
        )
        m = pattern.search(content)
        if m:
            insert_point = m.start(2)
            skip_block = """
        # PATCH: Skip NCCL weight sync init when using LoRA + remote engines.
        # LoRA uses file-based sync (save adapters -> HTTP /load_lora), not NCCL broadcast.
        # Without this, create_sender blocks forever waiting for inference engine to join NCCL group.
        _lora_rank = getattr(self.cfg.trainer.policy.model.lora, "rank", 0)
        _run_locally = getattr(self.cfg.generator, "run_engines_locally", True)
        if _lora_rank > 0 and not _run_locally:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Skipping NCCL weight sync init (LoRA rank=%d, remote engines). Using file-based sync.",
                _lora_rank,
            )
            self._weight_transfer_sender = None
            return

"""
            content = content[:insert_point] + skip_block + content[insert_point:]
            WORKER_PATH.write_text(content)
            print("   Patch (LoRA weight sync skip): APPLIED (flexible match)")
        else:
            print("   Patch (LoRA weight sync skip): FAILED - pattern not found")
            # Debug: show what we have
            idx = content.find("init_weight_sync_state")
            if idx >= 0:
                print("   Context around init_weight_sync_state:")
                print(repr(content[idx:idx+500]))
            sys.exit(1)


if __name__ == "__main__":
    main()
