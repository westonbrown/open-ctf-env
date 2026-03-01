# Training Data

This folder contains the curated datasets used by the training pipeline.

## Active Datasets

| File | Purpose |
|---|---|
| `sft_v6.jsonl` | Stage-1 supervised fine-tuning traces (high-quality solved trajectories). |
| `online_rl_quality.jsonl` | Stage-2 online RL seed tasks (challenge prompts + metadata + `ground_truth_flag`). |
| `online_rl_quality.jsonl.manifest.json` | Manifest for online RL metadata containing dataset fingerprints. |

## What `online_rl_quality.jsonl` Is For

`online_rl_quality.jsonl` is not an offline reward dataset.
It seeds online RL episodes with challenge context and ground-truth flags so the environment can score live rollouts.

In online RL:
- The model generates actions live.
- Tools run live against challenge targets.
- Rewards are computed online from rollout behavior + flag verification.

## Scalable Practice (Recommended)

Treat online RL data as generated artifacts from benchmark infra, not hand-edited files.

1. Generate from registry + metadata:

```bash
python src/open_ctf/cli/generate_online_rl.py \
  --registry configs/challenges/cybench.yaml \
  --metadata configs/challenges/cybench_metadata.json \
  --output data/online_rl_quality.jsonl
```

2. (RunPod/DGX) Apply target overrides and verify endpoints before writing data:

```bash
python src/open_ctf/cli/generate_online_rl.py \
  --registry configs/challenges/cybench.yaml \
  --target-map configs/challenges/cybench_target_map_runpod.json \
  --probe-targets \
  --strict-target-probe \
  --output data/online_rl_quality.jsonl
```

This generator now also writes `data/online_rl_quality.jsonl.manifest.json`
with source file hashes (registry/metadata/target map) and challenge coverage.

3. Audit registry/data/target consistency:

```bash
python scripts/deploy/online_rl_readiness_audit.py \
  --run-root /tmp/online_rl_audit \
  --registry configs/challenges/cybench.yaml \
  --online-rl-quality data/online_rl_quality.jsonl
```

4. Preflight gate before launch (fails on registry/data drift):

```bash
open-ctf-validate --mode grpo-preflight \
  --online-rl-data data/online_rl_quality.jsonl \
  --challenge-registry configs/challenges/cybench.yaml \
  --target-map configs/challenges/cybench_target_map_runpod.json \
  --require-manifest \
  --require-target-map-coverage
```

This keeps data naming, target mapping, and challenge reachability reproducible across new instances.
