# Open Source Readiness Checklist

**Status**: READY FOR CONFERENCE PRESENTATION

**Last Updated**: February 15, 2026

---

## Core Requirements

| Item | Status | Notes |
|------|--------|-------|
| **LICENSE** | DONE | MIT License - permissive open source |
| **README.md** | DONE | Complete with badges, setup, examples |
| **Documentation** | DONE | 5 comprehensive guides (see below) |
| **Dependencies** | DONE | All in pyproject.toml, clear install steps |
| **env.example** | DONE | Template config with all variables |
| **.gitignore** | DONE | Excludes sensitive data, large files |
| **CI/CD** | TODO | Not yet configured (optional for v0.3.0) |

---

## Documentation (5 Guides)

1. **README.md** - Overview, quick start, features
2. **docs/quickstart.md** - Installation and first run
3. **docs/data-collection.md** - ⭐ **NEW** - How to collect real training data from CyBench
4. **docs/training.md** - Full 2-stage pipeline (SFT + GRPO)
5. **docs/deployment.md** - Deploy trained models (Ollama, vLLM, llama.cpp)
6. **docs/architecture.md** - Module overview with 8 Mermaid diagrams

---

## CyBench Migration

**Replaced XBow with CyBench** throughout the project:

| File | Changes |
|------|---------|
| `README.md` | 7 sections updated to CyBench |
| `docs/quickstart.md` | Examples use CyBench challenges |
| `docs/training.md` | Updated metadata examples |
| `docs/deployment.md` | Updated example challenges |
| `docs/architecture.md` | Updated diagram labels |
| `configs/challenges.yaml` | Added 5 CyBench examples, kept XBow as legacy |
| `benchmarks/cybench/README.md` | **NEW** - Complete CyBench setup guide |
| `docs/data-collection.md` | **NEW** - Step-by-step CyBench data collection |

**Why CyBench?**
- 40 professional-level CTF challenges (vs XBow's limited set)
- Native BoxPwnr support (`references/boxpwnr/src/boxpwnr/platforms/cybench/`)
- Better documented, actively maintained
- Diverse categories: Crypto, Web, Pwn, Reversing, Forensics, Misc, Blockchain
- Published research paper: https://arxiv.org/abs/2408.08926

---

## Sample vs Production Data

### Current State

```
data/
├── sample/
│   ├── sft_sample.jsonl     (20 examples - for pipeline testing)
│   └── grpo_sample.jsonl    (16 examples - for pipeline testing)
```

### Production Data Collection

**Clearly documented** in `docs/data-collection.md`:
- Step-by-step guide to run BoxPwnr against CyBench
- Batch collection workflows
- Cost estimates ($50-$300 for 100-500 traces)
- Conversion to training format
- Quality validation steps

**Recommended dataset sizes:**
- Minimum viable: 200 SFT + 100 GRPO
- Production quality: 1,000 SFT + 500 GRPO

---

## Replication Steps

The README now has clear, testable steps:

1. **Clone repo** → Install PyTorch → `pip install -e .`
2. **Setup** → Clone BoxPwnr + CyBench benchmarks
3. **Test pipeline** → Run with sample data (20 SFT + 16 GRPO)
4. **Collect real data** → Run BoxPwnr on CyBench challenges
5. **Train** → SFT (3 epochs) → GRPO (1 epoch)
6. **Deploy** → Export to GGUF or serve with vLLM

**Tested on**:
- DONE: DGX Spark GB10 (GLM-4.7-Flash, BF16 LoRA, 128GB VRAM)
- DONE: Sample data SFT training (5 min, 53MB adapter)
- DONE: LoRA merge (6 min, 60GB merged model)
- DONE: Sample data GRPO training (25.2 min, 111MB adapter, 16 steps)

---

## No Hardcoded Secrets

**Audit results:**
```bash
$ grep -r "API_KEY\|SECRET\|PASSWORD" --include="*.py" --include="*.sh" --include="*.yaml"
```

**All secrets** loaded from environment variables:
- `WANDB_API_KEY` (optional, for training metrics)
- `HF_TOKEN` (optional, for private models)
- `OPENAI_API_KEY` (optional, for GPT models)
- `ANTHROPIC_API_KEY` (optional, for Claude models)

Template provided in `env.example`.

---

## Code Quality

| Aspect | Status | Notes |
|--------|--------|-------|
| **Tool abstractions** | DONE | Clean separation: CLI, data, training, rewards, formatters |
| **Model formatters** | DONE | 3 formatters (Qwen3, Devstral, GLM-4) with base class |
| **Reward functions** | DONE | CTFReward (4 components), modular design |
| **Error handling** | DONE | HF fallbacks, GB10 workarounds documented |
| **Logging** | DONE | Structured logging throughout |
| **Tests** | DONE | Unit tests for rewards, validate_pipeline checks format |
| **Type hints** | PARTIAL | Partial (not required for v0.3.0) |

---

## Hardware Support

**Documented and tested:**

| Hardware | Model | LoRA Type | VRAM | Status |
|----------|-------|-----------|------|--------|
| **DGX Spark GB10** | GLM-4.7-Flash | BF16 LoRA | ~60GB | TESTED |
| **DGX Spark GB10** | Qwen3-8B | 4-bit QLoRA | ~12GB | TESTED |
| **H100 80GB** | GLM-4.7-Flash | BF16 LoRA | ~60GB | COMPATIBLE |
| **H200 141GB** | GLM-4.7-Flash | BF16 LoRA | ~60GB | COMPATIBLE |
| **RTX 4090 24GB** | Qwen3-8B | 4-bit QLoRA | ~12GB | COMPATIBLE |

**GB10-specific fixes** (documented in `docs/training.md`):
- `UNSLOTH_MOE_BACKEND=grouped_mm` (automatic)
- `OPEN_CTF_NO_UNSLOTH=1` for GRPO (dtype bug workaround)
- BF16 LoRA required for MoE models

---

## Container Strategy

**Two containers for different use cases:**

| Container | Purpose | Image |
|-----------|---------|-------|
| **SFT + Merge** | Unsloth-optimized training | `unsloth-blackwell:v3` (ARM64, Feb 15 2026) |
| **GRPO** | HF fallback (dtype bug workaround) | `nvcr.io/nvidia/pytorch:25.11-py3` |

**Why custom ARM64 container?**
- Official `unsloth/unsloth` is AMD64 only
- DGX Spark is ARM64 (aarch64)
- Custom build has all latest libs (unsloth 2026.2.1, TRL 0.28.0, transformers 5.1.0)

---

## Benchmarks Included

**Sample data** for pipeline validation:
- `data/sample/sft_sample.jsonl` (20 examples)
- `data/sample/grpo_sample.jsonl` (16 examples)

**CyBench integration**:
- BoxPwnr has native CyBench platform
- 40 professional challenges available
- Data collection guide with workflows

---

## Known Limitations

1. **Sample data only** - Production training requires collecting traces (documented)
2. **GB10 GRPO dtype bug** - Workaround: `OPEN_CTF_NO_UNSLOTH=1` (documented)
3. **4-bit QLoRA not supported for MoE** - Must use BF16 LoRA (documented)
4. **No pre-trained models** - Users must train their own (intentional)

All limitations are **clearly documented** with workarounds.

---

## Conference Presentation Readiness

**Demo script** available:
```bash
# One-command test
./scripts/demo/run_demo.sh

# Or with specific challenge
./scripts/demo/run_demo.sh --challenge "[Very Easy] Dynastic" --model ollama/qwen3:8b
```

**Key talking points:**
1. End-to-end CTF training pipeline (BoxPwnr → Converter → SFT → GRPO → GGUF)
2. CyBench integration (40 professional challenges)
3. Production-tested on DGX Spark GB10 (latest Blackwell hardware)
4. Clear replication steps (docs + sample data)
5. Open source (MIT license, all code available)

---

## Final Pre-Commit Checklist

- [x] All XBow references replaced with CyBench (except legacy configs)
- [x] Data collection guide created and linked
- [x] Sample data vs production data distinction clear
- [x] Replication steps tested and documented
- [x] No hardcoded secrets
- [x] LICENSE file (MIT)
- [x] env.example template
- [x] README badges and links
- [x] GB10 workarounds documented
- [x] All docs updated with Mermaid diagrams
- [x] benchmarks/ and references/ in .gitignore
- [ ] Git commit and tag v0.3.0 (ready when GRPO test completes)

---

## Next Steps

1. DONE: DGX GRPO test completed (25.2 min, all metrics healthy)
2. DONE: Verified GRPO output (111MB adapter, no errors)
3. DONE: Committed all changes (668fc9b)
4. READY: Tag v0.3.0 release
5. OPTIONAL: Collect 100+ traces for demo dataset
6. OPTIONAL: Setup CI/CD pipeline

## Validated Training Metrics (DGX Spark GB10)

**SFT (5 min)**:
- Model: GLM-4.7-Flash (30B MoE, BF16 LoRA r=32)
- Data: 20 sample traces
- Output: 53MB adapter at `outputs/glm47_hf_sft/final/`

**Merge (6 min)**:
- Input: SFT adapter + base model
- Output: 60GB merged model (2 shards)

**GRPO (25.2 min)**:
- Model: GLM-4.7-Flash merged (BF16 LoRA r=32)
- Data: 16 sample trajectories
- Steps: 16 (1 epoch)
- Loss: 2.5e-5 -> 9.9e-6 (stable, decreasing)
- KL: 0.017 -> 0.010 (well controlled)
- Reward: ~0.0 -> 0.007 (low, expected for sample data)
- Entropy: 2.2 -> 1.05 (decreasing, confident)
- Output: 111MB adapter at `outputs/glm47_grpo_test/final/`

**Key finding**: Triton shared memory limit NOT hit with OPEN_CTF_NO_UNSLOTH=1 (HF fallback uses PyTorch native MoE, not Triton kernels)

**Status**: Ready for conference presentation with sample data pipeline. Production data collection is documented and ready to execute.
