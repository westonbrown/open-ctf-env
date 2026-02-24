# Contributing to Open CTF Environment

Open CTF Environment is a pipeline for post-training security LLMs on CTF challenge trajectories. It combines LlamaFactory for supervised fine-tuning, SkyRL for online reinforcement learning with live tool execution, and GEPA for prompt evolution -- producing locally deployable security agents from open-weight models.

## Development Setup

```bash
git clone https://github.com/westonbrown/open-ctf-env.git
cd open-ctf-env
pip install -e ".[dev]"
```

For training stages, install the relevant extras:

```bash
pip install -e ".[sft]"    # Stage 1: LlamaFactory SFT
pip install -e ".[grpo]"   # Stage 2: SkyRL GRPO
pip install -e ".[gepa]"   # Stage 3: GEPA prompt evolution
```

## Running Tests

```bash
pytest tests/
```

All tests should pass without a GPU. Tests that require a GPU or running OpenEnv server are skipped automatically.

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check .
ruff format .
```

## Architecture Overview

The project has three training stages, each backed by a dedicated framework:

| Stage | Framework | Purpose |
|-------|-----------|---------|
| **SFT** | [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) | Supervised fine-tuning on expert CTF traces |
| **GRPO** | [SkyRL](https://github.com/NovaSky-AI/SkyRL) | Online reinforcement learning with live tool execution |
| **GEPA** | [DSPy](https://github.com/stanfordnlp/dspy) + [GEPA](https://arxiv.org/abs/2507.19457) | Prompt evolution (no weight updates) |

The environment server ([OpenEnv](https://github.com/OpenEnvs/OpenEnv)) provides 13 tools (shell, Python, file ops, flag submission) over HTTP. During online GRPO, the model generates tool calls that execute against live Docker containers.

## Adding a New Model

1. Create a LlamaFactory SFT config at `configs/llamafactory/<model>.yaml`.
2. Create a SkyRL GRPO config at `configs/skyrl/<model>.yaml`.
3. If the model uses a non-standard chat template, add a formatter in `src/open_ctf/formatters/`.
4. Test with the validation pipeline: `open-ctf-validate`.

See existing configs (`nanbeige_3b.yaml`, `glm47_flash.yaml`, `devstral_24b.yaml`) for reference.

## Adding a New Benchmark

1. Write an `exec_fn` that maps tool calls to your target environment (Docker exec, SSH, HTTP).
2. Register it with the OpenEnv server instance.
3. Create GRPO training data with `ground_truth_flag` fields pointing to your challenges.
4. Point the training config to the new OpenEnv endpoint.

No changes to the reward function, tool definitions, or training loop are needed.

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with clear, descriptive commits.
3. Run `pytest tests/` and `ruff check .` to verify nothing is broken.
4. Open a PR against `main` with a description of what changed and why.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](./LICENSE).
