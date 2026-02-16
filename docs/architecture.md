# Architecture

Open CTF Environment is a 2-stage training pipeline for fine-tuning LLMs on CTF tasks using BoxPwnr agent traces.

## System Overview

```mermaid
graph TB
    subgraph "Data Collection"
        A[CTF Challenges<br/>CyBench/HTB/PortSwigger] --> B[BoxPwnr Agent]
        B --> C[Raw Traces<br/>conversation.json + stats.json]
    end

    subgraph "Data Processing"
        C --> D[BoxPwnrConverter<br/>Lossless trace conversion]
        D --> E[ChatML JSONL<br/>17 native tools preserved]
        E --> F[DatasetSplitter<br/>SFT/GRPO separation]
        F --> G[SFT Dataset<br/>Successful traces]
        F --> H[GRPO Dataset<br/>Multi-turn + ground_truth_flag]
    end

    subgraph "Training Pipeline"
        G --> I[SFT Training<br/>Unsloth + TRL<br/>LoRA r=64]
        I --> J[LoRA Adapter<br/>~60MB]
        J --> K[Merge<br/>Adapter + Base]
        K --> L[SFT Model<br/>BF16 merged]

        H --> M[GRPO Training<br/>TRL GRPOTrainer<br/>CTF Reward]
        L --> M
        M --> N[GRPO Model<br/>Policy optimized]
    end

    subgraph "Deployment"
        N --> O[Export<br/>llama.cpp]
        O --> P[GGUF Q4_K_M<br/>~15GB]
        N --> Q[vLLM Serve<br/>BF16/FP8]
        P --> R[Ollama/llama.cpp]
        Q --> S[API Server]
    end

    style I fill:#e1f5e1
    style M fill:#e1f5e1
    style O fill:#fff4e1
    style B fill:#e1e8f5
```

## Module Structure

```mermaid
graph LR
    subgraph "CLI Layer (src/open_ctf/cli/)"
        CLI1[train.py<br/>sft/grpo/merge]
        CLI2[convert_traces.py]
        CLI3[run_agent.py]
        CLI4[evaluate.py]
        CLI5[validate_pipeline.py]
        CLI6[export_gguf.py]
        CLI7[split_dataset.py]
    end

    subgraph "Core Modules (src/open_ctf/)"
        D1[data/<br/>converter.py<br/>splitter.py]
        T1[training/<br/>sft.py<br/>grpo.py]
        R1[rewards/<br/>ctf_reward.py]
        F1[formatters/<br/>base.py<br/>qwen3/glm4/devstral]
        A1[agent/<br/>runner.py]
        E1[eval/<br/>evaluator.py]
        V1[envs/<br/>gym_env.py]
    end

    CLI1 --> T1
    CLI1 --> R1
    CLI2 --> D1
    CLI3 --> A1
    CLI4 --> E1
    CLI7 --> D1

    T1 --> F1
    T1 --> R1
    E1 --> A1

    style T1 fill:#e1f5e1
    style R1 fill:#ffe1e1
    style D1 fill:#e1e8f5
```

## Training Data Flow

```mermaid
sequenceDiagram
    participant BP as BoxPwnr Agent
    participant Conv as BoxPwnrConverter
    participant Split as DatasetSplitter
    participant SFT as SFT Trainer
    participant GRPO as GRPO Trainer

    BP->>Conv: conversation.json + stats.json
    Note over Conv: Preserve 17 native tools<br/>Handle tool-calling + chat formats<br/>Extract reasoning, flags
    Conv->>Split: ChatML JSONL (all traces)

    Note over Split: Success → SFT<br/>Multi-turn + flag → GRPO<br/>Cross-reference flags
    Split->>SFT: sft.jsonl
    Split->>GRPO: grpo.jsonl

    Note over SFT: Unsloth FastLanguageModel<br/>LoRA r=64, BF16<br/>Packing enabled
    SFT->>GRPO: SFT checkpoint (merged)

    Note over GRPO: TRL GRPOTrainer<br/>CTFReward (4 components)<br/>DAPO loss, beta=0.001
    GRPO->>GRPO: Final model
```

## CTF Reward Function

```mermaid
graph TD
    C[Completion] --> E[Extract Tool Calls + Text]
    E --> F1[Flag Score 0.30]
    E --> F2[Grammar Score 0.20]
    E --> F3[Efficiency Score 0.35]
    E --> F4[Format Score 0.15]

    F1 --> M[Weighted Sum]
    F2 --> M
    F3 --> M
    F4 --> M

    M --> N[Add Noise ±0.05]
    N --> R[Final Reward]

    subgraph "Flag Score"
        F1A[Exact match:<br/>ground_truth_flag] --> F1B[1.0]
        F1C[Pattern match:<br/>FLAG\{...\}] --> F1D[0.1]
        F1E[No flag] --> F1F[0.0]
    end

    subgraph "Grammar Score"
        F2A[Classify tools:<br/>RECON/ENUM/EXPLOIT] --> F2B[Check phase order]
        F2B --> F2C[Presence: 0.6<br/>Order: 0.4]
    end

    subgraph "Efficiency Score"
        F3A[optimal_steps /<br/>actual_steps] --> F3B[min(..., 1.0)]
        F3C[No metadata] --> F3D[0.5 neutral]
    end

    subgraph "Format Score"
        F4A[Valid tool_calls<br/>JSON structure] --> F4B[valid / total]
    end

    style F1 fill:#ffe1e1
    style F2 fill:#e1f5e1
    style F3 fill:#e1e8f5
    style F4 fill:#fff4e1
```

## Model Formatters

```mermaid
graph TB
    M[Model ID] --> F[ModelFormatter.from_model_id]

    F -->|qwen/openthinker| Q[Qwen3Formatter]
    F -->|glm| G[GLM4Formatter]
    F -->|devstral/mistral| D[DevstralFormatter]

    subgraph "Qwen3Formatter"
        Q1[ChatML format<br/>tool_calls array<br/>Hermes-style]
    end

    subgraph "GLM4Formatter"
        G1[observation role<br/>function call format<br/>Jinja template]
    end

    subgraph "DevstralFormatter"
        D1[INST tags<br/>Strict alternation<br/>Mistral tool format]
    end

    Q --> Q1
    G --> G1
    D --> D1

    Q1 --> T[format_messages]
    G1 --> T
    D1 --> T

    T --> O[Model-native text<br/>Ready for tokenization]
```

## Training Execution Flow

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as open-ctf-train
    participant SFT as train_sft
    participant GRPO as train_grpo
    participant US as Unsloth
    participant TRL as TRL Trainer
    participant HF as HuggingFace

    U->>CLI: open-ctf-train sft
    CLI->>SFT: Load config + data

    alt Unsloth Available
        SFT->>US: _set_moe_backend()
        Note over US: UNSLOTH_MOE_BACKEND=grouped_mm
        SFT->>US: FastLanguageModel.from_pretrained
        SFT->>US: get_peft_model (LoRA)
        US->>TRL: SFTTrainer + SFTConfig
    else Unsloth Unavailable
        SFT->>HF: AutoModelForCausalLM
        SFT->>HF: PEFT LoraConfig + get_peft_model
        SFT->>HF: Add for_training/for_inference stubs
        HF->>TRL: SFTTrainer + SFTConfig
    end

    TRL->>SFT: Training complete
    SFT->>CLI: Return adapter path

    U->>CLI: open-ctf-train grpo
    CLI->>GRPO: Load SFT model + GRPO data

    alt Unsloth Available
        GRPO->>US: _set_moe_backend()
        GRPO->>US: Load model
    else OPEN_CTF_NO_UNSLOTH=1
        GRPO->>HF: Load with stubs
    end

    GRPO->>TRL: GRPOTrainer + CTFReward
    TRL->>GRPO: Training complete
    GRPO->>CLI: Return final model
```

## Hardware Compatibility

```mermaid
graph TB
    subgraph "DGX Spark GB10 (ARM64)"
        GB1[128GB Unified Memory]
        GB2[Blackwell sm_121a]
        GB3[99KB Shared Mem Limit]
    end

    subgraph "Container Requirements"
        C1[ARM64 Architecture<br/>NOT AMD64]
        C2[Unsloth 2026.2.1+<br/>Transformers 5.0+<br/>TRL 0.28+]
        C3[CUDA 12.1/13.0<br/>Triton 3.6+]
    end

    subgraph "Training Configuration"
        T1[UNSLOTH_MOE_BACKEND=<br/>grouped_mm]
        T2[load_in_4bit=False<br/>Use BF16 LoRA]
        T3[LoRA r=64, alpha=128<br/>target: attn+FFN+out_proj]
        T4[Router layers excluded]
    end

    GB3 -.->|Workaround| T1
    GB2 -.->|Requires| C3
    GB1 -.->|Enables| T2

    C1 --> D1[unsloth-blackwell:v3<br/>20.5GB, ARM64]
    C2 --> D1
    C3 --> D1

    T1 --> D1
    T2 --> D1
    T3 --> D1
    T4 --> D1

    D1 --> R[Training Works<br/>~60GB VRAM used]

    style GB3 fill:#ffe1e1
    style T1 fill:#e1f5e1
    style D1 fill:#e1e8f5
    style R fill:#d4edda
```

## Evaluation Pipeline

```mermaid
graph LR
    subgraph "Model Serving"
        M1[Trained Model] --> S1[vLLM Server]
        M1 --> S2[Ollama]
        M1 --> S3[llama.cpp]
    end

    subgraph "Challenge Setup"
        C1[CyBench Benchmark] --> D1[Docker Compose Up]
        D1 --> T1[Target Running]
    end

    subgraph "Evaluation"
        S1 --> E1[ModelEvaluator]
        S2 --> E1
        S3 --> E1
        T1 --> E1

        E1 --> R1[AgentRunner]
        R1 --> BP[BoxPwnr Solver]
        BP --> O1[Trace Output]
    end

    subgraph "Metrics"
        O1 --> M2[Solve Rate]
        O1 --> M3[Avg Turns]
        O1 --> M4[Avg Time]
        O1 --> M5[Flag Found %]
    end

    M2 --> CM[Compare Reports]
    M3 --> CM
    M4 --> CM
    M5 --> CM

    style E1 fill:#e1f5e1
    style CM fill:#fff4e1
```

## BoxPwnr Tool Categories

```mermaid
graph TD
    T[BoxPwnr Tools<br/>17 native tools] --> C1[Shell Execution]
    T --> C2[Interactive Sessions]
    T --> C3[Tmux Control]
    T --> C4[File Operations]
    T --> C5[Code Execution]
    T --> C6[Results]

    C1 --> T1[shell_command<br/>execute_command]
    C2 --> T2[exec_command<br/>write_stdin]
    C3 --> T3[tmux_send_and_read<br/>tmux_wait_and_read<br/>tmux_read_output<br/>tmux_cancel_command]
    C4 --> T4[read_file<br/>grep<br/>file_search<br/>apply_patch]
    C5 --> T5[python_code]
    C6 --> T6[flag_found<br/>web_search<br/>list_sessions<br/>close_session]

    style T fill:#e1e8f5
    style C1 fill:#ffe1e1
    style C3 fill:#e1f5e1
    style C5 fill:#fff4e1
```

## Skill Grammar (Reward Component)

```mermaid
graph LR
    subgraph "Phase Classification"
        TC[Tool Call] --> CL{Classify}
        CL -->|nmap, masscan,<br/>rustscan, ping| R[RECON]
        CL -->|gobuster, ffuf,<br/>feroxbuster, nikto| EN[ENUM]
        CL -->|sqlmap, hydra,<br/>python_code,<br/>msfconsole| EX[EXPLOIT]
    end

    subgraph "Grammar Scoring"
        R --> SEQ{Check Sequence}
        EN --> SEQ
        EX --> SEQ

        SEQ -->|RECON before ENUM| P1[+0.2]
        SEQ -->|ENUM before EXPLOIT| P2[+0.2]
        SEQ -->|All 3 phases present| P3[+0.6]

        P1 --> SUM[Sum<br/>max 1.0]
        P2 --> SUM
        P3 --> SUM
    end

    style R fill:#e1e8f5
    style EN fill:#e1f5e1
    style EX fill:#ffe1e1
```

## Hardware Requirements

```mermaid
graph TB
    subgraph "Supported Hardware"
        H1[DGX Spark GB10<br/>128GB unified<br/>ARM64]
        H2[H100 SXM<br/>80GB VRAM<br/>AMD64]
        H3[H200 SXM<br/>141GB VRAM<br/>AMD64]
        H4[A100 80GB<br/>80GB VRAM<br/>AMD64]
    end

    subgraph "Model Sizing"
        M1[GLM-4.7-Flash<br/>30B MoE<br/>~60GB BF16 LoRA]
        M2[Qwen3-8B<br/>8B dense<br/>~24GB 4-bit LoRA]
        M3[Devstral-2-123B<br/>123B dense<br/>Requires multi-GPU]
    end

    H1 -->|Fits| M1
    H1 -->|Fits| M2
    H2 -->|Fits| M1
    H2 -->|Fits| M2
    H3 -->|Fits| M1
    H3 -->|Fits| M2
    H3 -->|Fits| M3

    M1 -.->|Special config| C1[UNSLOTH_MOE_BACKEND=<br/>grouped_mm]
    M1 -.->|No 4-bit| C2[load_in_4bit=False]

    style H1 fill:#e1f5e1
    style M1 fill:#fff4e1
    style C1 fill:#ffe1e1
```

## Key Design Decisions

### 1. Lossless Trace Conversion

**Problem**: Early converters collapsed all tools to a single `shell` tool, losing fine-grained capability data.

**Solution**: BoxPwnrConverter preserves all 17 native tool names, handles both structured `tool_calls` and chat-command `<COMMAND>` formats, and extracts reasoning from multi-part content.

### 2. Dual-Backend Training

**Problem**: Unsloth has known issues on some platforms (GB10 GRPO dtype bug, Triton shared memory limits).

**Solution**: Both `sft.py` and `grpo.py` implement dual loading:
- **Path A**: Try Unsloth with `grouped_mm` backend (fast, optimized)
- **Path B**: Fall back to HuggingFace transformers + PEFT (slower, always works)
- Controlled via `OPEN_CTF_NO_UNSLOTH=1` environment variable

### 3. Model-Specific Formatters

**Problem**: Different model families (Qwen, GLM, Mistral) have incompatible chat templates and tool-calling conventions.

**Solution**: `ModelFormatter` abstract base with auto-detection factory. Each subclass handles:
- Role token mapping (e.g., GLM's `<|observation|>` for tool results)
- Tool call serialization (structured arrays vs inline function calls)
- Reasoning tag placement (interleaved vs separate)

### 4. MoE-Aware Configuration

**Problem**: MoE models have unique constraints:
- No 4-bit quantization support (BitsAndBytes limitation)
- Triton MoE kernels exceed GB10 shared memory limit (99KB vs 104KB+ needed)
- Router layer fine-tuning can destabilize training

**Solution**:
- Auto-detect MoE models, enforce BF16 LoRA
- Set `UNSLOTH_MOE_BACKEND=grouped_mm` to use `torch._grouped_mm` (no Triton)
- Exclude router layers from LoRA target_modules
- Document memory requirements (60GB for GLM-4.7-Flash)

## Configuration Files

### `src/open_ctf/configs/training.yaml`

Central configuration for both training stages:

```yaml
model:
  name: "unsloth/GLM-4.7-Flash"
  max_seq_length: 8192
  load_in_4bit: false  # MoE requires BF16

lora:
  r: 64               # Higher capacity than Unsloth demo (8)
  alpha: 128
  target_modules: [q_proj, k_proj, v_proj, o_proj,
                   gate_proj, up_proj, down_proj, out_proj]

sft:
  epochs: 3
  batch_size: 2
  learning_rate: 2.0e-4
  packing: true       # 3x throughput improvement

grpo:
  epochs: 1
  learning_rate: 5.0e-6
  beta: 0.001         # Low KL penalty for exploration
  loss_type: dapo     # Dynamic advantage normalization
  num_generations: 4
```

### `src/open_ctf/configs/challenges.yaml`

Challenge definitions for evaluation. Maps challenge IDs to vulnerability types, difficulty, platforms.

## CLI Entry Points

After `pip install -e .`, these commands are available:

| Command | Module | Purpose |
|---------|--------|---------|
| `open-ctf-train` | `cli.train` | SFT, GRPO, merge subcommands |
| `open-ctf-convert` | `cli.convert_traces` | BoxPwnr trace → ChatML conversion |
| `open-ctf-split` | `cli.split_dataset` | Split data into SFT/GRPO sets |
| `open-ctf-agent` | `cli.run_agent` | Run agent against CTF challenges |
| `open-ctf-eval` | `cli.evaluate` | Evaluate and compare models |
| `open-ctf-validate` | `cli.validate_pipeline` | Validate setup without GPU |
| `open-ctf-export` | `cli.export_gguf` | Export LoRA to GGUF quantized format |

## Container Strategy

```mermaid
graph TB
    subgraph "Training Containers"
        C1[unsloth-blackwell:v3<br/>ARM64, 20.5GB]
        C2[nvcr.io/nvidia/pytorch:25.11-py3<br/>ARM64, 19.5GB]
        C3[gogamza/unsloth-vllm-gb10<br/>ARM64, 41.6GB]
    end

    subgraph "Use Cases"
        U1[SFT with Unsloth<br/>Fast, optimized]
        U2[GRPO with HF fallback<br/>Dtype bug workaround]
        U3[vLLM inference<br/>FlashInfer backend]
    end

    C1 -->|Preferred| U1
    C2 -->|Fallback| U2
    C3 -->|Inference only| U3

    U1 -.->|If fails| C2

    style C1 fill:#e1f5e1
    style C2 fill:#fff4e1
    style C3 fill:#e1e8f5
```

**Why Custom Containers?**
- Official `unsloth/unsloth` is **AMD64 only**
- DGX Spark is **ARM64** (aarch64)
- `unsloth-blackwell:v3` is an ARM64 build with all required libraries

**Library Versions (unsloth-blackwell:v3, built Feb 15 2026)**:
- unsloth: 2026.2.1 (latest)
- transformers: 5.1.0 (required for GLM-4.7-Flash)
- trl: 0.28.0 (latest)
- peft: 0.18.1 (latest)
- torch: 2.10.0a0 (NVIDIA optimized)
- triton: 3.6.0

## Performance Characteristics

| Stage | VRAM Usage | Time (20 samples) | Throughput |
|-------|------------|-------------------|------------|
| **SFT** | ~60GB (BF16 LoRA) | ~15-30 min | 2-3 samples/min |
| **GRPO** | ~80GB (generation overhead) | ~45-60 min | 4 gens × 1-2 samples/min |
| **Merge** | ~45GB (model + adapter) | ~3-5 min | N/A |

**GB10 Notes**:
- Unified CPU-GPU memory: 273 GB/s bandwidth (shared)
- No dedicated VRAM - memory allocation is dynamic
- `nvidia-smi` reports `[N/A]` for memory stats
- Actual usage visible via `docker stats` or container monitoring

## Extension Points

### Adding New Model Formatters

1. Create `src/open_ctf/formatters/mymodel.py` extending `ModelFormatter`
2. Implement `format_messages()` and `get_tool_definitions()`
3. Add detection logic to `base.py` factory method
4. Add test case to `validate_pipeline.py`

### Adding New Reward Components

1. Create new reward function in `src/open_ctf/rewards/`
2. Ensure `__call__(completions, prompts=None, **kwargs)` signature
3. Add `__name__` attribute for TRL logging
4. Import and instantiate in `cli/train.py` `cmd_grpo()`

### Adding New Platforms

1. Install platform support in `references/boxpwnr/`
2. Add platform to `agent/runner.py` `_get_platform()`
3. Update challenge format in `src/open_ctf/configs/challenges.yaml`
4. Document in `docs/deployment.md`
