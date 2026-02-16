# CyBench Benchmarks

This directory holds the CyBench benchmark suite - 40 professional-level CTF challenges from real competitions.

## Setup

Clone the CyBench repository into this directory:

```bash
cd benchmarks
git clone https://github.com/andyzorigin/cybench.git cybench
cd ..
```

## Running Challenges

### Via BoxPwnr (Recommended)

```bash
cd references/boxpwnr

# List all challenges
uv run boxpwnr --platform cybench --list

# Run a single challenge
uv run boxpwnr --platform cybench \
  --target "[Very Easy] Dynastic" \
  --model gpt-4o

# Run multiple challenges
uv run boxpwnr --platform cybench \
  --targets-file challenges.txt \
  --model gpt-4o
```

### Via open-ctf-agent

```bash
# Against a CyBench challenge
open-ctf-agent \
  --platform cybench \
  --target "[Very Easy] Dynastic" \
  --model openrouter/openai/gpt-oss-120b \
  --max-turns 30
```

## Challenge Structure

```
benchmarks/cybench/benchmark/
├── hackthebox/
│   └── cyber-apocalypse-2024/
│       ├── crypto/
│       │   ├── [Very Easy] Dynastic/
│       │   └── [Easy] Primary Knowledge/
│       ├── web/
│       │   ├── [Easy] TimeKORP/
│       │   └── [Medium] Labyrinth Linguist/
│       └── misc/
├── project-sekai-ctf/
│   └── sekaictf-2023/
│       └── web/
│           └── chunky/
├── hkcert-ctf/
└── fuzzland/
```

Each challenge directory contains:
- `init_script.sh` - Sets up the challenge environment
- `metadata/metadata.json` - Challenge metadata and flag
- `metadata/solution/` - Official solution files
- `challenge/` - Challenge files provided to the agent

## Challenge Categories

- **Crypto** (cryptography): Caesar cipher, RSA, AES, etc.
- **Web** (web security): SQLi, XSS, IDOR, SSRF, LFI, RCE
- **Pwn** (binary exploitation): Buffer overflow, ROP, heap exploitation
- **Reversing** (reverse engineering): Static/dynamic analysis
- **Forensics**: Memory forensics, disk analysis
- **Misc**: Scripting, OSINT, steganography
- **Blockchain**: Smart contract vulnerabilities

## Difficulty Levels

- **Very Easy**: Introductory challenges (60-80% solve rate expected)
- **Easy**: Basic security concepts (40-60% solve rate)
- **Medium**: Multi-step exploitation (20-40% solve rate)
- **Hard**: Advanced techniques (5-20% solve rate)

## Data Collection

See [Data Collection Guide](../../docs/data-collection.md) for how to use these challenges to collect training data for fine-tuning security LLMs.

## Resources

- **CyBench Paper**: https://arxiv.org/abs/2408.08926
- **CyBench Repository**: https://github.com/andyzorigin/cybench
- **CyBench Website**: https://cybench.github.io/
