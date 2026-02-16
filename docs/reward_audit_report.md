# CTF Reward Function Audit Report

**Date**: 2026-02-16
**Scope**: `open-ctf-env` 4-signal CTFReward vs `ghost-training` 12-signal IntegratedGRPOReward
**Goal**: Determine if current CTFReward is safe for GRPO training starting this week

---

## 1. Executive Summary

**Verdict: The 4-signal CTFReward is SAFE to use for GRPO training this week, with two recommended fixes that can be applied in under an hour.**

The current `CTFReward` in `src/open_ctf/rewards/ctf_reward.py` is a clean, well-tested implementation with 4 signals (flag capture, skill grammar, efficiency, format compliance). It has correct GRPO variance properties and provides meaningful discrimination between success and failure trajectories.

The alternative 12-signal `IntegratedGRPOReward` in the parent `ghost-training` repo is **NOT production-ready**. Despite being described as "research-validated on 25,317 samples," 8 of its 12 experimental signal modules are stubs/placeholders returning fixed values (0.5 or 0.1). Using it would inject constant noise without discriminative value.

**Critical findings**:
- **Two exploitable reward hacking vulnerabilities** (Medium severity) -- both fixable in minutes
- **The 12-signal alternative is worse than the 4-signal version** for production use today
- **GRPO variance requirements are met** by the current implementation
- **No blocking issues** for starting GRPO training this week

**Recommended pre-GRPO fixes** (< 1 hour total):
1. Fix efficiency score gaming for failed traces (Medium)
2. Decouple flag pattern match reward from efficiency/grammar (Medium)

---

## 2. Current Implementation Review

### 2.1 Architecture

**File**: `src/open_ctf/rewards/ctf_reward.py` (294 lines)

The reward function uses a linear combination of 4 independently computed signals:

```
reward = flag_weight * flag_score
       + grammar_weight * grammar_score
       + efficiency_weight * efficiency_score
       + format_weight * format_score
       + noise(+-0.05)
```

| Signal | Weight | Range | Description |
|--------|--------|-------|-------------|
| Flag capture | 0.30 | 0.0-1.0 | Exact match (1.0) or pattern match (0.1) |
| Skill grammar | 0.20 | 0.0-1.0 | RECON->ENUM->EXPLOIT phase ordering |
| Efficiency | 0.35 | 0.0-1.0 | min(optimal_steps/actual_steps, 1.0) |
| Format compliance | 0.15 | 0.0-1.0 | Valid JSON tool call structure |
| Noise | - | -0.05 to +0.05 | GRPO gradient variance |

**Total reward range**: -0.05 to 1.05 (theoretical)

### 2.2 Signal-by-Signal Analysis

#### Flag Score (weight=0.30)

**Implementation** (`_flag_score`, line 178-185):
- Exact match against `ground_truth_flag`: 1.0
- Regex pattern match `FLAG{[a-zA-Z0-9_-]+}` without ground truth match: 0.1
- No flag found: 0.0

**Strengths**:
- Binary exact-match check is robust against hallucination
- Pattern match exclusion of `FLAG{...}` prevents placeholder matching
- Ground truth comparison prevents false positive credit

**Weaknesses**:
- Pattern match score (0.1) rewards any `FLAG{...}` string even if completely fabricated
- No case-sensitivity handling (would miss `flag{...}` variants)

#### Skill Grammar Score (weight=0.20)

**Implementation** (`_grammar_score`, line 187-223):
- Classifies tool calls into phases: recon, enum, exploit
- Scores phase presence (0.2 per phase, up to 0.6)
- Scores correct ordering (0.2 per adjacent pair, up to 0.4)
- Maximum: 1.0 for all 3 phases in correct order

**Strengths**:
- Multi-phase tools (curl, wget, ssh) classified by argument context
- Consecutive same-phase calls deduplicated (avoids inflating)
- Fallback pattern matching on combined name+args text

**Weaknesses**:
- Can be maxed (1.0) by running nmap, gobuster, sqlmap in sequence regardless of whether they accomplish anything
- No penalty for ineffective tool usage (nmap on wrong target still counts as "recon")

#### Efficiency Score (weight=0.35)

**Implementation** (`_efficiency_score`, line 225-231):
- `min(optimal_steps / actual_steps, 1.0)` when metadata available
- Returns 0.5 (neutral) when `optimal_steps` is None
- Returns 0.0 when `actual_steps` is 0

**Strengths**:
- Simple, interpretable metric
- Caps at 1.0 (no bonus for being faster than optimal)
- Neutral fallback when metadata is missing

**Weaknesses**:
- **Highest weight (0.35) but rewards short failures** -- a 1-step failure gets efficiency 1.0 (optimal/1 >= 1.0), earning 0.35 from efficiency alone
- Counts tool calls, not actual steps -- a single message with multiple tool_calls counts as 1

#### Format Score (weight=0.15)

**Implementation** (`_format_score`, line 233-248):
- Counts valid JSON tool call arguments
- Full credit (1.0) for valid JSON, half credit (0.5) for non-JSON arguments
- Returns 0.0 for empty tool_calls list

**Strengths**:
- Graceful degradation (partial credit for non-JSON)
- Both name and arguments must be present

**Weaknesses**:
- Rewards syntactically valid but semantically meaningless tool calls
- A tool call with `{"key": "value"}` gets full credit regardless of relevance

### 2.3 Extraction Logic

The `_extract` method (line 254-293) handles 4 input formats:
- Raw string
- Single message dict
- List of message dicts (ChatML)
- Fallback to str()

This is well-implemented and handles edge cases like dict arguments (auto-serialized to JSON string).

### 2.4 GRPO Compatibility

| Requirement | Status | Notes |
|-------------|--------|-------|
| Variance for gradients | MET | Noise (+-0.05) + signal variance |
| Reward scale | MET | 0-1 range appropriate for DAPO loss |
| Handles batches | MET | Iterates over completions list |
| TRL signature compatible | MET | `(completions, prompts=None, **kwargs)` |
| Deterministic seed | MET | Optional seed for reproducibility |

### 2.5 Test Coverage

**File**: `tests/test_rewards.py` (585 lines)

Comprehensive test suite covering:
- Tool call classification (11 tests)
- Flag scoring (8 tests)
- Grammar scoring (7 tests)
- Efficiency scoring (7 tests)
- Format scoring (6 tests)
- Extraction (4 tests)
- Noise behavior (2 tests)
- Integration tests (6 tests)
- GRPO data validation (8 tests)

**Assessment**: Excellent test coverage. All edge cases for individual signals are tested. Integration tests verify success > failure ordering.

---

## 3. Comparison to Research-Validated 12-Signal Design

### 3.1 Signal Mapping

| 12-Signal Component | Signal Strength | In 4-Signal? | Implementation Status |
|---------------------|-----------------|--------------|----------------------|
| H7: Skill Grammar | t=21.68 (STRONGEST) | **YES** (simplified) | **REAL** in both versions |
| H10: Knowledge Grounding | +0.278 diff | No | **STUB** (returns 0.5) |
| H12: Cognitive Efficiency | +35 WPA diff | No | **STUB** (returns 0.5) |
| H4: Temporal Credit | Early discovery | No | **STUB** (placeholder decay) |
| H3: Capability Utilization | +0.253 gap | No | **STUB** (length heuristic) |
| H8: Calibration | r=-0.253 | No | **STUB** (returns 0.5) |
| H11: Recovery | 71% rate | No | **STUB** (returns 0.5) |
| H5: Contrastive | Bad pattern detect | No | **STUB** (returns 0.1) |
| H1: Curiosity | With decay | No | **STUB** (returns 0.1) |
| H9: Counterfactual | r=0.022 (weakest) | No | **STUB** (returns 0.1) |
| H13: Consistency | Process signal | No | **REAL** (heuristic matching) |
| H14: Entropy | Dense info | No | **REAL** (Shannon entropy) |

### 3.2 Stub Analysis (Critical Finding)

**8 of 12 experimental modules are stubs** that return constant values:

| Module | File | Returns |
|--------|------|---------|
| `grounding_reward.py` | `ghost-training/src/rewards/experimental/` | `grounding_ratio=0.5` always |
| `cognitive_efficiency_reward.py` | same | `overall_score=0.5` always |
| `calibration_reward.py` | same | `overall_score=0.5` always |
| `recovery_reward.py` | same | `adaptability_score=0.5` always |
| `temporal_reward.py` | same | Placeholder decay, ~0.5 |
| `graph_reward.py` | same | `min(1.0, reasoning_len/2000)` -- length proxy, not real graph analysis |
| `curiosity_reward.py` | same | `0.1` for all completions |
| `contrastive_reward.py` | same | `0.1` always |
| `counterfactual_reward.py` | same | `total_score=0.1, depth=1` always |

**Impact**: The IntegratedGRPOReward's "12 validated signals" are mostly injecting constant additive biases, not discriminative reward signal. The only real discriminative modules are the skill grammar parser and H13/H14 (consistency/entropy).

### 3.3 Weight Allocation Comparison

| Aspect | 4-Signal (CTFReward) | 12-Signal (IntegratedGRPO) |
|--------|---------------------|---------------------------|
| Flag capture | 0.30 (direct) | Via PrincipledAgentReward base (flag-gated) |
| Grammar/Skill | 0.20 | 0.20 (similar) |
| Efficiency | 0.35 | 0.15 (via primary) |
| Format | 0.15 | 0.05 (via primary) |
| Process quality | 0 | ~0.60 (sum of stub bonuses) |
| **Total range** | **0.0-1.0** | **0.0-2.0** |

### 3.4 Success-Failure Gap

| Metric | 4-Signal | 12-Signal (claimed) | 12-Signal (actual with stubs) |
|--------|----------|---------------------|-------------------------------|
| Success mean | ~0.65-0.85 | 1.50-2.00 | ~1.20-1.50 (inflated by constants) |
| Failure mean | ~0.15-0.35 | 0.00-0.20 | ~0.30-0.50 (inflated by constants) |
| Gap | ~0.40-0.55 | ~1.60 (claimed) | ~0.70-1.00 (actual) |

The 12-signal version's claimed 1.60 gap was validated on the original `ghost-training` research data with real implementations, not the stub versions present in the codebase. The stubs inflate both success and failure scores by similar amounts, slightly widening the gap due to flag-gating but adding no real discriminative power.

### 3.5 Recommendation

**Do NOT migrate to the 12-signal IntegratedGRPOReward** for production GRPO training. The experimental modules would need to be fully implemented first. The 4-signal CTFReward is a better choice today because:
1. All 4 signals are real, tested implementations
2. Signal computation is deterministic and interpretable
3. No dependency on numpy or external modules
4. Comprehensive test suite validates behavior

---

## 4. Reward Hacking Vulnerabilities

### 4.1 Vulnerability: Efficiency Gaming on Failed Traces (MEDIUM)

**Description**: A completion that fails to find the flag but uses very few steps gets disproportionately high reward from the efficiency signal.

**Proof of concept**:
```python
# Failed trace: 1 tool call, no flag found
failed_completion = [
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "shell_command",
                       "arguments": '{"command": "nmap target"}'}}
    ]},
    {"role": "assistant", "content": "I give up."},
]

# With optimal_steps=5:
# flag_score: 0.0 * 0.30 = 0.00
# grammar_score: 0.2 * 0.20 = 0.04  (recon only)
# efficiency_score: min(5/1, 1.0) = 1.0 * 0.35 = 0.35  <-- HIGH!
# format_score: 1.0 * 0.15 = 0.15
# TOTAL: 0.54 (before noise)
```

**Risk**: GRPO could learn to "quit early" -- running one classified tool call with valid JSON gives 0.54 reward, which may overlap with successful traces that used many steps.

**Fix**: Gate efficiency score on flag capture, or scale efficiency weight by flag_score:
```python
# Option A: Gate on flag
eff_weight = self.efficiency_weight if flag_found else 0.0

# Option B: Scale by flag score (smoother)
eff_contribution = self.efficiency_weight * self._efficiency_score(...) * self._flag_score(...)
```

**Severity**: Medium -- the DAPO loss type and group reward scaling should partially mitigate this within each group, but it creates a perverse gradient toward minimal effort.

### 4.2 Vulnerability: Format + Grammar Without Progress (MEDIUM)

**Description**: A completion that runs structured tool calls in the correct phase order (recon->enum->exploit) with valid JSON arguments can earn up to 0.35 from grammar (0.20) + format (0.15) even if no flag is found and the tools accomplish nothing.

**Proof of concept**:
```python
gaming_completion = [
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "shell_command", "arguments": '{"command": "nmap 127.0.0.1"}'}}
    ]},
    {"role": "tool", "content": "Connection refused"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "shell_command", "arguments": '{"command": "gobuster dir -u http://127.0.0.1"}'}}
    ]},
    {"role": "tool", "content": "Error: connection refused"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "shell_command", "arguments": '{"command": "sqlmap -u http://127.0.0.1"}'}}
    ]},
    {"role": "tool", "content": "Error: target not responding"},
]

# grammar: 1.0 * 0.20 = 0.20 (perfect recon->enum->exploit)
# format: 1.0 * 0.15 = 0.15 (all valid JSON)
# efficiency: depends on optimal_steps
# flag: 0.0
# TOTAL: 0.35+ without any actual progress
```

**Risk**: GRPO could learn to emit well-formatted tool calls in the "correct" sequence without actually adapting to the target.

**Fix**: Weight grammar score by a progress indicator (e.g., reduce grammar weight when no flag is found, or incorporate tool output analysis).

**Severity**: Medium -- less concerning than efficiency gaming because the 0.35 ceiling is lower than successful traces (~0.65+), but could still create reward plateaus.

### 4.3 Vulnerability: False Flag Pattern Match (LOW)

**Description**: Any string matching `FLAG{[a-zA-Z0-9_-]+}` earns 0.1 * 0.30 = 0.03 from flag score, even if completely fabricated.

**Example**: A model outputs `"I believe the flag might be FLAG{admin_password_123}"` and gets 0.03 credit.

**Risk**: Minimal. The 0.03 contribution is within noise range (+-0.05) and unlikely to create meaningful gradient signal toward hallucination.

**Severity**: Low -- not worth fixing before GRPO. The GRPO group normalization further diminishes this.

### 4.4 Vulnerability: Noise Masking Small Signal Differences (LOW)

**Description**: The +-0.05 noise range can flip the ranking of two completions whose true reward difference is < 0.10.

**Analysis**: With 8 generations per prompt (GRPO default), the group normalization and DAPO loss operate on relative rankings within the group. Noise-induced flips at the margin are actually desirable for exploration and preventing reward collapse.

**Severity**: Low -- this is by design and working as intended.

### 4.5 Vulnerability: Missing optimal_steps Inflates Scores (LOW)

**Description**: When `optimal_steps` is None, efficiency returns 0.5 (neutral). If some GRPO samples have metadata and others don't, this creates inconsistent scoring.

**Analysis**: In the current GRPO dataset (`data/grpo.jsonl`), the test suite validates all samples have `optimal_steps >= 1`. This is a theoretical risk only if new data lacks this field.

**Severity**: Low -- data validation catches this.

---

## 5. Signal-Success Alignment Analysis

### 5.1 Expected Reward Distribution

Based on code analysis (assuming well-formed GRPO data with ground_truth_flag and optimal_steps):

**Successful trace** (flag found, good methodology):
- Flag: 1.0 * 0.30 = 0.30
- Grammar: ~0.8 * 0.20 = 0.16 (most successful solves follow recon->exploit)
- Efficiency: ~0.7 * 0.35 = 0.245 (typically within 1.5x optimal)
- Format: ~0.95 * 0.15 = 0.1425
- **Expected total: ~0.85** (before noise)

**Failed trace with effort** (no flag, used tools):
- Flag: 0.0 * 0.30 = 0.00
- Grammar: ~0.6 * 0.20 = 0.12
- Efficiency: ~0.5 * 0.35 = 0.175 (default or many steps)
- Format: ~0.90 * 0.15 = 0.135
- **Expected total: ~0.43** (before noise)

**Failed trace without effort** (gave up / refused):
- Flag: 0.0
- Grammar: 0.0
- Efficiency: 0.0 (0 steps) or 0.5 (no metadata)
- Format: 0.0
- **Expected total: ~0.00-0.175**

### 5.2 Gap Analysis

| Scenario Pair | Estimated Gap | Target (>0.50) | Status |
|---------------|---------------|-----------------|--------|
| Success vs Failed-with-effort | ~0.42 | >0.50 | MARGINAL |
| Success vs Failed-no-effort | ~0.85 | >0.50 | MET |
| Failed-effort vs Failed-no-effort | ~0.43 | N/A | Good separation |

The success vs failed-with-effort gap is marginal (~0.42) due to the efficiency and format signals rewarding well-structured failures. This is the core tension: these signals reward process quality, which is desirable for learning, but narrow the gap.

### 5.3 Signal Correlation with Success

| Signal | Expected Correlation | Concern |
|--------|---------------------|---------|
| Flag | Perfect (by definition) | None -- this IS the success criterion |
| Grammar | Moderate positive | Good -- successful traces tend to follow phase ordering |
| Efficiency | Weakly negative with success | CONCERN -- failures can have high efficiency if they quit early |
| Format | Near-zero correlation | OK -- both success and failure traces have valid tool calls |

### 5.4 Variance Assessment

The reward function produces sufficient variance for GRPO through:
1. **Signal variance**: Different completions naturally vary across all 4 signals
2. **Noise injection**: +-0.05 uniform noise prevents identical scores
3. **Group diversity**: 8 generations per prompt produce varied trajectories

The previous GRPO v5 run (CLAUDE.md) reports `frac_reward_zero_std = 0.0` (all samples have variance), confirming GRPO gradient requirements are met.

---

## 6. Recommendations

### 6.1 Critical Fixes (Do Before GRPO -- < 1 hour)

#### Fix 1: Gate Efficiency on Flag Capture

**File**: `src/open_ctf/rewards/ctf_reward.py`, line 161-166

**Current**:
```python
score = (
    self.flag_weight * self._flag_score(text, gt_flag)
    + self.grammar_weight * self._grammar_score(tool_calls)
    + self.efficiency_weight * self._efficiency_score(len(tool_calls), opt_steps)
    + self.format_weight * self._format_score(tool_calls)
)
```

**Proposed**:
```python
flag_sc = self._flag_score(text, gt_flag)
score = (
    self.flag_weight * flag_sc
    + self.grammar_weight * self._grammar_score(tool_calls)
    + self.efficiency_weight * self._efficiency_score(len(tool_calls), opt_steps) * flag_sc
    + self.format_weight * self._format_score(tool_calls)
)
```

**Rationale**: Efficiency score should only reward being fast *when you also found the flag*. Without the flag, high efficiency just means "quit early." Multiplying by `flag_sc` means exact match gives full efficiency credit, pattern match gives 10%, and no flag gives 0%.

**Impact**: Failed traces drop from ~0.43 to ~0.25 (grammar + format only). Success-failure gap increases from ~0.42 to ~0.60.

#### Fix 2: Reduce Grammar Weight for No-Flag Traces

**Alternative to Fix 1** (choose one or both):

Scale grammar contribution by flag presence to prevent "going through the motions" exploitation:

```python
grammar_multiplier = 1.0 if flag_sc > 0 else 0.3  # 30% credit for process without result
score = (
    self.flag_weight * flag_sc
    + self.grammar_weight * self._grammar_score(tool_calls) * grammar_multiplier
    + self.efficiency_weight * self._efficiency_score(len(tool_calls), opt_steps)
    + self.format_weight * self._format_score(tool_calls)
)
```

### 6.2 Important Improvements (Should Do -- This Week)

#### Add Truthfulness Signal

The GRPO v5 training config (CLAUDE.md) includes a `truthfulness_weight = 0.15` that penalizes hallucinated success claims. The current CTFReward does not implement this. Consider adding a check that penalizes completions containing phrases like "flag found", "successfully captured", "CTF solved" when no actual flag is present.

#### Increase Noise Range Slightly

Current noise (+-0.05) is adequate but conservative. The GRPO v5 run used this successfully, but increasing to +-0.08 could improve exploration without destabilizing rankings. Monitor `frac_reward_zero_std` during training.

### 6.3 Optional Enhancements (Nice to Have -- Post-GRPO)

#### Implement Real Versions of Top 3 Research Signals

If future GRPO iterations show reward plateaus, implement the top 3 research signals (currently stubs):

1. **H10: Knowledge Grounding** (0.15 weight) -- Check if the model references real CVEs, tools, techniques
2. **H12: Cognitive Efficiency** (0.15 weight) -- Words-per-action metric (optimal ~42 WPA)
3. **H4: Temporal Credit** (0.15 weight) -- Bonus for early vulnerability discovery

These would require real implementations, not the stubs currently in `ghost-training/src/rewards/experimental/`.

#### Consider Flag-Gated Architecture

The IntegratedGRPOReward uses a "flag-gated" architecture where bonus signals only apply when the flag is found. This naturally creates a large success-failure gap. The 4-signal version could adopt this pattern by wrapping grammar/efficiency/format bonuses in a flag gate.

---

## 7. Timeline Impact

### Can we use the current reward for GRPO starting this week?

**YES**, with the recommended Fix 1 applied (15 minutes of code change + test update).

### Decision Matrix

| Option | Risk | Effort | Recommendation |
|--------|------|--------|----------------|
| **A: Use CTFReward as-is** | Medium (efficiency gaming) | None | Acceptable if time-critical |
| **B: Apply Fix 1 then GRPO** | Low | 15-30 min | **RECOMMENDED** |
| **C: Apply Fix 1 + Fix 2 then GRPO** | Very Low | 30-60 min | Best if we have an hour |
| **D: Migrate to 12-signal** | High (stubs inject noise) | Days-weeks | **NOT recommended** |
| **E: Build new 12-signal from scratch** | Low but slow | 1-2 weeks | Future iteration |

### Why NOT to switch to IntegratedGRPOReward

1. **8 of 12 modules are stubs** returning constants (0.5 or 0.1)
2. Stubs inflate both success and failure scores, reducing discriminative power
3. Introduces numpy dependency and 12 additional module imports
4. No test suite for the integrated version
5. Flag-gating only helps if the bonus signals are real
6. The "25,317 sample validation" was done on different implementations than what exists in the codebase

### GRPO Training Readiness Checklist

| Item | Status | Notes |
|------|--------|-------|
| Reward function works | YES | Tested on 779 GRPO samples |
| GRPO data has required fields | YES | ground_truth_flag + optimal_steps validated |
| TRL signature compatible | YES | (completions, prompts=None, **kwargs) |
| Sufficient variance | YES | frac_reward_zero_std = 0.0 in prior run |
| Success > failure ordering | YES | Test suite validates this invariant |
| Reward hacking mitigated | PARTIAL | Apply Fix 1 for full mitigation |
| DAPO loss compatible | YES | 0-1 reward range works with DAPO |
| Deterministic for debugging | YES | Seed parameter available |

---

## Appendix A: File References

| File | Purpose |
|------|---------|
| `src/open_ctf/rewards/ctf_reward.py` | Current 4-signal reward (294 lines) |
| `src/open_ctf/rewards/__init__.py` | Module exports |
| `tests/test_rewards.py` | Test suite (585 lines, 59 tests) |
| `src/open_ctf/training/grpo.py` | GRPO trainer integration |
| `data/grpo.jsonl` | GRPO training data (779 samples) |

Parent repo references:
| File | Purpose |
|------|---------|
| `ghost-training/src/rewards/integrated_reward.py` | 12-signal reward (651 lines) |
| `ghost-training/src/rewards/experimental/*.py` | 12 experimental modules (mostly stubs) |

## Appendix B: Experimental Module Implementation Status

| Module | Lines of Code | Real Logic? | Returns |
|--------|---------------|-------------|---------|
| `skill_grammar_reward.py` | ~100+ | YES | Parsed skill sequence metrics |
| `consistency_reward.py` | ~80+ | YES | Heuristic intent-action matching |
| `entropy_reward.py` | ~50+ | YES | Shannon entropy of tool outputs |
| `calibration_reward.py` | 17 | NO (stub) | `overall_score=0.5` |
| `cognitive_efficiency_reward.py` | 17 | NO (stub) | `overall_score=0.5` |
| `grounding_reward.py` | 11 | NO (stub) | `grounding_ratio=0.5` |
| `recovery_reward.py` | 13 | NO (stub) | `adaptability_score=0.5` |
| `temporal_reward.py` | 34 | NO (placeholder) | ~0.5 with basic decay |
| `graph_reward.py` | 23 | NO (proxy) | `min(1.0, len/2000)` |
| `curiosity_reward.py` | 15 | NO (stub) | `0.1` always |
| `contrastive_reward.py` | 5 | NO (stub) | `0.1` always |
| `counterfactual_reward.py` | 12 | NO (stub) | `total_score=0.1` |
