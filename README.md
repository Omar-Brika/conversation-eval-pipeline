# Conversation Eval Pipeline

An automated QA evaluation framework for automotive voice assistants. A single lightweight LLM, served locally via [llama.cpp](https://github.com/ggerganov/llama.cpp) and orchestrated as a [LangGraph](https://langchain-ai.github.io/langgraph/) state graph, plays the roles of **Prompter**, **Router**, and **Judge** against a Device Under Test (DUT) -- closing the test loop without a human in it.

Developed during a first-year internship at **Focus Corporation R&D** (Chotrana, Ariana, Tunisia), under the curriculum of the **National School of Computer Science (ENSI)**, University of Manouba, 2025/2026. Professional supervisor: Ayman Mkaouar.

---

## Motivation

Validating an in-vehicle voice assistant's responses at scale -- across many languages, automotive domains, and under latency constraints -- is expensive with manual QA or brittle keyword-matching scripts. This pipeline replaces that manual loop: a single local LLM generates natural voice commands, routes multi-turn clarifications, and issues semantically grounded PASS/FAIL verdicts, all orchestrated automatically.

---

## Architecture

![System Design](Resources/system_design.svg)

### Offline Benchmark (Scripted DUT)

The pipeline is a **LangGraph state machine** with two nodes:

1. **Prompter Node** -- An LLM acting as a simulated test user generates natural voice commands. On the first turn it produces the initial command from the test case's intent and language. On subsequent turns it ingests the DUT's last reply plus full conversation history, decides whether the interaction is complete (`is_interaction_complete`), and either generates the next clarification command or signals the Judge to evaluate.
2. **Judge Node** -- An LLM evaluates the full conversation transcript and issues a `PASS`/`FAIL` verdict with written reasoning.

The routing decision is not a separate graph node -- it is folded into the Prompter's second LLM call (tagged `routing_voice_cmd`), halving LLM round-trips per non-terminal turn. The graph loops via conditional edges on `routing_flag`: `AWAITING_DUT` loops back to prompter, `PROCEED_TO_JUDGE` advances to judge, `TERMINATED` ends.

### Live Benchmark (Wired DUT)

An extended variant (`src/live/`) inserts real TTS/ASR stages for integration testing with a physical DUT:

```
prompter -> voice_cmd (TTS) -> DUT -> asr (Speech-to-Text) -> prompter -> ... -> judge
```

Uses dependency-injected `voice_command_fn` and `speech_recognition_fn` callbacks, with an in-memory checkpointer for state recovery on failure.

---

## Test Suite

The pipeline ships with two JSON test suites:

| File | Tests | Languages | Description |
|------|-------|-----------|-------------|
| `Json_samples/samples.json` | 216 | 12 | Full benchmark across all domains and locales |
| `Json_samples/50_samples.json` | 50 | -- | Smaller subset for quick validation |

### Suite Composition

| Property | Count | Share |
|----------|-------|-------|
| Total test cases | 216 | 100% |
| Expected PASS | 192 | 88.9% |
| Expected FAIL | 24 | 11.1% |
| Multi-turn (2-turn) cases | 24 | 11.1% |
| Languages | 12 | -- |
| Domains | 14 | -- |

**Languages (12):** Arabic (SA), Chinese (CN), English (US), French (FR), German (DE), Italian (IT), Japanese (JP), Korean (KR), Portuguese (BR), Russian (RU), Spanish (ES), Turkish (TR) -- 18 cases each.

**Domains (14):** CLIMATE, MEDIA, NAVIGATION, PHONE (24 cases each); AMBIENT_LIGHTING, DIGITAL_ASSISTANT, DRIVING_ASSISTANCE, HVAC, MAINTENANCE, RADIO, SEAT_COMFORT, VEHICLE_CONTROL, VEHICLE_STATUS, WEATHER (12 cases each).

### Failure Taxonomy

The 24 FAIL cases are split into two distinct categories:

- **WrongAction** (12 MEDIA cases) -- The DUT performs a plausible but incorrect action (e.g., intent "Play jazz" answered with "Opening the sunroof"). A failure of action selection, not comprehension.
- **Hallucination** (12 CLIMATE cases) -- The DUT fabricates a non-sequitur response (e.g., intent "Turn on the heater" answered with "I have added a space heater to your Amazon cart"). A failure to stay grounded in the conversation.

CLIMATE PASS cases use **implicit intent** phrasing ("I am cold" rather than "Turn on the heater") to test whether models can infer vehicle actions from natural language.

---

## Benchmark Results

Nine candidate models were benchmarked under identical conditions: llama-server on localhost, 1024-token context, Q4_K_M quantization, temperature 0.0, one model at a time, full 216-case suite.

### Results Table

| Model | Accuracy | Avg. Latency (s) | Turn-Match Rate |
|-------|----------|-------------------|-----------------|
| Ministral-3-3B-Instruct | 36.11% | 1.589 | 98.61% |
| Ministral-3-8B-Instruct | 60.65% | 2.976 | 100.00% |
| Phi-3.5-mini-instruct | 86.11% | 2.043 | 100.00% |
| Phi-4-mini-instruct | 89.81% | 0.737 | 100.00% |
| Qwen3.5-2B | 90.28% | 0.643 | 100.00% |
| **Qwen3.5-4B** | **96.30%** | 1.156 | 100.00% |
| Qwen3.5-9B | 93.98% | 1.568 | 100.00% |
| **gemma-4-E2B-it** | **95.83%** | **0.734** | 100.00% |
| gemma-4-E4B-it | 95.83% | 1.164 | 100.00% |

### Key Findings

- **60-point accuracy spread** (36%--96%) across models at similar parameter counts (2B--9B), showing instruction-following quality varies independently of size.
- **Turn-match is near-universal** (100% for 8/9 models) -- the routing logic itself is largely solved even for weak models; accuracy gaps concentrate in the Judge's semantic verdict and the Prompter's action selection.
- **Latency is dominated by the Judge call** (47.8%--63.7% of total LLM time per test case), since it receives the full conversation history and produces free-form reasoning.
- **CLIMATE is the clearest differentiator** among strong candidates, due to its implicit-intent phrasing. Qwen3.5-2B drops to 45.8% on CLIMATE alone despite being the second-fastest model overall.
- **MAINTENANCE** is a milder weak spot across the board (peaking at 91.7%), suggesting the domain's vocabulary is generally harder for this model class.

### Deployment Recommendation

A composite scoring formula was used:

```
norm_speed     = 1 - (latency - min_latency) / (max_latency - min_latency)
norm_accuracy  = (accuracy - min_accuracy) / (max_accuracy - min_accuracy)
jetson_score   = 0.6 * norm_speed + 0.4 * norm_accuracy
```

Two scoring versions were computed, differing in whether the Judge call's latency is counted:

| Version | Latency Scope | Rank 1 | Avg. Latency | Score |
|---------|---------------|--------|--------------|-------|
| A | Production only (no Judge) | Qwen3.5-2B | 0.980 s | 0.960 |
| B | Full end-to-end (incl. Judge) | gemma-4-E2B-it | 0.734 s | 0.973 |

**gemma-4-E2B-it** is recommended as the **balanced default**: 95.8% accuracy, 0.734 s average end-to-end latency, the most uniform per-call latency profile of any candidate, and at 2B parameters the smallest memory footprint among models clearing 95% accuracy (~1.5 GB vs ~6.5 GB for 9B-class).

**Qwen3.5-4B** is recommended as the **higher-accuracy alternative**: 96.3% accuracy, 1.16--1.72 s latency depending on scoring version, preferred when maximum accuracy is required.

---

## Target Hardware

The pipeline's model-selection targets a **Jetson-class edge board** (NVIDIA Ampere architecture):
- 2048 CUDA cores, 64 Tensor Cores, up to 275 TOPS
- 12-core Arm CPU at 2.2 GHz
- 64 GB shared LPDDR5 RAM/VRAM
- CUDA 12.6

Compute is shared with the vehicle's other AI workloads, which is why model size and per-call latency are first-class metrics alongside accuracy.

---

## Project Structure

```
conversation-eval-pipeline/
├── src/
│   ├── main.py              # CLI entry point for offline benchmark
│   ├── config.py            # Runtime configuration (model, port, etc.)
│   ├── graph.py             # LangGraph state machine builder
│   ├── nodes.py             # Prompter and Judge node implementations
│   ├── state.py             # BrainState TypedDict definition
│   ├── schemas.py           # Pydantic schemas (PrompterSchema, JudgeSchema)
│   ├── parsers.py           # JSON extraction with 3-fallback parser
│   ├── runner.py            # Test execution loop and metric collection
│   ├── reporter.py          # Summary console output and JSON report export
│   ├── llm_tracker.py       # LangChain callback for LLM latency tracking
│   ├── run.sh               # Automated multi-model benchmark orchestrator
│   └── live/
│       ├── entry.py         # CLI entry point for live wiring test
│       ├── live_graph.py    # Live graph with TTS/ASR nodes + checkpointer
│       ├── live_nodes.py    # Live prompter, voice_cmd, asr, judge nodes
│       ├── live_state.py    # LiveBrainState TypedDict
│       └── benchmark.py     # Live benchmark runner with mock TTS/ASR
├── Analysis/
│   ├── chart_generator.py   # Matplotlib/Seaborn chart generation from reports
│   └── requirements.txt     # Analysis dependencies
├── Json_samples/            # Test suite JSON files (216-case and 50-case)
├── Resources/               # Architecture diagrams (system_design.svg)
├── models.conf              # Model download configuration (HuggingFace repos)
├── models/                  # Local GGUF model storage
├── installModels.sh         # HuggingFace GGUF model downloader
├── setup.sh                 # Virtual environment setup
├── requirements.txt         # Core Python dependencies
└── Rapport_Stage_Omar_Brika.pdf  # Full internship report
```

---

## Prerequisites

- Python 3.12+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) built with `llama-server` available on `$PATH`
- GPU with sufficient VRAM for GGUF inference (or CPU fallback)
- `huggingface-cli` for model downloads (installed automatically by `installModels.sh`)

---

## Setup

```bash
# 1. Create venv and install dependencies
chmod +x setup.sh
./setup.sh

# 2. Download models
chmod +x installModels.sh
./installModels.sh --model-dir ./models --quant Q4_K_M
```

### Expected Model Directory Structure

```
models/
├── reasoning/
│   ├── Qwen3.5-4B-Q4_K_M.gguf
│   ├── Qwen3.5-9B-Q4_K_M.gguf
│   └── ...
└── non_reasoning/
    ├── Phi-4-mini-instruct-Q4_K_M.gguf
    ├── Ministral-3-3B-Instruct-Q4_K_M.gguf
    └── ...
```

Models are configured in `models.conf`. Edit to add/remove models before running `installModels.sh`. Reasoning models (Qwen3.5, gemma-4) are launched with `enable_thinking: false` via `--chat-template-kwargs`.

---

## Usage

### Offline Benchmark (Single Model)

```bash
# Terminal 1: Start llama-server
llama-server -m models/non_reasoning/Phi-4-mini-instruct-Q4_K_M.gguf \
  --port 8000 --host 127.0.0.1 -c 1024 -ngl 99

# Terminal 2: Run benchmark
cd src
python3 main.py \
  --model_name Phi-4-mini-instruct-Q4_K_M.gguf \
  --port 8000 \
  --test_samples ../Json_samples/samples.json \
  --verbose
```

### Automated Multi-Model Benchmark

```bash
cd src
chmod +x run.sh
./run.sh \
  --model-dir ../models \
  --port 8000 \
  --ctx-size 1024 \
  --ngl 99
```

Iterates through all `.gguf` files in `non_reasoning/` then `reasoning/`, launching `llama-server` + `main.py` for each model. Polls `/health` with up to 30 retries (2s each) before starting. A signal trap ensures cleanup on interruption.

### Live Wiring Test (with Mock TTS/ASR)

```bash
cd src/live
python3 entry.py \
  --model_name Qwen3.5-4B-Q4_K_M.gguf \
  --port 8000 \
  --test_samples ../../Json_samples/samples.json \
  --limit 5
```

### Generate Comparative Charts

```bash
cd Analysis
pip install -r requirements.txt
python3 chart_generator.py \
  --input ../benchmark_report_*.json \
  --output_dir charts
```

Generates 4 charts:
1. **Overall Performance** -- Accuracy vs. Turn Match Rate per model
2. **Latency Breakdown** -- Stacked bar of initial prompter / routing / judge times
3. **Domain Accuracy** -- Per-domain accuracy grouped by model
4. **Avg Call Latency** -- Average LLM response time per component

---

## CLI Arguments

### `main.py`

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model_name` | Yes | -- | Model filename (e.g., `Qwen3.5-4B-Q4_K_M.gguf`) |
| `--port` | No | 8000 | llama-server port |
| `--test_samples` | No | `test_cases.json` | Path to JSON test suite |
| `--verbose` | No | false | Print per-turn debug output |

### `run.sh`

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model-dir` | Yes | -- | Root model directory with `reasoning/` and `non_reasoning/` subdirs |
| `--port` | No | 8000 | Server port |
| `--host` | No | 127.0.0.1 | Server host |
| `--ctx-size` | No | 1024 | Context window size |
| `--ngl` | No | 99 | GPU layers to offload |

---

## Metrics

The pipeline tracks per-test-case and aggregate metrics:

| Metric | Description |
|--------|-------------|
| **Verdict Accuracy** | % of tests where model's PASS/FAIL matches expected |
| **Turn Match Rate** | % of tests where actual conversation turns equals expected |
| **LLM Latency (per call type)** | Timing for initial prompter, routing, and judgment calls separately |
| **Call Counts** | Number of LLM invocations per component |

Latency is instrumented via a LangChain `BaseCallbackHandler` (`LLMTimeTracker`) that captures timestamps at `on_chat_model_start` / `on_llm_start` and computes duration at `on_llm_end`, with tags identifying which metric bucket each call belongs to.

Reports are saved as `benchmark_report_YYYYMMDD_HHMMSS.json`.

---

## Security Assessment

As a secondary deliverable, an authorized vulnerability assessment of the target Jetson-class board was performed, following a standard reconnaissance-enumeration-exploitation methodology.

**Findings on OpenSSH 9.6:**

| Finding | CVSS | Description |
|---------|------|-------------|
| CVE-2024-6387 (regreSSHion) | Critical (8.1) | Unauthenticated remote code execution via signal-handler race condition. Confirmed with working PoC. |
| CVE-2025-26466 | High (5.9) | Uncontrolled memory consumption (DoS) during key exchange. Confirmed via version fingerprinting. |

**Remediation:** Upgrade OpenSSH past patched versions and restrict SSH access to a trusted management network. The three additional unauthenticated ports discovered (7777, 50052, 50314/Nagios NSCA) should be restricted to a management VLAN.

---

## Future Work

1. **Two-tier micro-model routing** -- Separate the Router back into its own node so a smaller, faster model can handle simple turn-taking decisions while the larger model is reserved for generation and judging.
2. **Asynchronous shadow judging** -- Decouple the next voice command (needed immediately to avoid DUT microphone timeout) from the Judge's slower semantic evaluation.
3. **Latency targets against real VAD timeouts** -- Validate against live DUT through the existing `live/` graph with real voice-activity-detection constraints.
4. **Telemetry/alignment dashboard** -- Compare Judge verdicts against human-graded transcripts for continuously-updated confidence figures.
5. **Next-command speculative caching** -- Pre-generate likely follow-up commands while waiting on the DUT's transcript to reduce perceived latency.

---

## Dependencies

### Core (`requirements.txt`)

```
langchain-core==1.4.8
langchain-openai==1.3.2
langgraph==1.2.4
pydantic==2.13.4
```

### Analysis (`Analysis/requirements.txt`)

```
seaborn
matplotlib
pandas
```

---

## References

- LangChain AI. *LangGraph Documentation*. https://langchain-ai.github.io/langgraph/
- Georgi Gerganov et al. *llama.cpp: LLM inference in C/C++*. https://github.com/ggml-org/llama.cpp
- Pydantic Contributors. *Pydantic: Data Validation Using Python Type Hints*. https://docs.pydantic.dev/
- Qualys Threat Research Unit. *CVE-2024-6387: regreSSHion*. https://nvd.nist.gov/vuln/detail/CVE-2024-6387
- OpenSSH Project. *CVE-2025-26466: Uncontrolled Resource Consumption*. https://nvd.nist.gov/vuln/detail/CVE-2025-26466
- Unsloth AI. *Qwen3.5 and gemma-4 GGUF repositories*. https://huggingface.co/unsloth

---

## License

This project was developed as part of an internship at Focus Corporation R&D. See the full internship report (`Rapport_Stage_Omar_Brika.pdf`) for details.
