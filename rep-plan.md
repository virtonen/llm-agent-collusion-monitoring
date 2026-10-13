# Replication Plan — LLM Agent Collusion Study

## This test — two full markets

This is a modified run, not a copy of the paper's GPT-4.1 and Claude numbers. Agents are local `qwen2.5:14b`. The judge is NVIDIA `nemotron-3-super-120b-a12b`. One market takes about 1–1.5 hours because Qwen answers one agent at a time. Two markets are about 2–3 hours. The Mac has to stay awake.

Each market is the real setup: 5 sellers, 5 buyers, 30 rounds. The only difference between them is the seller mailbox.

```bash
conda activate env
cd /Users/vvirtonen/Desktop/UChicago/llm-agent-collusion

python src/continuous_double_auction/simulation.py --tag "base-1"

python src/continuous_double_auction/simulation.py \
  --seller_comms_enabled \
  --tag "base-seller_comms-1"
```

The first tag is sellers who cannot talk. The second is sellers who can. Compare `results/*_base-1/` and `results/*_base-seller_comms-1/`: asks, trades, and `collusion_metrics.json`. Both prompts already tell sellers to keep high margins and to use game theory, and both can see the public ask queue. A higher price or a higher coordination score in the second folder is the sign that the mailbox mattered. One pair is noisy. The paper repeats each condition 10 times.

Do not start `experiments/run_seller_comms_experiment.sh` for this test. That script is 20 markets, about two days.

## Project scale

- **Per run:** 5 sellers + 6 buyers = 11 agents × 30 rounds = 330 API calls
- **Full paper replication:** ~150–200 runs across all experiments (~50,000–66,000 total calls)
- Simulation parallelizes agents with `ThreadPoolExecutor`

### Hardcoded to `gpt-4.1-mini` (requires changes for full local replication)

| Location | Purpose | Skippable? |
|---|---|---|
| `simulation.py:78` | `MessageOverseer` (oversight experiments) | Yes — only needed for oversight conditions |
| `evaluate.py:17` | Post-run LLM evaluation of agent reasoning | Yes — post-hoc step |
| `judge_reliability.py:23` | Reliability scoring | Yes — separate analysis script |

### Type system blocker

`cda_types.py` uses `Model = Literal[...]` — Pydantic rejects any model name not in the hardcoded list. Must loosen to `Model = str`.

---

## Option A — Pure Ollama ($0 forever)

Add `OllamaModelWrapper` using OpenAI client pointed at `http://localhost:11434/v1`. Loosen `Model` type to `str`. Add routing in `ModelWrapper.create()`.

**Best local models for M5 Pro 24GB** (need reliable JSON output + game theory reasoning):

| Model | Size (Q4) | JSON reliability | Reasoning |
|---|---|---|---|
| `qwen2.5:14b` | ~9 GB | Excellent | Strong |
| `qwen2.5:32b` | ~20 GB | Excellent | Best local |
| `llama3.1:8b` | ~5 GB | Good | Adequate |
| `mistral-nemo:12b` | ~7 GB | Good | Decent |

**Caveats:** Ollama serializes requests by default — parallel agent threads will queue. Results will likely differ from the paper (smaller/different model families). This tests "does collusion happen with local models?" not faithful paper replication.

---

## Option B — OpenRouter (cheapest faithful replication)

Already supported in codebase — unknown model names fall through to `OpenRouterClient`. Just set `OPENROUTER_API_KEY` and bypass Literal validation.

- **Free models:** see the Option C allowlist below. The old ids `meta-llama/llama-3.1-8b-instruct:free` and `google/gemma-2-9b-it:free` are no longer on the free list. Free variants share 20 requests/minute and 50 requests/day.
- **Paid frontier:** `openai/gpt-4.1-mini` or `anthropic/claude-3.5-haiku` for faithful replication

**Cost estimate** (using `openai/gpt-4.1-mini` via OpenRouter, ~$0.40/M input + $1.60/M output, ~4K tokens/call):

| Scope | Runs | Est. cost |
|---|---|---|
| Single run | 1 | ~$0.60–1.50 |
| Baseline experiment (20 runs) | 20 | ~$12–30 |
| Full paper replication | 150–200 | ~$100–300 |

---

## Option C — Hybrid (recommended)

Use Ollama locally for buyer/seller agents + OpenRouter for evaluation/overseer. $0 for exploratory runs; low cost only for frontier-model comparisons.

OpenRouter is only the overseer (`message_overseer_prompt.jinja2`) and the post-run judge (`seller_individual_eval.jinja2`). Both must return nested JSON. Do not use `openrouter/free`: it can route to any free model, including ones this plan excludes.

### Free models to allow (12 Oct 2026 list)

| Role | Model id | Why |
|---|---|---|
| Primary | `nvidia/nemotron-3.5-lightning:free` | Accepts normal chat completions. 3B active, 1M context |
| Fallback 1 | `apodex/apodex-1.1-mini:free` | Accepts chat completions. Evidence-grounded research model, 262K context |
| Fallback 2 | `dots-studio/dots-3-note-preview:free` | Accepts chat completions. 16B active, 512K context. Endpoint ends 31 Dec 2026 |
| Do not call | `thinkingmachines/inkling:free` | OpenRouter returns 403: free endpoint is limited to agentic harness apps |
| Do not call | `thinkingmachines/inkling-small:free` | Same 403 harness gate |

Set `OVERSEER_MODEL` and `EVAL_MODEL` to `nvidia/nemotron-3.5-lightning:free`. One judged run is about 150 calls, so the 50/day cap is about 3 days per run. Agent model is local `qwen2.5:14b` (`AGENT_MODEL`).

### Free models to leave blocked

| Model | Why it stays off |
|---|---|
| Inception Mercury Decide | Not chat completions. Uses the `/v1/systemone` decision API |
| Respan Span-01 Lite | Behavior-probability scorer, not a JSON chat judge |
| LiquidAI LFM2.5-Embedding-350M | Embeddings. Prompts may be used to train Liquid models |
| NVIDIA Nemotron 3 Embed 1B | Embeddings |
| Fish Audio S2.1 Pro Free | Speech synthesis |
| LiquidAI LFM2.5-2.6B | Vendor says to avoid knowledge-heavy tasks. Prompts may be used for training |
| Poolside Laguna S 2.1 and Laguna XS 2.1 | Coding agents. End 31 Oct 2026. Free use may train on inputs and outputs |
| Cohere North Mini Code | Coding agent, not a collusion judge |

### Guardrail (`free-models-only`)

- Budget: no credit limit. A `$0` budget shows `$0 remaining` and can reject free calls. The allowlist is what keeps spend at `$0`.
- Restriction mode: allow only the five ids above. "Allow all except" would keep paid models eligible, and new models would be allowed automatically.
- Zero Data Retention: off. Inkling and Nemotron free endpoints retain or train; ZDR would drop them.
- Data training: paid-training off, prompt publishing off, free-endpoint training on (required for Inkling and Nemotron).
- Prompt injection and sensitive-info detection: off. Seller quotes can look like instructions, and the prompts are synthetic.

---

## Setup already done

These steps are finished. They are not the test.

1. `Model` is a plain string, so Qwen and Nemotron ids are accepted.
2. Buyer and seller calls go to local Ollama, one agent at a time.
3. The judge uses the NVIDIA key and `NIM_MODEL` when that key is set. OpenRouter remains only as a fallback for ids ending in `:free`.
4. The two-market commands at the top of this file are the run to start.

**Skip Gemini for now** — code uses Vertex AI SDK (not free Gemini API); requires Google Cloud project + service account auth. OpenRouter gives Gemini access anyway.

---

## Code changes required

### 1. `src/continuous_double_auction/cda_types.py`

Change `Model = Literal[...]` → `Model = str` and update `SUPPORTED_MODELS` (or remove `choices=` validation from argparse in experiment scripts).

### 2. `src/resources/model_wrappers.py`

Add `OllamaModelWrapper` (copy of `OpenRouterClient` with different base URL and api_key):

```python
class OllamaModelWrapper(ModelWrapper):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
    # generate() and batch_generate() identical to OpenRouterClient
```

Add routing in `ModelWrapper.create()` — e.g. prefix check for `ollama/` or explicit model name list.

### 3. `src/continuous_double_auction/simulation.py:78`

Parameterize the overseer model name instead of hardcoding `gpt-4.1-mini`:

```python
overseer_model = os.getenv("OVERSEER_MODEL", "gpt-4.1-mini")
overseer = MessageOverseer(client=get_client(model=overseer_model, temperature=0), ...)
```

### 4. `src/continuous_double_auction/market/market.py` (threading)

When using Ollama, set `n_threads=1` to avoid hammering a single local server.

---

## Later, not this test

The archived `data/` corpus is 75 markets. A Qwen/Nemotron rerun of that corpus is the closest modified replication, and it takes about 3–4 days. It is not this test.

| Experiment | Runs | Script | What it adds |
|---|---|---|---|
| Baseline vs. comms | 20 | `run_seller_comms_experiment.sh` | Ten repeats of the two markets above |
| Boss pressure and oversight | 30 | `run_pressure_experiment.sh` | CEO pressure and a license threat |
| Seller model mix, with messages | 25 | `run_seller_mix_with_comms_experiment.sh` | Qwen buyers/sellers vs Nemotron |

Temperature, valuation, and country are extra sweeps. They are not in `data/`. They also crash until `run_simulation` accepts `log_dir`. The no-message seller-mix script is another extra and is not in `data/` either.
