# Diagnostic Decision Support Agent

A focused clinician-facing prototype for **synthetic/de-identified cases**. It turns a pasted clinical note into structured evidence, retrieves reference passages **before** reasoning, generates a ranked differential, runs an independent blind challenge, revises the differential, recommends next diagnostic information-gathering steps, applies deterministic safety checks, and records explicit human review.

## LLM inference

This project has one LLM path only. It uses the MAQ IndiaAI chat-completions gateway configured in `config/llm.json`:

- model: `qwen-3.8-27b`
- endpoint: `https://indiaai.maqsoftware.net/v1/chat/completions`
- max input tokens: `122880`
- max output tokens: `8192`

There is **no Groq integration, no local model, no rule-based reasoning fallback, and no demo-response fallback**. If the gateway fails or returns invalid structured output, the request fails visibly.

If the gateway requires a bearer token, set `INDIAAI_API_KEY` in `.env`. If your network handles gateway authentication separately, leave it blank.

## Workflow

```text
Pasted synthetic/de-identified note
        ↓
Qwen intake extraction
        ↓
Pydantic validation
        ↓
Code-derived lab abnormalities + timeline
        ↓
Qwen clinical problem representation
        ↓
Keyword retrieval from curated corpus
        ↓
Qwen rerank of retrieved candidates
        ↓
Qwen ranked differential with passage IDs
        ↓
Qwen blind challenge (primary answer hidden)
        ↓
Code comparison of the two candidate lists
        ↓
Qwen revision using disagreements
        ↓
Qwen next-best diagnostic steps
        ↓
Code safety gate
pass | pass_with_warnings | escalate | block
        ↓
Explicit human review + audit record
```

## What was intentionally removed

The repository does not include prepared patient demos, Synthea demo data, CSV lab pages, numbered phase endpoints, deterministic diagnosis rules, provider switching, or silent fallbacks. The UI exposes one primary workflow only.

Business data is kept outside Python code:

- `config/llm.json` — model/gateway configuration
- `config/safety_rules.json` — configurable red-flag thresholds
- `data/clinical_corpus.json` — curated reference corpus used for retrieve-first evidence

No API key or patient record is committed.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

The UI requires confirmation that the pasted content is synthetic or de-identified.

## API

- `POST /api/analyze/stream` — primary UI path; streams stage and Qwen activity as SSE, then emits the final validated result
- `POST /api/analyze` — non-streaming API path
- `GET /api/runs/{run_id}` — retrieve the persisted exact run output
- `POST /api/runs/{run_id}/review` — record clinician accept/reject review
- `GET /health` — service and model configuration check
- `GET /docs` — OpenAPI documentation

Example request:

```json
{
  "note": "<synthetic/de-identified clinical note>",
  "patient_reference": "optional-reference",
  "confirm_synthetic": true
}
```

## Safety boundary

This is a decision-support prototype, not a diagnostic device. It must not make autonomous clinical decisions or provide treatment orders. The safety gate verifies citation integrity, numeric grounding, treatment/dosing language, and configured red flags before human review.

## Tests

```bash
pytest -q
```


## Live progress

The browser uses the streaming analysis endpoint. Each run emits:

```text
stage_start
llm_request
llm_connected
llm_delta
llm_complete
stage_complete
...
final
```

`llm_delta` is emitted only when the IndiaAI gateway actually sends streamed Qwen output. Structured JSON is accumulated server-side and validated only after the stream completes. There is no alternate model or inference fallback.


## GitHub Actions validation

The repository includes `.github/workflows/ci.yml`.

Every push validates dependencies, Python compilation, unit/integration tests, application startup, `/health`, and the clinician UI.

A real MAQ IndiaAI end-to-end job is also available. It uses the repository secret `INDIAAI_API_KEY` or `INDIAAI` without printing the secret value. The live job runs on manual workflow dispatch or on a main-branch commit whose message contains `[live]`.

The live validation exercises the configured Qwen model through the full synthetic workflow:

```text
IndiaAI probe
→ Intake
→ Clinical synthesis
→ Evidence retrieval/reranking
→ Initial differential
→ Blind challenge
→ Differential revision
→ Next-best diagnostic steps
→ Safety gate
→ persistence
```
