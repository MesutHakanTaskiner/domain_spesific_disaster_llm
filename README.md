# Domain-Specific Disaster LLM

End-to-end pipeline to build, fine-tune, and serve a disaster-focused assistant that turns social posts into structured JSON (needs, urgency, location, etc.). Includes rule-based data filtering, template generation, optional OpenAI correction, Unsloth + QLoRA fine-tuning, and a FastAPI + web UI for inference.

## Contents
- Preprocessing (`preprocess_data/`): filters HumAID/CrisisBench/Kaggle tweets; builds training JSON; optional OpenAI schema correction.
- Fine-tuning (`fine_tune/`): Unsloth QLoRA trainer.
- Inference (`app.py`): FastAPI API with Hugging Face base models + optional LoRA merge; serves `frontend/index.html`.
- Prompts: `system_prompt.txt` defines the output schema and labeling rules.
- Data dirs: `raw_data/` (inputs), `data/` (filtered CSVs), `fine_tune/` & `unsloth-outputs/` (model artifacts), `frontend/` (UI).

## Prerequisites
- Python 3.10+ recommended.
- NVIDIA GPU + CUDA strongly recommended for training/inference; CPU will be slow.
- Hugging Face token (`HF_API_TOKEN`) if using gated models (e.g., Llama family).
- Optional: OpenAI key for schema correction.

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```
Notes:
- The requirements are GPU-heavy (torch, triton, xformers, bitsandbytes, unsloth). Expect large downloads.
- In `fine_tune/finetune_unsloth.py`, `load_in_4bit=False` because Windows 4-bit kernels are unstable.

## Environment variables
Create `.env` (only what you need):
- Inference: `HF_API_TOKEN` (for gated base models).
- OpenAI correction: `OPENAI_API_KEY`, optional `OPENAI_MODEL_NAME` (default `gpt-4.1-mini`), `SYSTEM_PROMPT_PATH`, `INPUT_JSON_PATH`, `OUTPUT_JSON_PATH`.

## Data preparation
Place raw files:
- HumAID TSVs under `raw_data/Humaid_events_set1/`
- CrisisBench TSVs under `raw_data/crisis_data/`
- Kaggle Turkey/Syria earthquake CSV as `raw_data/kaggle_tweets.csv`

Filter likely help-call tweets (tune `MODE` and `ALLOW_RTS_IF_MATCH` at top of scripts):
```bash
python preprocess_data/humaid_data.py          # -> data/humaid_help_calls.csv
python preprocess_data/crisis_data.py          # -> data/crisis_help_calls.csv
python preprocess_data/kaggle_data.py          # -> data/help_tweets_strict.csv
```
Scripts use regex cues for SOS/needs/location, drop offers/announcements, add audit flags (`_sos`, `_need`, `_haz`, etc.).

## Build training JSON
Convert CSV rows into `{"user": "...", "assistant": {...}}` with rule-based labels:
```bash
python preprocess_data/create_template_dataset.py ^
  --input data/humaid_help_calls.csv ^
  --output fine_tune/disaster_dataset_kaggle.jsonl ^
  --train-ratio 1.0 --val-ratio 0.0 --test-ratio 0.0 --pretty
```
Output keys: needs, urgency, life_threatening, location{country, province, district, neighborhood, address_note, geo_confidence}, people_count, vulnerable_groups, post_type, confidence.

## Optional: schema correction via OpenAI
```bash
# set OPENAI_API_KEY in .env
python preprocess_data/fix_dataset_openai.py
```
Reads/writes paths from env (defaults to `system_prompt.txt`, `input.json`, `output.json`) and enforces the schema with `response_format={"type": "json_object"}`.

## Fine-tuning (Unsloth + QLoRA, Gemma-2-2B example)
- Prepare JSONL with `messages` per sample; path in script: `data/disaster_train.jsonl`.
- Run:
```bash
python fine_tune/finetune_unsloth.py
```
Key knobs: `max_seq_length`, `dtype` (bf16/float16), `load_in_4bit` (False on Windows), LoRA rank/targets, `SFTConfig` (epochs, batch size, grad accum, LR, save steps). Outputs: adapter in `gemma2b_disaster_lora_only/` and checkpoints in `gemma2b-unsloth-disaster-finetune/`.

Use your own adapter in the API/UI:
1) Note the adapter folder you saved (e.g., `gemma2b_disaster_lora_only/` or a run under `fine_tune/`).
2) Add a new entry to `MODEL_REGISTRY` in `app.py` with `base_model` (the original HF model) and `adapter_path` (your saved adapter dir that contains `adapter_config.json`).
3) Start the server (see below) and call `/chat` with your new `model_id`.

## Inference service (FastAPI + UI)
`app.py` serves the API and static frontend.

Model registry (`MODEL_REGISTRY` in `app.py`): map `model_id` -> `base_model` + optional `adapter_path` (must contain `adapter_config.json` to merge). Defaults include `gemma2_disaster` (google/gemma-2-2b-it), `Qwen2-7B`, `llama3.2_3b`, `Qwen2.5-0.5b-Instruct`.

Start server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
- Uses `system_prompt.txt`. For Gemma, system text is merged into the user turn; others get a system role.
- If `HF_API_TOKEN` is set, it is passed to `from_pretrained`.

Frontend:
- Open `http://localhost:8000/`.
- Pick model, paste tweet, set `max_tokens`/`temperature`/`top_p`, click Send.
- UI shows parsed fields + raw JSON and a badge for JSON validity.

API example:
```bash
curl -X POST http://localhost:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"model_id\":\"gemma2_disaster\",\"prompt\":\"...\",\"max_tokens\":512,\"temperature\":0.1,\"top_p\":0.9}"
```
Response: `{`"raw_text"`: "...", "is_json": true|false, "parsed": {...}|null, "error": "..."}`
Server extracts the first `{...}` block; keep temperature low to reduce spillover text.

## Prompt/schema (runtime)
`system_prompt.txt` demands a single JSON object with:
- needs (list), urgency (low|normal|high|critical), life_threatening (bool),
- location {country, province, district, neighborhood, address_note, geo_confidence},
- people_count (int|null), vulnerable_groups (list), post_type (help_request|information|emotion/prayer|complaint), confidence (0-1).
Use `null` when unknown.

## Troubleshooting
- OOM during fine-tune: lower `per_device_train_batch_size`, raise `gradient_accumulation_steps`, reduce `max_seq_length`.
- Adapter not loaded: ensure `adapter_config.json` exists in `adapter_path`; otherwise base model is used with a warning.
- Slow/CPU inference: set smaller base models in `MODEL_REGISTRY`.
- JSON parse errors: tighten prompts/temperature and ensure the model outputs only one JSON object.

## File map (quick)
- `preprocess_data/`: `humaid_data.py`, `crisis_data.py`, `kaggle_data.py`, `create_template_dataset.py`, `fix_dataset_openai.py`.
- `fine_tune/`: `finetune_unsloth.py`, `disaster_dataset_kaggle.jsonl` (example).
- `frontend/index.html`: web UI served at `/`.
- `system_prompt.txt`: schema and rules.
- `app.py`: FastAPI server + model loader.
