#!/usr/bin/env python
import os
import json
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

MODEL_NAME = "gpt-4.1-mini"
NUM_SAMPLES = 500
OUTPUT_PATH = "synthetic_disasters.jsonl"

SYSTEM_PROMPT_PATH = Path("system_prompt.txt")
GENERATOR_PROMPT_PATH = Path("generator_prompt.txt")

MAX_TOTAL_CALLS = NUM_SAMPLES * 3  # safety cap so we don't loop forever
SLEEP_SECONDS_ON_ERROR = 5

# load from .env (not strictly needed if OpenAI() will read from env directly,
# but harmless)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", MODEL_NAME)

client = OpenAI()  # uses OPENAI_API_KEY from env / .env


def load_prompt(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def build_generator_instructions(system_prompt: str, template: str) -> str:
    return template.replace("{{SYSTEM_PROMPT}}", system_prompt)


def generate_one_example(generator_instructions: str) -> Optional[dict]:
    try:
        response = client.responses.create(
            model=MODEL_NAME,
            instructions=generator_instructions,
            input="Generate one new training example now.",
            temperature=1.0,
            # IMPORTANT: no `text={"format": ...}` here, it caused your 400 error
        )
    except Exception as e:
        print(f"[ERROR] API call failed: {e}")
        return None

    raw = response.output_text
    if not raw:
        print("[ERROR] Empty response.output_text")
        return None

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Top-level JSON decode failed: {e}")
        print("Raw (truncated):", raw[:300])
        return None

    if not isinstance(obj, dict):
        print("[ERROR] Top-level result is not a dict")
        return None

    messages = obj.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        print("[ERROR] 'messages' must be a list of length 3")
        return None

    return obj


def main() -> None:
    system_prompt = load_prompt(SYSTEM_PROMPT_PATH)
    generator_template = load_prompt(GENERATOR_PROMPT_PATH)
    generator_instructions = build_generator_instructions(system_prompt, generator_template)

    print(f"Model: {MODEL_NAME}")
    print(f"Target samples: {NUM_SAMPLES}")
    print(f"Output file: {OUTPUT_PATH}")

    generated = 0
    total_calls = 0
    seen_user_posts = set()

    # Optional: start with a fresh file each run
    # If you prefer to append across runs, remove this if-block.
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        while generated < NUM_SAMPLES and total_calls < MAX_TOTAL_CALLS:
            total_calls += 1
            sample = generate_one_example(generator_instructions)

            if sample is None:
                print(f"[WARN] Failed generation (call {total_calls}), retrying after sleep.")
                time.sleep(SLEEP_SECONDS_ON_ERROR)
                continue

            msgs = sample.get("messages", [])
            if len(msgs) != 3 or not isinstance(msgs, list):
                print("[WARN] Bad 'messages' shape, skipping.")
                continue

            user_msg = msgs[1]
            user_content = user_msg.get("content") if isinstance(user_msg, dict) else None
            if not isinstance(user_content, str):
                print("[WARN] No user content string, skipping.")
                continue

            # enforce uniqueness of the synthetic tweet
            if user_content in seen_user_posts:
                print("[INFO] Duplicate user post detected, skipping.")
                continue

            seen_user_posts.add(user_content)

            # write ONE example per line to JSONL
            line = json.dumps(sample, ensure_ascii=False)
            out_f.write(line + "\n")

            # flush & sync so you can see it immediately in the file
            out_f.flush()
            os.fsync(out_f.fileno())

            generated += 1
            print(f"[OK] {generated}/{NUM_SAMPLES} (calls used: {total_calls})")

    print(f"Done. Generated {generated} unique samples.")


if __name__ == "__main__":
    main()
