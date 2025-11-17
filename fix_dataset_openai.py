import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-mini")
SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "system_prompt.txt")
INPUT_PATH = os.getenv("INPUT_JSON_PATH", "input.json")
OUTPUT_PATH = os.getenv("OUTPUT_JSON_PATH", "output.json")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=API_KEY)

with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()


def process_record(record: dict) -> dict:
    user_content = json.dumps(record, ensure_ascii=False)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)
    return parsed


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for idx, record in enumerate(data):
        print(f"Processing record {idx + 1}/{len(data)}")
        updated = process_record(record)
        results.append(updated)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(results)} records. Output written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
