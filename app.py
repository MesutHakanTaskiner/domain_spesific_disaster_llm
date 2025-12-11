import os
import json
from typing import Optional, Any, Dict

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()


app = FastAPI(title="Disaster JSON Labeler")

# -------------------------------------------------------------
# HF TOKEN - PUT YOUR TOKEN HERE
# -------------------------------------------------------------
# WARNING: Do NOT commit this file to any public repo with real token.
HF_TOKEN = os.getenv("HF_API_TOKEN", "").strip()


# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
MODEL_REGISTRY = {
    "Qwen2-7B": {
        "base_model": "Qwen/Qwen2-7B",
        "adapter_path": "fine_tune/Qwen2-7b",
    },
    "llama3.2_3b": {
        "base_model": "meta-llama/Llama-3.2-3B",
        "adapter_path": "fine_tune/llama3.2_3b",
    },
    "Qwen2.5-0.5b-Instruct": {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter_path": "fine_tune/Qwen2.5-0.5b-Instruct",
    },
    "gemma2_disaster": {
        "base_model": "google/gemma-2-2b-it",
        "adapter_path": "fine_tune/gemma2b",  # LoRA klasörünü buraya koy
    },
}


SYSTEM_PROMPT_FILE = "system_prompt.txt"

# Cache for loaded models and tokenizers
LOADED_MODELS: Dict[str, Dict[str, Any]] = {}  # model_id -> {"model": ..., "tokenizer": ..., "device": ...}


# -------------------------------------------------------------
# Request / response models
# -------------------------------------------------------------
class ChatRequest(BaseModel):
    model_id: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9


class ChatResponse(BaseModel):
    raw_text: str
    is_json: bool
    parsed: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# -------------------------------------------------------------
# System prompt loader
# -------------------------------------------------------------
def load_system_prompt() -> str:
    """Load system prompt from SYSTEM_PROMPT_FILE if exists."""
    if not os.path.exists(SYSTEM_PROMPT_FILE):
        return ""
    try:
        with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content
    except Exception:
        return ""


# -------------------------------------------------------------
# Model loader with optional LoRA merge (uses HF_TOKEN)
# -------------------------------------------------------------
def load_model(model_id: str):
    """
    Load (or return cached) model + tokenizer for the given model_id.
    Uses LoRA adapter if adapter_config.json exists in adapter_path.
    HF_TOKEN is passed to Hugging Face gated models (e.g. Llama).
    """
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {model_id}")

    # Return from cache
    if model_id in LOADED_MODELS:
        obj = LOADED_MODELS[model_id]
        return obj["model"], obj["tokenizer"], obj["device"]

    config = MODEL_REGISTRY[model_id]
    base_model_id = config["base_model"]
    adapter_path = config.get("adapter_path") or ""

    # Common kwargs for HF auth
    common_hf_kwargs = {}
    if HF_TOKEN:
        # Transformers >= 4.38: "token" is the recommended arg
        common_hf_kwargs["token"] = HF_TOKEN

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    print(f"[INFO] Loading base model: {base_model_id} on {device}")

    # Load base model
    try:
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch_dtype,
                device_map="cuda",
                **common_hf_kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch_dtype,
                **common_hf_kwargs,
            )
    except Exception as e:
        print(f"[ERROR] Base model loading error: {e}")
        raise HTTPException(status_code=500, detail=f"Base model loading error: {e}")

    # Check for LoRA adapter
    use_lora = False
    if adapter_path and os.path.isdir(adapter_path):
        adapter_config_file = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(adapter_config_file):
            use_lora = True

    if use_lora:
        print(f"[INFO] Loading LoRA adapter from: {adapter_path}")
        try:
            peft_model = PeftModel.from_pretrained(
                model,
                adapter_path,
                **({} if not HF_TOKEN else {"token": HF_TOKEN}),
            )
            model = peft_model.merge_and_unload()
            print("[INFO] LoRA adapter merged successfully.")
        except Exception as e:
            print(f"[ERROR] Adapter error: {e}")
            raise HTTPException(status_code=500, detail=f"Adapter loading/merging error: {e}")
    else:
        if adapter_path:
            print(f"[WARN] No adapter_config.json found in '{adapter_path}'. Using base model only.")
        else:
            print("[INFO] No adapter_path set. Using base model only.")

    # Tokenizer: try LoRA folder first, then base model
    try:
        if adapter_path and os.path.isdir(adapter_path):
            tokenizer = AutoTokenizer.from_pretrained(
                adapter_path,
                **common_hf_kwargs,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id,
                **common_hf_kwargs,
            )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            **common_hf_kwargs,
        )

    LOADED_MODELS[model_id] = {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
    }
    return model, tokenizer, device


# -------------------------------------------------------------
# Helper: extract JSON from text
# -------------------------------------------------------------
def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Try to find a JSON object {...} in the text and parse it.
    Raises ValueError if parsing fails.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object boundaries found in text.")

    json_str = text[start : end + 1]
    return json.loads(json_str)


# -------------------------------------------------------------
# Generation helper (uses chat template if available)
# -------------------------------------------------------------
def generate_answer(
    model_id: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    model, tokenizer, device = load_model(model_id)
    system_prompt = load_system_prompt()

    messages = []
    # Gemma için system rolü kullanma, system + user'ı merge et
    if "gemma" in MODEL_REGISTRY[model_id]["base_model"].lower():
        merged_user = (system_prompt.strip() + "\n\n" + user_prompt.strip()).strip() if system_prompt else user_prompt
        messages.append({"role": "user", "content": merged_user})
    else:
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

    # Build prompt text
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        if system_prompt:
            text = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
        else:
            text = user_prompt

    inputs = tokenizer([text], return_tensors="pt")

    # Move tensors to device
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    else:
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

    # Remove token_type_ids if present (Qwen/LLaMA do not use it)
    inputs.pop("token_type_ids", None)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    # Strip prompt tokens if chat template is used
    if hasattr(tokenizer, "apply_chat_template"):
        new_tokens = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
    else:
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if device == "cuda":
        torch.cuda.empty_cache()

    print(response)

    return response.strip()


# -------------------------------------------------------------
# FastAPI endpoint: /chat
# -------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if request.model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {request.model_id}")

    try:
        raw = generate_answer(
            model_id=request.model_id,
            user_prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

    # Try to parse JSON
    try:
        parsed = extract_json_from_text(raw)
        return ChatResponse(
            raw_text=raw,
            is_json=True,
            parsed=parsed,
            error=None,
        )
    except Exception as e:
        # Return raw text if JSON parsing fails
        return ChatResponse(
            raw_text=raw,
            is_json=False,
            parsed=None,
            error=str(e),
        )


# -------------------------------------------------------------
# Serve frontend from separate folder
# -------------------------------------------------------------
@app.get("/", include_in_schema=False)
def serve_frontend():
    """
    Serve the static frontend HTML file.
    Expected path: ./frontend/index.html
    """
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if not os.path.exists(frontend_path):
        raise HTTPException(status_code=500, detail="Frontend file not found. Expected: frontend/index.html")
    return FileResponse(frontend_path)
