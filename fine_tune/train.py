import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig


def main():
    # ---- Load env and login to Hugging Face ----
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if hf_token is None or hf_token.strip() == "":
        raise RuntimeError("HF_TOKEN is not set in .env file")

    # Non-interactive login
    login(token=hf_token)

    # ---- Paths ----
    project_root = Path(__file__).resolve().parent.parent
    jsonl_path = project_root / "sample_data" / "disaster_train.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found at: {jsonl_path}")

    # ---- Config ----
    MODEL_NAME = "meta-llama/Llama-3.2-3B"
    MAX_SEQ_LENGTH = 1024

    # You had BF16 originally; keep it. Change to float16 if you need.
    DTYPE = torch.bfloat16

    # ---- Load dataset ----
    dataset = load_dataset(
        "json",
        data_files={"train": str(jsonl_path)},
    )["train"]

    print("Sample raw example from dataset:")
    print(dataset[0])

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Important for causal LM padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def messages_to_text(examples):
        """Merge system+user into one user turn, keep assistant as answer."""
        texts = []
        for conv in examples["messages"]:
            system_msg = ""
            user_msg = ""
            assistant_msg = ""

            # Extract first system, first user, first assistant
            for m in conv:
                role = m.get("role", "")
                content = m.get("content", "")
                if role == "system" and system_msg == "":
                    system_msg = content or ""
                elif role == "user" and user_msg == "":
                    user_msg = content or ""
                elif role == "assistant" and assistant_msg == "":
                    assistant_msg = content or ""

            # Merge system + user into a single user message
            merged_user = (system_msg.strip() + "\n\n" + user_msg.strip()).strip()

            # Build chat WITHOUT any "system" role
            chat = [
                {"role": "user", "content": merged_user},
                {"role": "assistant", "content": assistant_msg.strip()},
            ]

            # Convert to plain text using model's chat template
            text = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False,  # include assistant answer in training
            )
            texts.append(text)

        return {"text": texts}

    dataset_text = dataset.map(
        messages_to_text,
        batched=True,
        remove_columns=dataset.column_names,  # only keep "text"
    )

    print("Sample processed text:")
    print(dataset_text[0]["text"][:500])
    print("Type:", type(dataset_text[0]["text"]))  # should be <class 'str'>

    # ---- 4-bit quantization config for QLoRA ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=DTYPE,
        bnb_4bit_quant_type="nf4",
    )

    # ---- Load base model in 4-bit ----
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # ---- Prepare model for k-bit training (QLoRA) ----
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Training config ----
    training_args = SFTConfig(
        output_dir=str(project_root / "llama3.2_3b"),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        bf16=False,   # you set this to False
        fp16=True,    # and this to True
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        report_to="none",
    )

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_text,
        dataset_text_field="text",
        args=training_args,
    )

    trainer.train()

    # ---- Save LoRA adapter ----
    OUTPUT_LORA_DIR = project_root / "Llama-3.2-3B_lora"
    trainer.save_model(str(OUTPUT_LORA_DIR))
    tokenizer.save_pretrained(str(OUTPUT_LORA_DIR))

    print(f"LoRA adapter and tokenizer saved to: {OUTPUT_LORA_DIR}")

    # ---- Merge LoRA into base model ----
    print("Loading base model in full precision for merging...")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
    )

    lora_model = PeftModel.from_pretrained(base_model, str(OUTPUT_LORA_DIR))
    merged_model = lora_model.merge_and_unload()

    MERGED_DIR = project_root / "llama3.2_3b_disaster_merged_bf16"
    merged_model.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))

    print(f"Merged model saved to: {MERGED_DIR}")


if __name__ == "__main__":
    main()
