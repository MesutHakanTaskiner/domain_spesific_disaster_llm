import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig

os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # force GPU 0

JSONL_PATH = "data/disaster_train.jsonl"
max_seq_length = 1024         # keep it smaller at first for safety
dtype = "bfloat16"            # or "float16" if bf16 not supported
load_in_4bit = False          # IMPORTANT: no 4bit on Windows for now
model_name = "unsloth/gemma-2-2b-it-bnb-4bit"  # still ok; we load it in fp16/bf16


def build_dataset(jsonl_path: str):
    dataset = load_dataset(
        "json",
        data_files={"train": jsonl_path},
        split="train",
    )

    def messages_to_gemma2_text(examples):
        texts = []
        for conv in examples["messages"]:
            system_msg = ""
            user_msg = ""
            assistant_msg = ""

            for m in conv:
                role = m.get("role", "")
                content = m.get("content", "")
                if role == "system" and system_msg == "":
                    system_msg = content
                elif role == "user" and user_msg == "":
                    user_msg = content
                elif role == "assistant" and assistant_msg == "":
                    assistant_msg = content

            merged_user = (system_msg.strip() + "\n\n" + user_msg.strip()).strip()

            text = (
                "<start_of_turn>user\n"
                + merged_user
                + "\n<end_of_turn>\n"
                "<start_of_turn>model\n"
                + assistant_msg.strip()
                + "\n<end_of_turn>\n"
            )
            texts.append(text)

        return {"text": texts}

    dataset = dataset.map(
        messages_to_gemma2_text,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return dataset


def main():
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("current device:", torch.cuda.current_device() if torch.cuda.is_available() else "no cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    # -------- 1) Load model on CPU, then push to GPU --------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_name,
        max_seq_length = max_seq_length,
        dtype          = dtype,
        load_in_4bit   = load_in_4bit,
        device_map     = None,     # <- no auto device map, we control it
    )

    # move base model to GPU
    model.to("cuda")
    print("After base load, param device:", next(model.parameters()).device)

    # -------- 2) Add LoRA, then push again to be safe --------
    model = FastLanguageModel.get_peft_model(
        model,
        r                           = 16,
        lora_alpha                  = 16,
        lora_dropout                = 0.05,
        target_modules              = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias                        = "none",
        use_gradient_checkpointing  = "unsloth",
        random_state                = 3407,
        use_rslora                  = False,
        loftq_config                = None,
    )

    model.to("cuda")
    print("After LoRA, param device:", next(model.parameters()).device)

    # -------- 3) Build dataset --------
    dataset = build_dataset(JSONL_PATH)
    print("Example text snippet:\n", dataset[0]["text"][:300])

    # -------- 4) Trainer config --------
    training_args = SFTConfig(
        output_dir                      = "gemma2b-unsloth-disaster-finetune",
        num_train_epochs                = 1,
        per_device_train_batch_size     = 1,
        gradient_accumulation_steps     = 4,
        learning_rate                   = 2e-4,
        lr_scheduler_type               = "cosine",
        warmup_ratio                    = 0.03,
        logging_steps                   = 10,
        save_steps                      = 200,
        save_total_limit                = 2,
        bf16                            = True,
        fp16                            = False,
        max_seq_length                  = max_seq_length,
        packing                         = False,
        report_to                       = "none",
        dataset_num_proc                = 1,
    )

    trainer = SFTTrainer(
        model               = model,
        tokenizer           = tokenizer,
        train_dataset       = dataset,
        dataset_text_field  = "text",
        args                = training_args,
    )

    # just to be sure Trainer got GPU
    print("Trainer device:", trainer.args.device)

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part    = "<start_of_turn>model\n",
    )

    trainer.train()

    model.save_pretrained("gemma2b_disaster_lora_only")
    tokenizer.save_pretrained("gemma2b_disaster_lora_only")

    print("Training finished.")


if __name__ == "__main__":
    main()
