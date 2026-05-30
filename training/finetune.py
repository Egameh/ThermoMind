"""
training/finetune.py

QLoRA fine-tuning for thermodynamics domain adaptation.

Supports: Llama-3.2-3B-Instruct and Mistral-7B-Instruct-v0.2

Key design decisions:
    - 4-bit NF4 quantisation: fits large models in Colab GPU memory
    - LoRA targeting attention + MLP layers: MLP stores factual domain knowledge
    - Rank r=16, alpha=32: sufficient capacity for domain adaptation
    - 2 epochs: prevents memorisation on small dataset
    - Gradient accumulation x32: effective batch size 32 on single GPU

Usage:
    # Llama 3B
    python training/finetune.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --dataset data/thermo_dataset.jsonl \
        --output ./outputs/llama-thermo \
        --hf_token YOUR_TOKEN

    # Mistral 7B
    python training/finetune.py \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --dataset data/thermo_dataset.jsonl \
        --output ./outputs/mistral-thermo \
        --hf_token YOUR_TOKEN
"""

import os
import json
import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

MAX_LENGTH = 1536  # Hardware constraint: Colab GPU memory limit
# Higher token lengths may improve performance (Along side a mode diverse dataset.)

def load_quantized_model(model_id: str, hf_token: str):
    """Load model in 4-bit NF4 quantisation."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )
    return model, tokenizer


def apply_lora(model):
    """
    Apply LoRA adapters to attention and MLP layers.

    Targeting MLP layers (gate_proj, up_proj, down_proj) in addition to
    attention projections is critical for domain knowledge transfer.
    MLP layers store factual associations; attention layers handle context.
    """
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention
            "gate_proj", "up_proj", "down_proj",       # MLP
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def format_dataset(dataset, tokenizer):
    """Apply chat template to instruction/output pairs."""
    def _format(examples):
        texts = []
        for instr, out in zip(examples["instruction"], examples["output"]):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a senior Thermodynamics Engineer. "
                        "Provide clear, accurate answers. Use step-by-step "
                        "derivations with LaTeX when the question requires "
                        "mathematical detail."
                    ),
                },
                {"role": "user", "content": instr},
                {"role": "assistant", "content": out},
            ]
            texts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
        return {"text": texts}

    return dataset.map(_format, batched=True)


def train(model_id: str, dataset_path: str, output_dir: str, hf_token: str):
    """Run fine-tuning pipeline."""
    model, tokenizer = load_quantized_model(model_id, hf_token)
    model = apply_lora(model)

    print(f"\n📂 Loading dataset: {dataset_path}")
    raw = load_dataset("json", data_files=dataset_path, split="train")
    print(f"   Entries: {len(raw)}")

    formatted = format_dataset(raw, tokenizer)
    split = formatted.train_test_split(test_size=0.1, seed=42)
    train_data = split["train"]
    val_data = split["test"]
    print(f"   Train: {len(train_data)} | Val: {len(val_data)}")

    sft_config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_length=MAX_LENGTH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,   # effective batch = 32
        learning_rate=5e-5,
        num_train_epochs=2,
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_steps=100,
        warmup_ratio=0.1,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=sft_config,
    )

    print(f"\n Training | Model: {model_id}\n")
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/training_log.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ Model and training log saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset", type=str, default="data/thermo_dataset.jsonl")
    parser.add_argument("--output", type=str, default="./outputs/thermo-model")
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Provide --hf_token or set HF_TOKEN")

    train(args.model, args.dataset, args.output, hf_token)
