"""Optional single-GPU QLoRA seam for conversational teacher-trace JSONL.

Each JSONL record must contain a Hugging Face ``messages`` list.  This adapter
trains; the chapter's four-set harness remains the release authority.
"""

from __future__ import annotations

import argparse


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("eval_file")
    parser.add_argument("output_dir")
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit("Install transformers, datasets, peft, trl, accelerate, and bitsandbytes.") from exc

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    dataset = load_dataset("json", data_files={"train": args.train_file, "eval": args.eval_file})
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        bf16=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        max_length=2048,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
