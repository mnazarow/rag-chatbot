"""QLoRA-дообучение базовой модели на собранном датасете (GPU).

Обучает LoRA-адаптер поверх базовой модели (4-bit) на finetune/data/train.jsonl
и сохраняет адаптер в finetune/adapter. Адаптер затем подаётся через vLLM
(--enable-lora), без слияния весов — это быстро применять и откатывать.

Запуск:  python finetune/train_lora.py [--base HF_MODEL] [--epochs 3]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import settings  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "train.jsonl"
ADAPTER_DIR = Path(__file__).resolve().parent / "adapter"


def _default_base() -> str:
    """Базовая (не-квантованная) HF-модель для обучения.
    Берём VLLM_MODEL, но убираем суффикс -AWQ/-GPTQ — обучать нужно fp16-базу."""
    m = settings.get("VLLM_MODEL") or "Qwen/Qwen2.5-7B-Instruct"
    return m.replace("-AWQ", "").replace("-GPTQ", "").replace("-Int4", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=_default_base(), help="HF id базовой модели (fp16)")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()

    if not DATA.exists():
        raise SystemExit(f"Нет датасета {DATA}. Сначала: python finetune/build_dataset.py")

    import torch
    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    print(f"Базовая модель: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="auto",
        trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=str(DATA), split="train")

    def fmt(ex):
        return {"text": tok.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)}
    ds = ds.map(fmt, remove_columns=ds.column_names)

    cfg = SFTConfig(
        output_dir=str(ADAPTER_DIR / "_run"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_strategy="epoch",
        bf16=True, max_seq_length=args.max_len, packing=False,
        dataset_text_field="text")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds)
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    tok.save_pretrained(str(ADAPTER_DIR))
    print(f"Адаптер сохранён: {ADAPTER_DIR}")
    print("Подайте его через vLLM: кнопка «Применить дообученную модель» в админке.")


if __name__ == "__main__":
    main()
