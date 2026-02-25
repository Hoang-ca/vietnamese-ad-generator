"""
Data preprocessing: CSV → ChatML-formatted HuggingFace Dataset.

Usage:
    python src/preprocess.py --input_dir data/ --output_dir data/tokenized/
"""

import argparse
import os

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

from config import MODEL_NAME, MAX_SEQ_LENGTH


def load_and_clean_csv(csv_path: str, split_name: str) -> pd.DataFrame:
    """Read CSV, deduplicate, and validate required columns."""
    df = pd.read_csv(csv_path)
    print(f"[{split_name}] Loaded {len(df)} rows")

    if "id" in df.columns:
        df.drop_duplicates(subset=["id"], keep="first", inplace=True)
        print(f"[{split_name}] After dedup: {len(df)} rows")

    if "advertisement" not in df.columns:
        raise ValueError(f"Column 'advertisement' not found in {csv_path}")
    return df


def to_chatml(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Convert product rows to ChatML prompt/completion pairs."""
    pn_col = "product_name" if "product_name" in df.columns else None
    desc_col = (
        "cleaned_description"
        if "cleaned_description" in df.columns
        else ("description" if "description" in df.columns else None)
    )

    records = []
    for _, row in df.iterrows():
        name = str(row.get(pn_col, "")).strip() if pn_col else ""
        desc = str(row.get(desc_col, "")).strip() if desc_col else ""
        adv = str(row["advertisement"]).strip()
        if not adv:
            continue

        user_content = f"tạo quảng cáo cho sản phẩm sau:\nTên sản phẩm: {name}\nMô tả: {desc}"
        prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
        full_text = f"{prompt}{adv}<|im_end|>"
        records.append({"prompt": prompt, "completion": adv, "full_text": full_text})

    df_out = pd.DataFrame(records)
    print(f"[{split_name}] Processed: {len(df_out)} valid samples")
    return df_out


def tokenize_with_label_masking(examples, tokenizer, max_length):
    """Tokenize with label masking: -100 for prompt tokens."""
    full_enc = tokenizer(
        examples["full_text"],
        max_length=max_length,
        truncation=True,
        padding=False,
        add_special_tokens=False,
    )
    prompt_enc = tokenizer(
        examples["prompt"],
        max_length=max_length,
        truncation=True,
        padding=False,
        add_special_tokens=False,
    )

    all_labels = []
    for i in range(len(full_enc["input_ids"])):
        ids = full_enc["input_ids"][i]
        prompt_len = min(len(prompt_enc["input_ids"][i]), len(ids))
        labels = [-100] * prompt_len + list(ids[prompt_len:])
        all_labels.append(labels)

    return {
        "input_ids": full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"],
        "labels": all_labels,
    }


def process_split(csv_path: str, split_name: str, tokenizer, output_dir: str):
    """Full pipeline: CSV → cleaned → ChatML → tokenized → saved to disk."""
    df = load_and_clean_csv(csv_path, split_name)
    df_chatml = to_chatml(df, split_name)

    if df_chatml.empty:
        print(f"[{split_name}] No valid samples!")
        return None

    hf_ds = Dataset.from_pandas(df_chatml[["prompt", "full_text"]])
    tok_ds = hf_ds.map(
        lambda ex: tokenize_with_label_masking(ex, tokenizer, MAX_SEQ_LENGTH),
        batched=True,
        batch_size=1000,
        remove_columns=["prompt", "full_text"],
        desc=f"Tokenizing {split_name}",
    )

    save_path = os.path.join(output_dir, f"{split_name}_tokenized")
    tok_ds.save_to_disk(save_path)
    print(f"[{split_name}] Saved {len(tok_ds)} samples → {save_path}")

    lengths = [len(x) for x in tok_ds["input_ids"]]
    print(
        f"[{split_name}] Seq lengths — min: {min(lengths)}, max: {max(lengths)}, "
        f"mean: {np.mean(lengths):.0f}, median: {np.median(lengths):.0f}"
    )
    return tok_ds


def main():
    parser = argparse.ArgumentParser(description="Preprocess advertisement CSV data")
    parser.add_argument("--input_dir", required=True, help="Directory containing CSV files")
    parser.add_argument("--output_dir", required=True, help="Output directory for tokenized data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    csv_files = {
        "train": "Final_advertisement_filtered_adv_lte1024_train_90.csv",
        "val": "Final_advertisement_filtered_adv_lte1024_val_5.csv",
        "test": "Final_advertisement_filtered_adv_lte1024_test_5.csv",
    }

    for split_name, filename in csv_files.items():
        csv_path = os.path.join(args.input_dir, filename)
        if os.path.exists(csv_path):
            process_split(csv_path, split_name, tokenizer, args.output_dir)
        else:
            print(f"⚠ {csv_path} not found, skipping {split_name}")


if __name__ == "__main__":
    main()
