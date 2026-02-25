"""
Inference script for the Vietnamese Advertisement Generator.

Usage:
    python src/inference.py \
        --adapter vmhdaica/advertisement-lora \
        --product_name "Áo thun nam thể thao" \
        --description "Vải mè siêu nhẹ, co giãn 4 chiều"
"""

import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_model(adapter_path: str, base_model: str = "Qwen/Qwen3-0.6B"):
    """Load base model + LoRA adapter."""
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    return model, tokenizer


def generate_ad(
    model,
    tokenizer,
    product_name: str,
    description: str,
    max_new_tokens: int = 1024,
    num_beams: int = 2,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate an advertisement for the given product."""
    user_content = (
        f"tạo quảng cáo cho sản phẩm sau:\n"
        f"Tên sản phẩm: {product_name}\n"
        f"Mô tả: {description}"
    )
    prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=2048 - max_new_tokens,
        add_special_tokens=False,
    ).to(model.device)

    eos_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id != tokenizer.unk_token_id:
        eos_ids.append(im_end_id)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return generated.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate Vietnamese product ads")
    parser.add_argument("--adapter", default="vmhdaica/advertisement-lora")
    parser.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--product_name", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter, args.base_model)

    ad = generate_ad(
        model, tokenizer,
        product_name=args.product_name,
        description=args.description,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        temperature=args.temperature,
    )

    print("\n" + "=" * 60)
    print("📢 Generated Advertisement:")
    print("=" * 60)
    print(ad)
    print("=" * 60)


if __name__ == "__main__":
    main()
