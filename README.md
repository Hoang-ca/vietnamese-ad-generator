# 🎯 Vietnamese Advertisement Generator

> Comparative study of **4 fine-tuned language models** (BARTpho, mT5, ViT5, Qwen3-0.6B) for automatically generating Vietnamese product advertisements from product name and description. The best-performing model — **Qwen3-0.6B with LoRA** — is published on HuggingFace for inference.

[![HuggingFace Model](https://img.shields.io/badge/🤗_HuggingFace-vmhdaica/advertisement--lora-blue)](https://huggingface.co/vmhdaica/advertisement-lora)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Problem Statement

E-commerce platforms require large volumes of product advertisements. Writing these manually is time-consuming and inconsistent. This project fine-tunes a language model to **generate creative, engaging Vietnamese advertisements** given a product's name and description.

## 🏗️ Approach

```
Product Name + Description → Fine-tuned LM → Advertisement
```

**Pipeline overview:**
1. **Data Collection**: Scraped 103K+ product records from Tiki e-commerce platform; used GPT-4o mini to generate structured advertisements following a problem/solution format
2. **Data Preparation**: Formatted as ChatML conversations with prompt/completion label masking; filtered to ≤1,024 tokens → 89,029 final samples
3. **Tokenizer Extension**: Added 1,083 emoji/symbol tokens commonly used in Vietnamese ads
4. **Multi-Model Fine-tuning**: Trained **4 models** with LoRA adapters — BARTpho-word-base, mT5-Small, ViT5-base, and Qwen3-0.6B — to compare Seq2Seq vs. causal LM approaches for Vietnamese ad generation
5. **Comprehensive Evaluation**: Compared all 4 models using both automated metrics (BLEU, ROUGE) and human evaluation (grammar, readability, relevance) on the same test set

## 📊 Dataset

| Split | Samples | Ratio |
|-------|---------|-------|
| Train | 80,125  | 90%   |
| Val   | 4,452   | 5%    |
| Test  | 4,452   | 5%    |

- **Source**: Vietnamese e-commerce product data from Tiki
- **Ad generation**: GPT-4o mini API with structured problem/solution prompt (~650–750 words per ad)
- **Max sequence length**: 1,024 tokens (median ~1,286 after ChatML formatting)
- **Preprocessing**: Deduplicated by product ID; removed HTML tags and hashtags; added 1,083 emoji tokens to vocabulary

## ⚙️ Training Configuration

**All 4 models** were fine-tuned with LoRA under comparable settings:

| Hyperparameter | Value |
|----------------|-------|
| Models Trained | BARTpho-word-base, mT5-Small, ViT5-base, **Qwen3-0.6B** |
| LoRA Rank (r) | 8 |
| LoRA Alpha | 16 (Qwen3) / 32 (Seq2Seq models) |
| LoRA Dropout | 0.05 |
| LoRA Targets | All attention + MLP projection layers |
| Effective Batch Size | 64 |
| Learning Rate | 3e-4 → 2e-4 → 1e-4 → 5e-5 (manual decay) |
| Max Epochs | 10 (early stop on val plateau) |
| Precision | FP16 |
| Optimizer | AdamW (fused) |
| Hardware | 2× NVIDIA T4 16GB (Kaggle) |

> The published model on HuggingFace is **Qwen3-0.6B** (best ROUGE-1/ROUGE-2) with only 0.84% trainable parameters via LoRA.

## 📈 Evaluation Results (4 Models Compared)

### Automated Metrics — Full Test Set (4,452 samples)

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|------|---------|---------|---------|
| BARTpho-word-base | **0.4214** | 0.7838 | 0.5530 | **0.4443** |
| mT5-Small | 0.1063 | 0.5852 | 0.2721 | 0.3213 |
| ViT5-base | 0.2229 | 0.8440 | 0.5779 | 0.4183 |
| **Qwen3-0.6B** ⭐ | 0.2711 | **0.8504** | **0.5978** | 0.4254 |

**Qwen3-0.6B** was selected as the best overall model — achieving the highest **ROUGE-1** and **ROUGE-2** scores, indicating superior content coverage. BARTpho-word-base led in **BLEU** (n-gram precision) and **ROUGE-L** (longest common subsequence), but produced shorter, less complete outputs.

### Human Evaluation — 100-Sample Subset (scored 0–1)

| Model | Grammar & Logic | Readability | Relevance |
|-------|----------------|-------------|-----------|
| BARTpho-word-base | 0.537 | 0.562 | 0.652 |
| mT5-Small | 0.254 | 0.280 | 0.504 |
| ViT5-base | **0.660** | **0.670** | **0.768** |
| **Qwen3-0.6B (ours)** | 0.643 | 0.635 | 0.688 |

Human evaluation assessed grammatical correctness, logical coherence, readability, and relevance to the source product information on 100 randomly selected test samples.

## 🚀 Quick Start

### Inference with the Published Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "vmhdaica/advertisement-lora")
tokenizer = AutoTokenizer.from_pretrained("vmhdaica/advertisement-lora")

# Create prompt
product_name = "Áo thun nam thể thao chạy bộ thoáng khí PROMAX"
description = "Chất liệu vải mè siêu nhẹ, co giãn 4 chiều, thấm hút mồ hôi."

prompt = f"""<|im_start|>user
tạo quảng cáo cho sản phẩm sau:
Tên sản phẩm: {product_name}
Mô tả: {description}<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        num_beams=4,
        no_repeat_ngram_size=3,
        temperature=0.7,
        top_p=0.9,
    )

generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(generated)
```

### Reproduce Training

```bash
# 1. Clone this repository
git clone https://github.com/Hoang-ca/vietnamese-ad-generator.git
cd vietnamese-ad-generator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset (see scripts/download_data.md)

# 4. Run training notebook on Kaggle with GPU T4 x2
#    Upload notebooks/fine-tune-gen-qc.ipynb to Kaggle
```

## 📁 Repository Structure

```
vietnamese-ad-generator/
├── README.md                          # This file
├── MODEL_CARD.md                      # Model card with limitations & ethical notes
├── requirements.txt                   # Python dependencies with versions
├── .gitignore                         # Ignore data, checkpoints, caches
├── LICENSE                            # MIT License
│
├── data/
│   └── samples.csv                    # 3 example rows showing data format
│
├── notebooks/
│   ├── fine-tune-gen-qc.ipynb         # Training notebook
│   └── inference.ipynb                # Colab inference notebook (run the model)
│
├── src/
│   ├── config.py                      # Centralized hyperparameter configuration
│   ├── preprocess.py                  # CSV → ChatML data processing pipeline
│   └── inference.py                   # Inference / generation script
│
├── scripts/
│   └── download_data.md               # Instructions to obtain the dataset
│
└── artifacts/
    └── metrics.json                   # Evaluation metrics (BLEU, ROUGE, human eval)
```

## 🔑 Key Design Decisions

- **LoRA over full fine-tuning**: Only 0.84% parameters trainable → fits on consumer GPUs (2×T4) while preserving base model knowledge
- **ChatML format**: Uses Qwen's native `<|im_start|>/<|im_end|>` template for natural instruction-following behavior
- **Emoji tokenizer expansion**: Added 1,083 emoji tokens to reduce tokenization artifacts in Vietnamese ad text (heavy emoji usage)
- **Label masking**: Only computes loss on the advertisement completion, not the prompt — prevents the model from memorizing prompts
- **Dynamic padding**: Pads to batch max length instead of global max, saving ~40% compute for variable-length sequences
- **Manual LR decay**: Trained epoch-by-epoch; reduced LR (3e-4 → 5e-5) when validation metrics plateaued; selected best checkpoint

## 📚 References

- Hu, E. J., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Qwen Team. (2025). *Qwen3 Technical Report.*
- Nguyen, L. Q., et al. (2021). *BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese.*
- Phan, L., et al. (2022). *ViT5: Pretrained Text-to-Text Transformer for Vietnamese.*
- Xue, L., et al. (2021). *mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer.*

## 🙏 Acknowledgements

- [Qwen Team](https://huggingface.co/Qwen) for the Qwen3-0.6B base model
- [Hugging Face](https://huggingface.co/) for Transformers, PEFT libraries
- [Kaggle](https://www.kaggle.com/) for GPU compute resources

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The LoRA adapter weights are published on [HuggingFace](https://huggingface.co/vmhdaica/advertisement-lora).
