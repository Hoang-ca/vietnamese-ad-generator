"""
Centralized configuration for the Vietnamese Advertisement Generator.
All hyperparameters and paths are defined here for reproducibility.
"""

import os
import torch

# ──────────────────────────────────────────────────────────────
# Paths (adjust to your environment)
# ──────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "data")

TRAIN_CSV = os.path.join(DATA_DIR, "Final_advertisement_filtered_adv_lte1024_train_90.csv")
VAL_CSV   = os.path.join(DATA_DIR, "Final_advertisement_filtered_adv_lte1024_val_5.csv")
TEST_CSV  = os.path.join(DATA_DIR, "Final_advertisement_filtered_adv_lte1024_test_5.csv")

OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", "output")
CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")
FINAL_MODEL_DIR = os.path.join(OUTPUT_DIR, "final_best_model")

# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-0.6B"

# ──────────────────────────────────────────────────────────────
# LoRA
# ──────────────────────────────────────────────────────────────
LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ──────────────────────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────────────────────
NUM_TRAIN_EPOCHS            = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE  = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE               = 3e-4
WEIGHT_DECAY                = 0.01
WARMUP_RATIO                = 0.03
MAX_SEQ_LENGTH              = 1024
COMPUTE_DTYPE               = torch.float16
SEED                        = 42
