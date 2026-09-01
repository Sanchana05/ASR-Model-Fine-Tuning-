"""
config.py
---------
Single source of truth for hyperparameters, paths, and language settings.
Centralizing config avoids "magic numbers" scattered across scripts and makes
it trivial to switch the target language (Shrutilipi covers ~12 Indic languages)
or swap whisper-small <-> whisper-base if you hit OOM even after optimization.
"""

import os

# ----------------------------------------------------------------------------
# Model selection
# ----------------------------------------------------------------------------
# whisper-small (244M params) fits comfortably with 8-bit + LoRA on a T4.
# Drop to "openai/whisper-base" (74M params) if you still see OOM with long
# audio clips or want a bigger batch size.
MODEL_ID = "openai/whisper-small"

# ----------------------------------------------------------------------------
# Dataset selection
# ----------------------------------------------------------------------------
# Shrutilipi is organized by language config. Change LANGUAGE to any of:
# "hindi", "marathi", "tamil", "telugu", "kannada", "malayalam", "gujarati",
# "bengali", "odia", "punjabi", "sanskrit", "urdu" (check the dataset card for
# the exact available config names before running).
DATASET_ID = "ai4bharat/Shrutilipi"
LANGUAGE = "hindi"
WHISPER_LANGUAGE_TOKEN = "hi"       # Whisper's internal language code for Hindi
WHISPER_TASK = "transcribe"

# Cap dataset size for a free Colab session (T4 sessions have a wall-clock
# limit ~12h and disk is limited too). Increase once the pipeline is verified.
MAX_TRAIN_SAMPLES = 8000
MAX_EVAL_SAMPLES = 500

# Streaming avoids downloading the full (large) Shrutilipi corpus to Colab's
# ~78GB local disk, which routinely fills up before training even starts.
USE_STREAMING = True

# ----------------------------------------------------------------------------
# Audio / feature settings
# ----------------------------------------------------------------------------
SAMPLING_RATE = 16000            # Whisper's expected input sample rate
MAX_AUDIO_DURATION_S = 30        # Whisper's max input window (30s log-mel)
MAX_LABEL_LENGTH = 448           # Whisper decoder max positions

# ----------------------------------------------------------------------------
# LoRA (PEFT) configuration
# ----------------------------------------------------------------------------
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
# Whisper's attention projections are where LoRA gives the best
# accuracy-per-trainable-parameter tradeoff for ASR fine-tuning.
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# ----------------------------------------------------------------------------
# Training arguments (tuned for a 16GB T4)
# ----------------------------------------------------------------------------
OUTPUT_DIR = "./whisper-small-shrutilipi-hi-lora"
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2   # effective batch size = 8 * 2 = 16
LEARNING_RATE = 1e-3              # LoRA tolerates much higher LR than full FT
WARMUP_STEPS = 50
MAX_STEPS = 2000
EVAL_STEPS = 200
SAVE_STEPS = 200
LOGGING_STEPS = 25
NUM_WORKERS = 2                   # Colab typically gives 2 CPU cores

# ----------------------------------------------------------------------------
# Hub push settings (optional)
# ----------------------------------------------------------------------------
PUSH_TO_HUB = False
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "your-username/whisper-small-shrutilipi-hi-lora")
