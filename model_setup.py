"""
model_setup.py
----------------
PHASE 2: Extreme Memory Optimizations

Loads Whisper in 8-bit (via bitsandbytes) to shrink the base model's memory
footprint by ~4x vs fp32 / ~2x vs fp16, then wraps it with LoRA adapters so
that >99% of parameters stay frozen. This combination ("QLoRA-style" for
Whisper) is what makes whisper-small trainable on a 16GB T4 with room left
over for activations and the optimizer state.

Memory math (why this matters on a T4):
  - whisper-small full fp32 fine-tune: ~1GB weights + ~4GB optimizer states
    (Adam keeps 2 extra fp32 buffers/param) + gradients + activations
    -> routinely exceeds 16GB once you add a real batch size.
  - 8-bit weights + LoRA: base weights ~0.3GB (int8), LoRA adapter params
    are <1% of the model so gradients/optimizer states for them are
    negligible, and Adam is only tracking the tiny LoRA parameter count.
"""

import torch
from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import config


def load_quantized_whisper():
    """Loads Whisper with int8 weights via bitsandbytes."""
    # load_in_8bit swaps nn.Linear layers for bnb's Linear8bitLt, which
    # stores weights as int8 and dequantizes on-the-fly during the forward
    # pass. device_map="auto" lets accelerate place the (small) model
    # entirely on the single available GPU.
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = WhisperForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Required prep step for k-bit training: casts LayerNorm to fp32 for
    # numerical stability, enables gradient checkpointing hooks, and makes
    # the (frozen) input embeddings require grad so gradients can flow
    # through to the LoRA-adapted layers during backprop.
    model = prepare_model_for_kbit_training(model)

    # Whisper's generate() bakes in forced decoder ids for language/task;
    # we clear them here so the LoRA-tuned model doesn't fight with stale
    # forced tokens during training-time generation for eval metrics.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Gradient checkpointing trades compute for memory: instead of storing
    # every layer's activations for the backward pass, it recomputes them
    # on the fly. This alone can cut activation memory by 60-70%, which is
    # often the difference between OOM and a comfortable batch size of 8.
    model.gradient_checkpointing_enable()
    # use_cache is incompatible with gradient checkpointing (the cache
    # exists to SKIP recomputation, which defeats the point); must disable
    # during training and re-enable for inference.
    model.config.use_cache = False

    return model


def apply_lora(model):
    """Wraps the quantized base model with LoRA adapters."""
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        # SEQ_2_SEQ_LM is the correct task type for encoder-decoder models
        # like Whisper -- it ensures PEFT wraps both encoder and decoder
        # attention projections correctly rather than assuming decoder-only.
        task_type="SEQ_2_SEQ_LM",
    )
    peft_model = get_peft_model(model, lora_config)

    # Sanity check: print how few parameters we're actually training.
    # Typically <1% of total params for whisper-small with r=32 on q/v proj.
    peft_model.print_trainable_parameters()
    return peft_model


def build_model():
    base_model = load_quantized_whisper()
    peft_model = apply_lora(base_model)
    return peft_model


if __name__ == "__main__":
    m = build_model()
    print(f"Model dtype summary: {next(m.parameters()).dtype}")
