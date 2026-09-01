"""
merge_and_export.py
---------------------
PHASE 5 (part 1): Merge LoRA adapters back into the base Whisper model.

Why merge?
  During training we kept the base model in 8-bit + LoRA adapters separate
  for memory efficiency. But at INFERENCE time, keeping them separate means
  every forward pass pays the cost of on-the-fly LoRA delta computation
  (base_output + A @ B @ x) on top of int8 dequantization. Merging bakes
  the LoRA delta directly into the base weights ONE TIME, producing a
  single dense model that is faster to run and simpler to deploy (e.g. to
  Gradio, an API server, or further quantization for edge deployment).

  We reload the base model in fp16 (not 8-bit) for this step because
  bitsandbytes' Linear8bitLt layers don't support in-place weight merging
  cleanly -- merging needs full-precision arithmetic before we optionally
  re-quantize for deployment.
"""

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

import config


def merge_lora_adapter(
    adapter_dir: str = config.OUTPUT_DIR,
    merged_output_dir: str = f"{config.OUTPUT_DIR}-merged",
):
    print(f"Loading base model '{config.MODEL_ID}' in fp16 for merging...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from '{adapter_dir}'...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Merging adapter weights into base model...")
    # merge_and_unload() folds W_lora = W_base + (alpha/r) * A@B into the
    # base layer weights and removes the PEFT wrapper, returning a plain
    # WhisperForConditionalGeneration -- so downstream code (Gradio, HF
    # pipelines) doesn't need to know PEFT was ever involved.
    merged_model = peft_model.merge_and_unload()

    # Re-enable the KV cache now that we're done with gradient checkpointing
    # -- caching is a pure speed win at inference/generation time.
    merged_model.config.use_cache = True

    merged_model.save_pretrained(merged_output_dir)

    processor = WhisperProcessor.from_pretrained(adapter_dir)
    processor.save_pretrained(merged_output_dir)

    print(f"Merged model saved to: {merged_output_dir}")
    return merged_output_dir


if __name__ == "__main__":
    merge_lora_adapter()
