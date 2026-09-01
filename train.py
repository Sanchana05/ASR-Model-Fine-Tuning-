"""
train.py
---------
PHASE 4: Training & Checkpointing

Ties together Phase 1 (data), Phase 2 (quantized+LoRA model), and Phase 3
(metrics/collator) into a Seq2SeqTrainer run, with memory-optimized
TrainingArguments and a callback that saves only the (tiny) LoRA adapter
weights -- NOT the full base model -- at each checkpoint.

Run with:  python train.py
"""

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

import config
from data_pipeline import get_processed_datasets
from model_setup import build_model
from metrics_and_collator import (
    DataCollatorSpeechSeq2SeqWithPadding,
    build_compute_metrics_fn,
)


class SavePeftAdapterCallback(TrainerCallback):
    """
    Ensures that on every checkpoint save, we ONLY persist the LoRA adapter
    (a few MB) rather than the full merged model. Trainer's default
    `save_model` on a PeftModel already does this correctly, but we add an
    explicit callback so the behavior is visible/documented and so we can
    hook in an optional Hub push per checkpoint rather than only at the end.
    """

    def on_save(self, args, state, control, model=None, **kwargs):
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        # model.save_pretrained on a PEFT-wrapped model writes only the
        # adapter_config.json + adapter_model.safetensors -- tiny relative
        # to the ~1GB base model, which is critical for not blowing past
        # Colab's disk quota across many checkpoints.
        model.save_pretrained(checkpoint_dir)
        print(f"[checkpoint] Saved LoRA adapter to {checkpoint_dir}")

        if config.PUSH_TO_HUB:
            model.push_to_hub(
                config.HUB_MODEL_ID,
                commit_message=f"LoRA checkpoint @ step {state.global_step}",
            )
            print(f"[hub] Pushed adapter to {config.HUB_MODEL_ID}")
        return control


def main():
    # ---------------- Phase 1: data ----------------
    train_ds, eval_ds, processor = get_processed_datasets()

    # ---------------- Phase 2: model ----------------
    model = build_model()

    # ---------------- Phase 3: collator + metrics ----------------
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics = build_compute_metrics_fn(processor.tokenizer)

    # ---------------- Phase 4: training args ----------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        # Gradient accumulation lets us simulate a larger effective batch
        # size (8 * 2 = 16) without the memory cost of actually materializing
        # activations for 16 samples at once -- essential on a T4 where
        # per-device batch size is capped by log-Mel activation memory.
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        max_steps=config.MAX_STEPS,
        # fp16 mixed precision roughly halves activation memory vs fp32 and
        # speeds up matmuls on the T4's tensor cores. Combined with 8-bit
        # base weights, this is the second pillar (alongside LoRA + grad
        # checkpointing) of fitting Whisper training in 16GB.
        fp16=True,
        # Re-stated here for clarity even though model_setup.py already
        # enables it on the model -- Trainer needs this flag to correctly
        # handle the recomputation during its training loop.
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_steps=config.SAVE_STEPS,
        logging_steps=config.LOGGING_STEPS,
        # generation-based eval is what actually lets us compute WER/CER
        # (as opposed to just eval loss), at the cost of extra eval time.
        predict_with_generate=True,
        generation_max_length=225,
        # Only load the best model at the end if we're checkpointing WER,
        # so the final saved adapter is the best-performing one, not just
        # the last one trained (guards against late-stage overfitting).
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
        # Optimizer choice: paged_adamw_8bit (bitsandbytes) keeps optimizer
        # states in 8-bit too, shaving additional memory versus standard
        # AdamW -- meaningful since even a "tiny" LoRA param count still
        # carries 2x Adam moment buffers.
        optim="paged_adamw_8bit",
        dataloader_num_workers=config.NUM_WORKERS,
        remove_unused_columns=False,  # required for PEFT + custom columns
        label_names=["labels"],
        push_to_hub=config.PUSH_TO_HUB,
        hub_model_id=config.HUB_MODEL_ID if config.PUSH_TO_HUB else None,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftAdapterCallback()],
    )

    # Silence use_cache warning spam during training (already disabled in
    # model_setup.py for gradient checkpointing compatibility).
    model.config.use_cache = False

    print("Starting training...")
    trainer.train()

    print("Saving final LoRA adapter...")
    trainer.save_model(config.OUTPUT_DIR)
    processor.save_pretrained(config.OUTPUT_DIR)

    if config.PUSH_TO_HUB:
        trainer.push_to_hub()
        print(f"Final adapter pushed to {config.HUB_MODEL_ID}")


if __name__ == "__main__":
    main()
