"""
metrics_and_collator.py
-------------------------
PHASE 3: Custom Evaluation Metrics
PHASE 4 (partial): Data collator for speech-seq2seq training

WER (Word Error Rate) and CER (Character Error Rate) via jiwer, with
Indic-aware normalization applied before comparison so that:
  - Zero-width joiners/non-joiners don't inflate CER (they're often
    inserted inconsistently by different ASR/OCR sources feeding a
    corpus like Shrutilipi, and are invisible to human readers).
  - Whitespace/punctuation differences don't inflate WER when the
    underlying words match.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import torch

from data_pipeline import normalize_indic_text

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


# ------------------------------------------------------------------------
# Phase 3: compute_metrics
# ------------------------------------------------------------------------
def build_compute_metrics_fn(tokenizer):
    """
    Returns a compute_metrics callable bound to a specific tokenizer,
    as required by Seq2SeqTrainer's `compute_metrics` argument.
    """

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 (the ignore_index used for padded label positions)
        # back with the pad token id before decoding -- decoding -100
        # directly would raise an error / produce garbage tokens.
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Apply the SAME Indic-aware normalization to both hypothesis and
        # reference. Symmetry here is critical -- normalizing only one
        # side artificially inflates or deflates the error rate.
        pred_str_norm = [normalize_indic_text(p) for p in pred_str]
        label_str_norm = [normalize_indic_text(l) for l in label_str]

        # Guard against empty references after normalization (jiwer raises
        # on all-empty inputs), which can happen with noisy corpus rows.
        filtered = [
            (p, l) for p, l in zip(pred_str_norm, label_str_norm) if l.strip()
        ]
        if not filtered:
            return {"wer": 100.0, "cer": 100.0}
        preds_f, labels_f = zip(*filtered)

        wer = 100 * wer_metric.compute(predictions=list(preds_f), references=list(labels_f))
        # CER matters more than WER for Indic scripts because word
        # segmentation (what counts as a "word") is looser/less standardized
        # than in English -- character-level agreement is a more robust
        # signal of transcription quality for Devanagari/Brahmic scripts.
        cer = 100 * cer_metric.compute(predictions=list(preds_f), references=list(labels_f))

        return {"wer": wer, "cer": cer}

    return compute_metrics


# ------------------------------------------------------------------------
# Phase 4: Data collator for speech-to-text
# ------------------------------------------------------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Whisper needs two independent padding strategies in the same batch:
      1. `input_features` (log-Mel spectrograms) are padded/truncated to a
         fixed length by the feature extractor already, but we still batch
         them via the processor's feature_extractor.pad for correctness.
      2. `labels` (token ids) are variable-length and must be padded
         separately by the tokenizer, THEN have pad positions replaced
         with -100 so the loss function ignores them (standard seq2seq
         convention -- otherwise the model is penalized for not predicting
         the pad token, which teaches it nothing useful).
    """

    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # --- pad audio features ---
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # --- pad text labels ---
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Mask padding with -100 so it's excluded from the cross-entropy loss.
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If the BOS token was already prepended by the tokenizer AND will
        # also be prepended by the model's forward pass, strip the extra
        # one to avoid double-BOS sequences (a common Whisper fine-tuning
        # gotcha that silently degrades output quality).
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
