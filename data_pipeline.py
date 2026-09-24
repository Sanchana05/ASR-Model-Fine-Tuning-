"""
data_pipeline.py
-----------------
PHASE 1: Setup & Data Pipeline

Loads a language subset of ai4bharat/Shrutilipi, resamples audio to 16kHz,
extracts log-Mel features via WhisperFeatureExtractor, and tokenizes text
labels via WhisperTokenizer, with normalization tuned for Indic scripts.

Why streaming?
  Shrutilipi is a large multi-language ASR corpus. Downloading a full
  language split can be 50-100+ GB. On free Colab (limited disk + session
  time), `streaming=True` lets us pull only the samples we actually use,
  which is essential to even START training without hitting disk quota.
"""

import re
import unicodedata

from datasets import load_dataset, Audio, IterableDataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

import config


# ------------------------------------------------------------------------
# 1. Text normalization for Indic scripts
# ------------------------------------------------------------------------
# Raw Shrutilipi transcripts often contain:
#   - Zero-width joiners/non-joiners (ZWJ U+200D, ZWNJ U+200C) that are
#     visually invisible but change string equality / WER calculations.
#   - Inconsistent whitespace, punctuation (Devanagari danda "।"), and
#     mixed digit systems (Devanagari vs. Arabic numerals).
# We do NOT want aggressive Latin-style normalization (e.g. lowercasing
# does nothing for Devanagari, and stripping diacritics would destroy
# meaning), so this normalizer is conservative and script-aware.
_ZERO_WIDTH_CHARS = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_MULTI_SPACE = re.compile(r"\s+")
# Common punctuation to strip for WER/CER comparability (keeps the model's
# actual predictions untouched; this normalization is applied identically
# to references and hypotheses at eval time, and only to references at
# train-label-encoding time for consistency).
_PUNCT_TO_STRIP = re.compile(r"[।,.!?\"'“”‘’();:]")


def normalize_indic_text(text: str) -> str:
    """Conservative, script-agnostic cleanup safe for Devanagari and other
    Brahmic scripts used across Shrutilipi's languages."""
    if text is None:
        return ""
    # NFC normalization ensures combining marks (matras) are represented
    # consistently -- critical because two visually-identical Devanagari
    # strings can be byte-different without this step.
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_CHARS.sub("", text)
    text = _PUNCT_TO_STRIP.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


# ------------------------------------------------------------------------
# 2. Load processor components
# ------------------------------------------------------------------------
def load_processor():
    """WhisperProcessor bundles the feature extractor (audio -> log-Mel)
    and tokenizer (text -> token ids). Loading them together guarantees
    they stay in sync (same special tokens, same language/task tokens)."""
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.MODEL_ID)
    tokenizer = WhisperTokenizer.from_pretrained(
        config.MODEL_ID,
        language=config.WHISPER_LANGUAGE_TOKEN,
        task=config.WHISPER_TASK,
    )
    processor = WhisperProcessor.from_pretrained(
        config.MODEL_ID,
        language=config.WHISPER_LANGUAGE_TOKEN,
        task=config.WHISPER_TASK,
    )
    return feature_extractor, tokenizer, processor


# ------------------------------------------------------------------------
# 3. Load Shrutilipi (streaming) and cast audio column to 16kHz
# ------------------------------------------------------------------------
def load_shrutilipi_streaming():
    """
    Returns (train_stream, eval_stream) as IterableDataset objects.

    Note: `ai4bharat/Shrutilipi` config/column names vary by loading
    script version. Adjust `LANGUAGE`, and the text column name below
    ("transcription" is common; some mirrors use "text" or "sentence"),
    against the dataset card if this errors out.
    """
    train_stream = load_dataset(
        config.DATASET_ID,
        config.LANGUAGE,
        split="train",
        streaming=config.USE_STREAMING,
        trust_remote_code=True,
    )
    # Cast the audio column so `datasets` lazily decodes + resamples to
    # 16kHz on the fly, instead of us hand-rolling librosa.resample calls.
    # This is far cheaper than eagerly materializing a resampled copy.
    train_stream = train_stream.cast_column(
        "audio", Audio(sampling_rate=config.SAMPLING_RATE)
    )

    # Some Shrutilipi entries point at broken/missing audio files and decode
    # to None instead of raising -- drop those before they reach .map(),
    # since prepare_example indexes into audio["array"] unconditionally.
    train_stream = train_stream.filter(lambda ex: ex["audio"] is not None)

    # Shrutilipi's public loader may not expose a separate "validation"
    # split; carve one out of the (infinite, streamed) train iterator.
    eval_stream = train_stream.take(config.MAX_EVAL_SAMPLES)
    train_stream = train_stream.skip(config.MAX_EVAL_SAMPLES).take(
        config.MAX_TRAIN_SAMPLES
    )
    return train_stream, eval_stream


# ------------------------------------------------------------------------
# 4. Per-example preprocessing function
# ------------------------------------------------------------------------
def build_prepare_fn(feature_extractor, tokenizer, text_column="transcription"):
    """Returns a closure suitable for `.map()` over the streaming dataset."""

    def prepare_example(batch):
        audio = batch["audio"]

        # --- Audio -> log-Mel spectrogram features ---
        # Whisper's encoder expects a fixed-size (80, 3000) log-Mel input
        # (padded/truncated to 30s). The feature extractor handles the
        # padding/truncation and the log-Mel transform in one call, which
        # is both correct and much faster than a hand-written librosa path.
        features = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        )
        batch["input_features"] = features.input_features[0]

        # --- Text -> token ids ---
        raw_text = batch.get(text_column, batch.get("text", ""))
        clean_text = normalize_indic_text(raw_text)
        batch["labels"] = tokenizer(clean_text).input_ids

        return batch

    return prepare_example


def get_processed_datasets():
    """End-to-end Phase 1 entry point used by train.py."""
    feature_extractor, tokenizer, processor = load_processor()
    train_stream, eval_stream = load_shrutilipi_streaming()

    prepare_fn = build_prepare_fn(feature_extractor, tokenizer)

    # `remove_columns` on a streaming dataset needs explicit column names;
    # we drop the raw audio/text after conversion to keep memory low since
    # streamed batches don't get cached back to disk.
    columns_to_remove = train_stream.column_names or ["audio", "transcription"]

    train_ds = train_stream.map(prepare_fn, remove_columns=columns_to_remove)
    eval_ds = eval_stream.map(prepare_fn, remove_columns=columns_to_remove)

    return train_ds, eval_ds, processor
