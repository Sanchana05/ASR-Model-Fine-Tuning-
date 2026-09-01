"""
app.py
-------
PHASE 5 (part 2): Gradio Web App for the fine-tuned Whisper model.

Accepts either live microphone recording or an uploaded .wav/.mp3 file,
runs it through the merged fine-tuned model, and displays the transcription.

Run with:  python app.py
(On Colab, launch with `demo.launch(share=True)` to get a public link.)
"""

import gradio as gr
import numpy as np
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import config

MERGED_MODEL_DIR = f"{config.OUTPUT_DIR}-merged"

device = "cuda" if torch.cuda.is_available() else "cpu"
# fp16 on GPU for fast inference; fall back to fp32 on CPU (fp16 matmuls
# are not well-supported on most CPUs).
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading merged model from {MERGED_MODEL_DIR} onto {device} ({dtype})...")
model = WhisperForConditionalGeneration.from_pretrained(
    MERGED_MODEL_DIR, torch_dtype=dtype
).to(device)
model.eval()

processor = WhisperProcessor.from_pretrained(MERGED_MODEL_DIR)

# Pin the language/task so the model doesn't have to guess -- this matters
# because a fine-tuned-for-Hindi model can still mis-detect language on
# short/noisy mic clips if left to auto-detect.
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=config.WHISPER_LANGUAGE_TOKEN, task=config.WHISPER_TASK
)


def transcribe(audio):
    """
    `audio` from Gradio's Audio component (type="numpy") arrives as a
    (sample_rate, numpy_array) tuple for both mic recordings and uploaded
    files, so a single code path handles both input modes.
    """
    if audio is None:
        return "Please record or upload an audio clip."

    sample_rate, audio_array = audio

    # Gradio may hand us int16 PCM; convert to float32 in [-1, 1] since
    # that's what the feature extractor expects.
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32) / np.iinfo(np.int16).max

    # Collapse to mono if stereo -- Whisper is single-channel only.
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)

    # Resample to Whisper's required 16kHz if the mic/file wasn't already.
    if sample_rate != config.SAMPLING_RATE:
        audio_array = librosa.resample(
            audio_array, orig_sr=sample_rate, target_sr=config.SAMPLING_RATE
        )

    inputs = processor(
        audio_array, sampling_rate=config.SAMPLING_RATE, return_tensors="pt"
    )
    input_features = inputs.input_features.to(device=device, dtype=dtype)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=225,
        )

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy", label="Speak or upload audio"),
    outputs=gr.Textbox(label="Transcription"),
    title="Fine-tuned Whisper — Hindi ASR (Shrutilipi)",
    description=(
        "Fine-tuned via LoRA on the ai4bharat/Shrutilipi dataset. "
        "Record from your microphone or upload a .wav/.mp3 file."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    # share=True is convenient on Colab to get a public URL, since Colab
    # itself doesn't expose local ports.
    demo.launch(share=True)
