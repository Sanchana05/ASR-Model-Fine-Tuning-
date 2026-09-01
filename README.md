<<<<<<< HEAD
# Teaching Whisper to Understand Hindi — On a Free GPU

## Why this project exists

Here's a problem most people don't think about: OpenAI's Whisper is an
incredible speech recognition model, but it's mostly good at languages with
huge amounts of training data — English, Spanish, French. If you speak
Hindi, Tamil, Marathi, or one of the dozens of other Indic languages, Whisper
often stumbles: it mishears words, drops syllables, or garbles sentences
that a five-year-old native speaker would understand perfectly.

That's not because these languages are "harder." It's because the internet
(and therefore Whisper's training data) is disproportionately English. Over
1.5 billion people speak an Indic language as their first language. If
speech AI doesn't work for them, voice assistants, transcription tools,
accessibility software, and call-center automation all quietly leave them
out.

This project fixes a slice of that gap. It fine-tunes Whisper on
**Shrutilipi**, a large open dataset of Indic-language speech built by
AI4Bharat, so the model gets meaningfully better at understanding Hindi (and
can be pointed at Tamil, Telugu, Marathi, or any of the other languages in
the dataset with a one-line config change).

## Why this is a genuinely hard engineering problem (and how we solved it)

Fine-tuning a speech model normally needs a beefy GPU with 40-80GB of memory
— the kind of hardware you'd rent from a cloud provider for real money.
Most students, hobbyists, and early-career engineers don't have access to
that. What they *do* have is a free Google Colab notebook, which hands you
a T4 GPU with just 16GB of memory.

Naively trying to fine-tune Whisper on a T4 will crash almost immediately
with an "out of memory" error. So the interesting engineering challenge
here wasn't "can we fine-tune Whisper" — it was **"can we fine-tune Whisper
using hardware a student can get for free?"**

We got there by stacking five separate memory-saving tricks, each one
chipping away at the problem from a different angle:

1. **Shrink the model's weights (8-bit quantization).** Instead of storing
   every number in the model at full precision, we compress them to 8-bit
   integers — roughly a 4x size reduction, with barely any quality loss.
2. **Only train a tiny slice of the model (LoRA).** Rather than updating
   all 244 million parameters, we freeze the entire model and insert small,
   trainable "adapter" layers — under 1% of the original parameter count.
   This is the single biggest lever: it means the optimizer only has to
   track and update a fraction of the numbers.
3. **Don't store every layer's output (gradient checkpointing).** Normally,
   training keeps a copy of every layer's computation in memory so it can
   do the math backwards afterward. We instead throw most of that away and
   recompute it when needed — trading a bit of extra compute time for a lot
   of saved memory.
4. **Use lower-precision math during training (fp16).** Similar idea to
   point 1, but applied to the numbers flowing through the model during
   training, not just the stored weights.
5. **Stream the dataset instead of downloading it.** Shrutilipi is huge.
   Downloading the whole thing would fill up Colab's free disk before
   training even starts, so we pull only the audio clips we need, on
   demand, as training runs.

None of these tricks alone would be enough. It's the *combination* that
gets a 244-million-parameter speech model to fit and train comfortably
inside 16GB.

## What you'll actually build

By the end of this pipeline, you'll have:

- A Whisper model that's noticeably better at transcribing Hindi speech
  than the stock version.
- A lightweight adapter file (a few megabytes, not gigabytes) that holds
  everything the model learned — small enough to email, or upload to
  GitHub, or share with a classmate.
- A simple web app where you or anyone else can record their voice or
  upload an audio file and get a transcription back, live.

## How the pieces fit together

Think of this as an assembly line with five stations. Each file in this
repo is one station:

| Station | File | What happens here, in plain terms |
|---|---|---|
| 1. Feed the machine | `data_pipeline.py` | Pulls Hindi audio clips + their correct transcripts from Shrutilipi, and converts the audio into a format the model can actually read (a spectrogram — basically a picture of sound). |
| 2. Prep the model | `model_setup.py` | Loads Whisper in its shrunken, 8-bit form, and attaches the small trainable adapter layers (LoRA) described above. |
| 3. Grade the homework | `metrics_and_collator.py` | Defines how we measure "is the transcription actually good?" — using Word Error Rate and Character Error Rate — and handles the fiddly job of batching audio clips of different lengths together. |
| 4. Actually train it | `train.py` | Runs the training loop: play audio, predict text, compare to the correct answer, adjust the adapter weights, repeat — thousands of times, while saving progress along the way. |
| 5. Ship it | `merge_and_export.py` + `app.py` | Bakes the learned adapter back into the model for fast, simple inference, then wraps it in a Gradio app you can actually click around in. |

`config.py` is the control panel — every number you might want to tweak
(which language, how big a batch, how long to train) lives there so you
don't have to go hunting through the other files.

## Try it yourself

```bash
pip install -r requirements.txt

python train.py              # stations 1–4: teach the model Hindi
python merge_and_export.py   # station 5a: bake in what it learned
python app.py                # station 5b: talk to it
```

Everything is designed to run inside a single free Colab session, start to
finish.

## If something breaks

Out-of-memory errors are the most common thing you'll hit when
experimenting with your own settings. If that happens:

- Switch to the smaller `whisper-base` model in `config.py`.
- Cut the batch size in half (and double gradient accumulation to
  compensate — you'll train on the same effective batch size, just more
  slowly).
- Reduce the LoRA adapter size (`LORA_R`) from 32 down to 8 or 16.
- Restart your Colab runtime between major steps — leftover memory
  fragments from earlier cells are a sneaky, common cause of OOM errors
  that have nothing to do with your actual settings.

## The bigger point

This repo is small, but the idea behind it scales: a huge amount of what
makes AI feel "broken" or "not for me" for non-English speakers isn't some
fundamental limitation — it's a training data gap, and training data gaps
are fixable. You don't need a data-center GPU cluster to make a dent in
that. You need a good open dataset, a few well-chosen efficiency tricks,
and a free Colab notebook.
=======
# ASR-Model-Fine-Tuning-
>>>>>>> 2ed7e2f1735321377968021726435522a6a533b7
