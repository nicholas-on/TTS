# bibletts

A from-scratch, single-speaker text-to-speech system for Bible audio, as a
proof of concept for low-resource languages. English stands in for the target
language, and nothing in the pipeline uses a pretrained model.

The full plan, with milestones, data choices and risks, is in [PLAN.md](PLAN.md).

## Status

| Milestone | What | State |
|---|---|---|
| 0 | Inspect the data | notebook ready, not yet run on real data |
| 1 | Prepare the chapter corpus | notebook ready, not yet run on real data |
| 2 | CTC recogniser, alignment, segmentation | notebook ready, not yet run on real data |
| 3 | Acoustic model, text to speech (Griffin-Lim playback) | notebook ready, not yet run on real data |
| 4 | Vocoder | not started |
| 5 | Accuracy evaluation | not started |
| 6 | Whole-chapter reading | not started |

## Running the proof of concept on Google Colab

1. Go to https://colab.research.google.com, choose File → Open notebook →
   GitHub, enter `nicholas-on/TTS`, and open a notebook from `notebooks/`.
   For a private repository, tick "Include private repos".
2. The repository is private, so the notebooks need a GitHub token to clone it.
   One-time setup:
   - On GitHub: Settings → Developer settings → Personal access tokens →
     Fine-grained tokens → Generate new token. Repository access: "Only
     select repositories" → `nicholas-on/TTS`. Permissions → Repository
     permissions → Contents: Read-only. Generate, and copy the token.
   - In Colab: key icon in the left sidebar (Secrets) → Add new secret.
     Name `GITHUB_TOKEN`, value the token. Switch on "Notebook access"; Colab
     asks for this once per notebook.
3. Run the cells in order. The first cell clones this repository, installs it,
   and mounts Google Drive. Everything worth keeping is saved to Drive, so
   each notebook can run in a fresh session.

Run the notebooks in this order:

| Notebook | Runtime | Rough time | Produces |
|---|---|---|---|
| `00_inspect_data` | CPU | 30 min + listening | downloads; notes for `configs/data.yaml` |
| `01_prepare_data` | CPU | under 1 h | prepared chapters archive |
| `02_recogniser_and_alignment` | T4 GPU | 2–4 h | clips with per-character durations |
| `03_acoustic_model` | T4 GPU | several hours | the TTS model; speech from any text |

Times are guesses until the first real run. Notebook 00 ends with a notes
cell; apply what it finds to `configs/data.yaml` and commit before running
notebook 01. Training cells resume from their last checkpoint after a
disconnect: rerun the setup cells and the training cell.

Data lives on Google Drive under `MyDrive/bibletts/` (change this in
`configs/data.yaml`):

```
bibletts/
  raw/                 downloaded originals and sources.json (notebook 00)
  prepared/            chapters_v1.tar and the config used (notebook 01)
  ctc/                 recogniser checkpoints and clip lists; clips_final.jsonl
                       holds the clips with TTS durations (notebook 02)
  acoustic/            acoustic model checkpoints, training samples, and
                       outputs/ with synthesised speech (notebook 03)
```

## Layout

```
PLAN.md                the plan
configs/data.yaml      data, audio, text and split settings
configs/ctc.yaml       recogniser, training, segmentation and alignment settings
configs/acoustic.yaml  acoustic model, training and synthesis settings
src/bibletts/
  audio.py             decoding, mel-spectrograms, Griffin-Lim, loudness, pauses
  text.py              text cleaning, numbers to words, token vocabularies
  books.py             book codes, names and chapter counts
  download.py          downloads, finding audio links, matching files to chapters
  corpus.py            splits, chapter preparation, manifest, Drive archives
  viz.py               plots used by the notebooks
  colab.py             Colab helpers
  evaluate.py          character and word error rates
  ctc/
    model.py           the recogniser (conv + bidirectional GRU, CTC)
    data.py            mel caching, clip datasets, batching, SpecAugment
    train.py           training with checkpoint and resume
    align.py           forced alignment (Viterbi on the CTC graph)
    segment.py         cutting chapters into clips, durations for TTS
    pipeline.py        the milestone-2 steps the notebook calls
  acoustic/
    model.py           FastSpeech-style model: encoder, durations, decoder
    data.py            clip dataset and batching
    train.py           training with checkpoint and resume, CER on synthesised speech
    synthesis.py       text of any length to speech (Griffin-Lim for now)
  vocoder/             milestone 4 (empty for now)
notebooks/             one notebook per milestone
tests/                 checks that run on CPU in a few seconds
```

## Running the tests locally

Needs Python 3.10+ and ffmpeg.

```bash
pip install -e ".[dev]"
```

```bash
pytest
```
