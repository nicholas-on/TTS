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
| 3 | Acoustic model | not started |
| 4 | Vocoder | not started |
| 5 | Accuracy evaluation | not started |
| 6 | Whole-chapter reading | not started |

## Running on Google Colab

1. Open a notebook from `notebooks/` in Colab (File → Open notebook → GitHub,
   or upload it).
2. If this repository is private, create a fine-grained GitHub token with
   read-only access to it, and add it in Colab's Secrets panel (key icon, left
   sidebar) as `GITHUB_TOKEN`.
3. Run the cells in order. The first cell clones this repository, installs it,
   and mounts Google Drive.

Run the notebooks in order: `00_inspect_data`, `01_prepare_data`, then
`02_recogniser_and_alignment` (needs a GPU runtime). Notebook
00 ends with a notes cell; apply what it finds to `configs/data.yaml` and commit
before running notebook 01.

Data lives on Google Drive under `MyDrive/bibletts/` (change this in
`configs/data.yaml`):

```
bibletts/
  raw/                 downloaded originals and sources.json (notebook 00)
  prepared/            chapters_v1.tar and the config used (notebook 01)
  ctc/                 recogniser checkpoints and clip lists; clips_final.jsonl
                       holds the clips with TTS durations (notebook 02)
```

## Layout

```
PLAN.md                the plan
configs/data.yaml      data, audio, text and split settings
configs/ctc.yaml       recogniser, training, segmentation and alignment settings
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
  acoustic/ vocoder/   later milestones (empty for now)
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
