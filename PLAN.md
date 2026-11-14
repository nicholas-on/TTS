# From-scratch Bible TTS: English proof of concept

A plan for building a single-speaker text-to-speech system with no pretrained
models, trained on about 10 hours of read scripture, runnable on Google Colab.
English stands in for a low-resource language: nothing in the pipeline may rely
on anything that would not exist for the real target language.

## 1. Goal and rules

**Goal.** Given unseen Bible text, produce speech in which every word is
present, in order, and pronounced recognisably. Accuracy comes first.
Naturalness comes second, and expressiveness is out of scope.

**Rules of the simulation.**

- No pretrained models anywhere in the system: no aligner, no vocoder, no
  speech recogniser, no pronunciation dictionary, no phoneme converter.
- Algorithms with no learned weights are fine (Fourier transforms, Griffin-Lim,
  Viterbi, energy-based silence detection).
- Only about 10 hours of paired audio and text from one reader are used for
  training, including for the vocoder.
- Input is plain characters, because a low-resource language will have no
  pronunciation dictionary.

**Out of scope for the proof of concept.** Multiple speakers, emotional or
conversational style, real-time synthesis, numbers and abbreviations beyond what
the Bible text contains.

**What "done" looks like** (proposed targets, to be revised once measured):

| Check | Target |
|---|---|
| Segmented clips whose text matches the audio exactly (listen to 50) | at least 95% |
| Our own recogniser's character error rate on real held-out audio | 10% or lower, so it is usable as a measuring stick |
| Character error rate on synthesised speech vs. real speech, same text | within about 3 points |
| Skipped or repeated words in 50 synthesised verses, judged by a person | zero |
| Words a listener gets wrong in those 50 verses | under 3% |

## 2. The central design decision

Bible recordings come as one audio file per chapter, and the text comes as
verses. Before any TTS model can train, the chapters have to be cut into short
clips with exactly matching text. Normally a pretrained aligner does that. We
do not have one, and neither will the target language.

The plan therefore builds **one small speech recogniser from scratch first**,
trained with CTC loss on the same 10 hours, and uses it for three jobs:

1. **Segmentation.** Align each whole chapter to its text and cut it into
   clips of 2 to 12 seconds.
2. **Durations.** Tell the TTS model how many audio frames each character
   lasts. With durations given, the TTS model cannot skip or repeat anything,
   because every character gets its own stretch of frames.
3. **Evaluation.** Transcribe synthesised speech and count character errors,
   giving an accuracy number without any outside tool.

This replaces the in-model alignment (Monotonic Alignment Search) suggested
earlier. The recogniser is needed for segmentation anyway, it builds on ASR
knowledge you already have, and it keeps the TTS model itself as simple as
possible. In-model alignment stays on the list as an upgrade (section 9).

```
chapter audio + chapter text
        │
        ▼
[1] CTC recogniser (from scratch) ──► clips + per-character durations
        │                                        │
        │                                        ▼
        │                         [2] acoustic model: characters ─► mel-spectrogram
        │                                        │
        │                                        ▼
        │                         [3] vocoder: mel-spectrogram ─► waveform
        │                                        │
        └──────────── transcribes ◄──────────────┘
                 [4] accuracy = character error rate
```

## 3. Data

### Source

- **Audio:** World English Bible, read by one man, public domain, from
  eBible.org (`ebible.org/eng-web/audio/`, `ebible.org/webaudio/`).
- **Text:** World English Bible from the same site. eBible publishes several
  formats; the "read aloud" plain-text format (one file per chapter, footnotes
  removed) is the best starting point if it is available, otherwise the
  verse-per-line format.

**Not yet verified.** The sandbox this plan was written in could not reach
eBible.org, so the exact file names, audio format, bitrate, reader credit and
total length are unconfirmed. Milestone 0 exists to check them. If the recording
turns out to be unusable (heavy noise, music, several readers), the fallback is
any single-reader public-domain audiobook with matching text, treated the same
way.

### Selection and split

Split by chapter, before anything is trained, and never change it.

| Set | Contents | Size | Used for |
|---|---|---|---|
| Train | Matthew, Mark, Luke, John, plus as much of Acts as needed | about 10 hours after cleaning | all training |
| Dev | one chapter held out of each training book | about 15 minutes | tuning, early stopping |
| Test, same style | Romans, or the unused chapters of Acts | about 30 minutes | main accuracy number |
| Test, different style | Genesis 1–3, a few Psalms, one chapter of Proverbs, one of Revelation | about 30 minutes | generalisation |
| Test, names | a genealogy chapter (for example 1 Chronicles 1) | about 5 minutes | rare proper names |

The test sets reflect the real use: the model will be asked to read books it
never heard. Measure actual durations in Milestone 0 and adjust the book list to
land on 10 hours.

### Audio settings

| Setting | Value | Note |
|---|---|---|
| Sample rate | 22,050 Hz mono | drop to 16,000 Hz if the source has no content above 8 kHz |
| Mel-spectrogram | 80 bins, FFT 1024, hop 256, window 1024, 0–8,000 Hz, log scale | one frame is about 11.6 ms |
| Loudness | normalise each chapter to the same level | |
| Clip length | 2 to 12 seconds | cut only at pauses |

### Text settings

- Lowercase. Keep letters, apostrophe, and word boundaries.
- A word boundary is one token whose type records the punctuation there:
  plain space, comma, full stop, question mark, and so on. Punctuation carries
  pause and intonation cues, so the TTS model sees it. The recogniser treats
  all boundary types as a plain space.
- Strip verse numbers, footnote marks, section headings, quotation marks and
  brackets.
- Spell out any digits. The World English Bible mostly writes numbers as words,
  so this should be a short rule list.
- Add a start token and an end token to absorb leading and trailing silence.

## 4. Milestones

Each milestone ends with something to look at or listen to, and a gate that must
pass before moving on. Time estimates assume a Colab T4 GPU and are rough guesses
to be replaced by measurements.

### Milestone 0: look at the data (no GPU)

Read first: any introduction to sampling and spectrograms (Valerio Velardo's
"Audio Signal Processing for ML" videos cover it).

1. Download two or three chapters of audio and the matching text in Colab.
2. Listen. Write down: is it one reader throughout, is there music or a spoken
   chapter announcement ("Matthew, chapter one"), is there background noise,
   does the spoken text match the written text word for word.
3. Check the file format, sample rate and bitrate, and plot a spectrogram to see
   where the audio content stops.
4. Add up durations per book and fix the book list for the splits.
5. Convert one clip to a mel-spectrogram and back to audio with Griffin-Lim, and
   listen to it.

**Gate:** one reader, clean enough audio, text matches. A written note of every
quirk found (announcements, text differences).

### Milestone 1: data preparation (CPU, under an hour of compute)

1. Download the chosen books, convert to WAV at the chosen sample rate,
   normalise loudness.
2. Clean the text with the rules above. Print the full character inventory and
   check it contains nothing unexpected.
3. If chapters start with a spoken announcement, add that phrase to the start of
   each chapter's text so text and audio match.
4. Find pauses in each chapter with a plain energy threshold (quieter than about
   40 dB below the peak for at least 0.3 seconds).
5. Store everything on Google Drive as a few large archive files, because Drive
   is slow with many small files.

**Gate:** every chapter has a WAV file, a clean text file and a pause list. A
plot of pauses over a waveform looks right for five random chapters.

### Milestone 2: the recogniser, built by bootstrapping (about 1–2 GPU hours per round)

Read first: the CTC loss (Graves 2006; the Distill article "Sequence Modeling
with CTC" is the clearest explanation) and "CTC-Segmentation of Large Corpora"
(Kürzinger et al. 2020).

This is the riskiest part of the project and the part most like the real
low-resource problem.

**Model.** Mel-spectrogram in, characters out.

- Two convolution layers, then four bidirectional GRU layers of 256–384 units,
  then a linear layer over about 30 output tokens (blank, letters, apostrophe,
  space).
- Roughly 10 million parameters.
- SpecAugment (masking random time and frequency bands) to limit overfitting.
- One config switch for time resolution: half frame rate for whole-chapter
  alignment (smaller memory), full frame rate for the final durations.

**Round 0: rough clips with no model.**

1. Assume the reader speaks at a constant rate within a chapter, so the time at
   which each verse ends is proportional to the number of characters before it.
2. About every 20–30 seconds, cut at the detected pause nearest to a predicted
   verse end.
3. The result is rough clips whose edges may be wrong by a few words. Long
   clips are used on purpose so the wrong part is a small fraction.
4. Train the recogniser on these. CTC tolerates some mislabelled edges.

**Round 1: exact alignment.**

1. Run the round-0 model over each whole chapter to get per-frame character
   probabilities.
2. Find the best path through the whole chapter's text with the Viterbi
   algorithm (forced alignment). Write a simple version yourself on a short clip
   to understand it, then use `torchaudio.functional.forced_align` for
   chapter-length inputs. It is an algorithm, not a pretrained model.
3. Every character now has a time. Cut at verse or sentence ends that coincide
   with a pause, aiming for 2–12 seconds.
4. Score each clip by its average alignment probability and drop the worst few
   percent. This removes places where the reader departed from the text.

**Round 2: clean model.**

1. Retrain from scratch on the clean clips, at full frame rate.
2. Realign. If nearly all boundaries move by less than 50 ms, stop.
3. Align the dev and test chapters with this model too, so real reference clips
   exist for them. Their text and audio are never used for training.

**Durations from the alignment.** CTC marks each character with a short spike
and fills the gaps with blanks. Convention: each character owns the frames from
its own spike up to the next character's spike. The convention is imperfect but
consistent, and the TTS model only needs consistency.

**Gates:**

- Listen to 50 random clips while reading their text: at least 95% match
  exactly.
- Character error rate on dev clips, plain greedy decoding: 10% or lower.
- Plot durations over a spectrogram for ten clips: character boundaries sit
  roughly where the sounds change.

**If round 0 does not learn** (loss does not fall, output is noise): mark verse
boundaries by hand for about 30 minutes of audio in Audacity, train on that
clean seed, and continue from round 1. This is an hour or two of manual work and
is equally available in the target language.

### Milestone 3: the acoustic model (about 6–12 GPU hours)

Read first: "Let's build GPT" (Karpathy) for transformer blocks, then the
FastSpeech and FastSpeech 2 papers (Ren et al. 2019, 2020).

**Model.** Characters in, mel-spectrogram out, FastSpeech-style, the minimum
version.

| Part | Details |
|---|---|
| Character embedding | 256 dimensions, plus position encoding |
| Encoder | 4 transformer blocks (2 attention heads, convolutional feed-forward) |
| Duration predictor | 2 convolution layers and a linear layer; predicts log duration per character |
| Length regulator | repeats each character's encoding for its number of frames |
| Decoder | 4–6 transformer blocks, then a linear layer to 80 mel bins |
| Losses | L1 on the mel-spectrogram, mean squared error on log durations |
| Size | roughly 20 million parameters, dropout 0.2 |

No pitch or energy prediction in this version. Speech will sound flatter than
the reader, and words should still be correct.

**Order of work.**

1. Overfit on 10 clips until the output is near-perfect. This proves the code
   is right before spending GPU hours.
2. Train on the full set: batch size about 32, AdamW, learning-rate warm-up,
   mixed precision, clips grouped by length.
3. During training, every few thousand steps, synthesise the dev set and log:
   - mel-spectrogram error and duration error;
   - **character error rate from our recogniser applied straight to the
     predicted mel-spectrograms.** No vocoder is needed, so this accuracy curve
     is available from the first hour;
   - a few audio samples through Griffin-Lim.

**Gate:** Griffin-Lim samples of unseen dev verses are robotic but every word is
understandable, and the mel-level character error rate is close to that of real
audio.

### Milestone 4: the vocoder (the longest job, about 20–40+ GPU hours over several sessions)

Read first: the HiFi-GAN paper (Kong et al. 2020), and a short introduction to
GANs if they are new.

The vocoder needs only audio, no text. **Start it as soon as Milestone 2
produces clips, in a separate Colab session, alongside Milestone 3.**

1. Implement HiFi-GAN: a generator that upsamples the mel-spectrogram 256 times
   to a waveform, and two sets of discriminators. Compare against the public
   reference code (`jik876/hifi-gan`) while writing it.
2. Train on the training clips only, on random one-second slices.
3. Check progress by converting real dev mel-spectrograms to audio and
   listening. Usable quality is expected after roughly 100,000–200,000 steps;
   it keeps improving for far longer.
4. Afterwards, fine-tune for a short while on mel-spectrograms produced by the
   acoustic model (using real durations so they line up with real audio). The
   acoustic model's output is smoother than real spectrograms, and this teaches
   the vocoder to cope.

**Gate:** real mel-spectrogram → vocoder → audio is clearly the reader's voice,
and our recogniser scores it about the same as the original recording.

**If training time is the blocker:** switch the generator to a Vocos-style
design (Siuzdak 2023), which predicts a Fourier spectrum at frame rate and
inverts it directly. It is simpler and trains much faster per step.

### Milestone 5: measure accuracy (under an hour of compute, plus listening)

Run the same test text through four conditions. Each one isolates one part of
the system.

| Condition | What a bad score means |
|---|---|
| Real recording | the recogniser itself is weak; this is the floor |
| Real mel-spectrogram → vocoder | the vocoder is losing words |
| Text + real durations → acoustic model → vocoder | the acoustic model is the problem |
| Text only → full system | the duration predictor is the problem |

Report character error rate for each condition on each test set (same style,
different style, names).

Then:

1. **Error list.** Rank words by how often they are misrecognised and listen to
   the top 30. Expect rare names and irregular English spellings here. English
   spelling is unusually irregular, so a language with regular spelling should
   do better on this point.
2. **Human check.** Listen to 50 synthesised test verses without seeing the
   text and type what you hear, then count word errors. This is the number that
   counts, because the recogniser was trained on the same data as the TTS model
   and may share its blind spots.
3. **Optional, English only.** Transcribe the same audio with a strong
   off-the-shelf recogniser (for example Whisper) and check that its ranking of
   good and bad verses agrees with ours. It validates the measuring method and
   is not part of the system. The target language will not have this option, so
   it is worth knowing now whether our own recogniser can be trusted.

### Milestone 6: read a whole chapter

1. Split a chapter into verses or sentences, synthesise each, join them with
   pauses sized by punctuation.
2. Listen to an unseen chapter from start to finish.

**Gate:** the success targets in section 1.

## 5. Running on Colab

| Topic | Approach |
|---|---|
| Code | All logic lives in this repository as a Python package. Notebooks only clone the repo, install it, mount Drive and call scripts. One notebook per milestone. |
| Storage | Google Drive holds data archives, checkpoints and logs. At session start, copy the data archive to the Colab local disk and unpack it there. Expected size: about 2 GB of audio and 1 GB of mel-spectrograms. |
| Disconnects | Every training script saves a checkpoint to Drive every 1,000 steps (model, optimiser, step count, random seed), keeps the last three, and resumes automatically when rerun. Build this before the first long run. |
| GPU | A free T4 (16 GB) is enough for Milestones 2 and 3. The vocoder is feasible on the free tier across many sessions; Colab Pro makes it far less painful. |
| Speed | Mixed precision, precomputed mel-spectrograms, two data-loader workers. |
| Tracking | TensorBoard logs on Drive: loss curves, dev error rate, alignment plots, audio samples. |
| Settings | One config file per model, saved with each checkpoint. |

Rough total: 30–60 GPU hours, most of it the vocoder.

## 6. Repository layout

```
PLAN.md
README.md
pyproject.toml
configs/                 one file per model
src/bibletts/
  audio.py               loading, mel-spectrogram, Griffin-Lim, pause detection
  text.py                cleaning, tokens
  data.py                datasets, batching
  ctc/                   recogniser: model, training, forced alignment, segmentation
  acoustic/              FastSpeech-style model, training, synthesis
  vocoder/               HiFi-GAN generator, discriminators, training
  evaluate.py            character error rate, the four-condition report
notebooks/
  00_inspect_data.ipynb
  01_prepare_data.ipynb
  02_train_recogniser.ipynb
  03_train_acoustic.ipynb
  04_train_vocoder.ipynb
  05_evaluate.ipynb
tests/                   small checks that run on CPU
```

## 7. Risks

| Risk | Sign | Response |
|---|---|---|
| Rough round-0 clips are too noisy to learn from | recogniser loss stalls | hand-mark 30 minutes as a clean seed (Milestone 2) |
| Spoken text differs from written text | clusters of low alignment scores | drop low-scoring clips; fix systematic differences in text cleaning |
| Source audio is poor (noise, low bitrate) | seen in Milestone 0 | lower the sample rate; if severe, change recording |
| Whole-chapter alignment runs out of memory | crash on long chapters | align at half frame rate; restrict the search to a band around the expected position |
| Durations from CTC are too coarse | blurry or slurred consonants, poor boundary plots | switch to alignment learned inside the TTS model (section 9) |
| Rare names mispronounced | high error on the names test set | expected in English; record it and judge again on the target language |
| Smooth acoustic output sounds muffled through the vocoder | real mel sounds fine, predicted mel does not | vocoder fine-tuning step in Milestone 4 |
| Our recogniser misses real errors | human check disagrees with it | trust the human check; use the optional outside recogniser to diagnose |
| Colab sessions end mid-run | lost progress | checkpoint and resume from day one |

## 8. What changes for the real language

Only the items below change. Everything else is reused as is.

- The character inventory and text-cleaning rules.
- The chapter announcement phrase, if the recording has one.
- The pause-detection threshold, tuned to the recording.
- The book list for the splits.
- A native speaker does the human check.

## 9. After the proof of concept

In order of expected benefit to accuracy and quality:

1. **Pitch prediction** (FastPitch-style, with pitch measured by a classical
   algorithm such as YIN). Fixes the flat delivery.
2. **Alignment learned inside the TTS model** ("One TTS Alignment to Rule Them
   All", Badlani et al. 2021; Monotonic Alignment Search from Glow-TTS).
   Finer durations than CTC spikes give.
3. **Longer vocoder training**, or a move to a Vocos-style vocoder.
4. **Data-size study.** Retrain on 2, 5 and 10 hours to learn how much recording
   a new language really needs.

## 10. Reading list, in the order it is needed

| When | What |
|---|---|
| Milestone 0 | Valerio Velardo, "Audio Signal Processing for ML" (videos) |
| Milestone 2 | Hannun, "Sequence Modeling with CTC" (Distill); Kürzinger et al. 2020, CTC segmentation; Park et al. 2019, SpecAugment |
| Milestone 3 | Karpathy, "Let's build GPT"; Ren et al. 2019 and 2020, FastSpeech and FastSpeech 2; Shen et al. 2018, Tacotron 2 (for background on why attention alignment fails) |
| Milestone 4 | Kong et al. 2020, HiFi-GAN; Siuzdak 2023, Vocos |
| Overview | Tan et al. 2021, "A Survey on Neural Speech Synthesis" |
| Later | Łańcucki 2021, FastPitch; Badlani et al. 2021; Kim et al. 2020, Glow-TTS |
