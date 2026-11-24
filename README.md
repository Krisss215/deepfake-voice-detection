# Deepfake Voice Detection

Course project for *Advanced Topics in Machine Learning* (HIT): can a classifier detect synthetic speech, including speech made by a spoofing method it never saw during training?

## Files

| File | What it does |
| --- | --- |
| `build_dataset.py` | Builds the training set (1,520 clips, 4 s, 16 kHz mono) from the public corpora and writes `data/train.csv` |
| `download_models.py` | Downloads the two pretrained speech models used by approach B (wav2vec 2.0, WavLM) |
| `stage1_data_analysis.ipynb` | Stage 1: data analysis, baseline and shortcut checks |
| `features.py` | Feature extraction: MFCC, LFCC, log-spectrogram, self-supervised (SSL) embeddings |
| `run_experiments.py` | Every experiment in the stage 2 plan; writes results to `results/` |

## Step 1: build the data

```bash
pip install numpy pandas soundfile soxr requests pyarrow huggingface_hub remotezip
python3 build_dataset.py --out data
python3 download_models.py
```

The script does not save the full corpora (over 100 GB), only the selected clips (about 200 MB in the end). It reads the archives as streams and stops once it has enough clips, so the actual transfer is roughly 3–10 GB and a run can take one to two hours.

| Source | Taken | From |
| --- | --- | --- |
| ASVspoof 5, dev set | 300 bonafide + 300 spoof (A09–A16, split evenly) | Zenodo, record 14498691 |
| CodecFake | 300 spoof, from 10 codecs | Hugging Face: rogertseng/CodecFake |
| VCTK 0.92 | 300 bonafide – the original recordings CodecFake was made from | Edinburgh DataShare |
| M-AILABS, en_US | 320 bonafide, at most 10 clips per book | data.solak.de |
| *or* LibriSpeech dev-clean | 320 bonafide, at most 8 per speaker and 4 per chapter | OpenSLR 12 |

**M-AILABS substitute.** In November 2026 both M-AILABS download servers (data.solak.de, caito.de) were unreachable, and no other copy of the real en_US audio exists (the GitHub repo is a placeholder; MLAAD contains only the fakes). M-AILABS is built from LibriVox audiobooks, so its role – real, read audiobook speech with no matching fakes – is filled by LibriSpeech dev-clean, which comes from LibriVox too (40 speakers, ~340 MB):

```bash
python3 build_dataset.py --out data --only librispeech
```

In `train.csv` these rows have `source=LibriSpeech`; the analysis treats every real-only source the same way.

If one source fails, rerun only that one, e.g. `python3 build_dataset.py --out data --only mailabs`.
If you downloaded an archive in the browser, pass its local path instead of the URL (see `--help`).

## Step 2: run the experiments

```bash
pip install librosa scikit-learn torch transformers
python3 run_experiments.py --data data --ssl-models data/models/wav2vec2-base data/models/wavlm-base-plus
```

All scores are out-of-fold with speaker-disjoint folds, and hyperparameters (regularization strength, SSL layer) are chosen by nested cross-validation.

## Data decisions

- **Length:** source clips shorter than 3 s are skipped, and longer ones are cut to a random 4 s window. Otherwise long zero-padding would be an easy cue for identifying the corpus.
- **CodecFake and VCTK:** the project brief lists 300 real and 300 fake CodecFake clips. The real ones are VCTK recordings of the same speakers, so in `train.csv` both are marked `source=CodecFake`, and the `corpus` column keeps the true origin.
- **Speaker:** the `speaker` column includes the corpus name (`ASVspoof5:D_0123`), so speakers from different corpora are never merged in the split.
- **Audio is not included in this repository.** The data is used for the course only; run `build_dataset.py` to recreate it.

## Sources

- ASVspoof 5 – Wang et al., 2024 (CC BY 4.0)
- CodecFake – Wu, Tseng & Lee, Interspeech 2024 (CC BY 4.0)
- VCTK Corpus 0.92 – University of Edinburgh (CC BY 4.0)
- M-AILABS Speech Dataset (M-AILABS license)
- LibriSpeech – Panayotov et al., ICASSP 2015 (CC BY 4.0)
