# FastSpeech 2 Text-to-Speech: Indian English (Svarah)

FastSpeech 2 trained from scratch on about 5 hours of the Svarah Indian-English dataset, starting from the
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) implementation (commit `d4e79eb`).
Speech is produced with ming024's pretrained HiFi-GAN "universal" vocoder.

**This submission contains the English model only.** The Telugu model (IndicTTS Telugu) is not included.

- Technical report: [`Report/Technical_Report.pdf`](Report/Technical_Report.pdf)
- Test audio: [`Results/English/test_generated_audios/`](Results/English/test_generated_audios/) (E1, E2a, E2b, E3, E4)
- Variance-adaptor plots: [`Results/English/variance_adaptor_plots/`](Results/English/variance_adaptor_plots/) (duration, pitch, energy per sentence)
- Loss curves: [`training_loss.png`](Results/English/training_loss.png), [`validation_loss.png`](Results/English/validation_loss.png)
- Trained model: [`English/checkpoints/english_fastspeech2.pth.tar`](English/checkpoints/english_fastspeech2.pth.tar) (70 MB, model weights in half precision, step 24897)

## Repository layout

```
English/
  preprocess.py        Svarah download, selection, train/val split, wav + transcript files, lexicon; then features
  train.py             training (ming024 train.py adapted) and loss plots (--plot)
  inference.py         the 5 test sentences -> audio + duration / pitch / energy plots
  config.yaml          all English settings (ming024's 3 LibriTTS yaml files merged)
  checkpoints/         trained model
  data/split.csv       every selected clip with its speaker, split, duration and text
  data/preprocessed/   stats.json (pitch / energy normalization), speakers.json (speaker ids)
FastSpeech2/           ming024 code with the small changes listed below
Results/English/       loss plots, test audio, variance-adaptor plots
Report/                Technical_Report.pdf
requirements.txt
```

## Data

- Svarah (`ai4bharat/Svarah` on Hugging Face, gated): 6656 clips, 9.6 h, 117 Indian-English speakers, phone recordings.
- Kept clips of 1-11 s whose transcript has only letters and basic punctuation: 4212 clips, 5.70 h (all used).
- Speaker id = unique combination of the speaker details (Svarah has no speaker id column): 115 speakers.
- Split by sentence with a fixed hash (1 in 6 distinct sentences to validation), so no clip and no sentence is in both splits:

| Split | Clips | Hours | Speakers |
|---|---|---|---|
| train | 3532 | 4.76 | 115 |
| val | 680 | 0.94 | 107 (all also in train) |

## Changes to ming024/FastSpeech2

| File | Change | Why |
|---|---|---|
| `audio/stft.py` | named arguments for `pad_center` and `librosa.filters.mel`; removed forced `.cuda()` / `.cpu()` | current librosa; preprocessing without an NVIDIA GPU |
| `preprocessor/preprocessor.py` | read empty TextGrid intervals as `sp`; skip clips without a TextGrid | MFA 3 writes silences as empty intervals; the original reused the previous clip's pitch / energy |
| `utils/model.py` | vocoder paths from the package folder, `map_location`; `weights_only=False` when resuming | run from any folder or on CPU; current PyTorch refuses numpy numbers in checkpoints |
| `utils/tools.py` | energy targets converted to float32 | float64 energies crashed training ("Found dtype Double but expected Float") |
| `config/LibriTTS/*.yaml` -> `English/config.yaml` | one file; Svarah paths; `val_size: 0` (own split); 60000 steps, anneal 30k/40k/50k, save every 5000; `max_hours: 3` | one config per language; fits a Kaggle session |
| `prepare_align.py`, `preprocess.py` -> `English/preprocess.py` | Svarah download, filtering, selection, sentence-level split, lexicon (LibriSpeech lexicon + g2p_en for unknown words); then ming024's `Preprocessor` | Svarah-specific preparation |
| `train.py` -> `English/train.py` | single config, time limit, small half-precision final model, `--plot` | Kaggle time limit; loss plots |
| `synthesize.py` -> `English/inference.py` | g2p_en front end (part-of-speech choice for "conduct"), fixes for "10:45", hyphenated numbers and "AM", a pause at the start, one chosen voice (`spk029`), per-phoneme plots | ming024 used only the first dictionary pronunciation and made only mel plots |

## Reproduce

Preprocessing and training ran on Kaggle (GPU T4 x2, Internet on, Hugging Face token as the Kaggle secret `HF_TOKEN`).
Run these cells in one notebook, ideally with *Save Version -> Save & Run All* so the outputs are kept:

```
# 1. code and packages
!git clone https://github.com/na23b005/FastSpeech2-English-Telugu.git /kaggle/working/project
%cd /kaggle/working/project
!pip install -q g2p_en pyworld tgt unidecode
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

# 2. download Svarah, select, split, write wav + .lab + lexicon (~10 min)
!python English/preprocess.py prepare

# 3. install the Montreal Forced Aligner (conda-forge) and its English model
!curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /tmp bin/micromamba
!/tmp/bin/micromamba create -y -q -p /tmp/mfa -c conda-forge montreal-forced-aligner
!/tmp/bin/micromamba run -p /tmp/mfa mfa model download acoustic english_us_arpa

# 4. forced alignment -> phoneme durations (~30 min)
!/tmp/bin/micromamba run -p /tmp/mfa mfa align English/data/raw English/data/lexicon.txt english_us_arpa English/data/preprocessed/TextGrid -j 4 --clean --beam 100 --retry_beam 400

# 5. mel, pitch, energy, duration + train.txt / val.txt (~20 min)
!python English/preprocess.py features

# 6. unzip the vocoder (git keeps only the zip)
!unzip -o -q FastSpeech2/hifigan/generator_universal.pth.tar.zip -d FastSpeech2/hifigan

# 7. train (stops after 3 h and saves English/checkpoints/english_fastspeech2.pth.tar)
!python English/train.py

# 8. loss plots
!python English/train.py --plot
```

Resume a stopped run with `python English/train.py --restore_step N` (N = a checkpoint in `English/logs/ckpt/`).

Inference runs on a CPU in a few seconds per sentence:

```
pip install -r requirements.txt
python -c "import zipfile; zipfile.ZipFile('FastSpeech2/hifigan/generator_universal.pth.tar.zip').extractall('FastSpeech2/hifigan')"
python English/inference.py
```

All commands run from the repository root.

## References

- Y. Ren et al., "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech", ICLR 2021.
- ming024/FastSpeech2: https://github.com/ming024/FastSpeech2
- Svarah: https://huggingface.co/datasets/ai4bharat/Svarah
- Montreal Forced Aligner: https://montreal-forced-aligner.readthedocs.io
