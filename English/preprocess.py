"""Prepare the Svarah dataset for FastSpeech 2. Run from the project root:

    python English/preprocess.py prepare     # step 4: download, select ~6 h, split, write wav + lab + lexicon
    (run MFA, see WORKLOG step 5)
    python English/preprocess.py features    # step 5: mel, pitch, energy, duration + train.txt / val.txt
"""
import io
import os
import re
import sys

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))  # the English/ folder
sys.path.insert(0, os.path.join(HERE, "..", "FastSpeech2"))  # so we can import ming024's code
from text import _clean_text

config = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))["preprocess"]
paths = config["path"]
SR = config["preprocessing"]["audio"]["sampling_rate"]  # 22050
SPLIT_CSV = os.path.join(HERE, "data", "split.csv")  # list of selected clips and their split (kept in git)
SPEAKER_INFO = ["gender", "age-group", "primary_language", "native_place_state", "native_place_district",
                "highest_qualification", "job_category", "occupation_domain"]


def prepare():
    # 1) Download Svarah from Hugging Face (gated dataset: needs the HF_TOKEN environment variable)
    from huggingface_hub import snapshot_download
    folder = os.path.join(snapshot_download("ai4bharat/Svarah", repo_type="dataset"), "data")
    df = pd.concat([pd.read_parquet(os.path.join(folder, f)) for f in sorted(os.listdir(folder))], ignore_index=True)
    print(f"Svarah: {len(df)} clips, {df.duration.sum() / 3600:.2f} h")
    print("example audio paths:", [a["path"] for a in df.audio_filepath[:3]])

    # 2) Speakers: Svarah has no speaker id column, so one speaker = one unique combination of the speaker details
    df["speaker"] = "spk" + df.groupby(SPEAKER_INFO, dropna=False).ngroup().astype(str).str.zfill(3)
    df["name"] = [f"svarah{i:05d}" for i in df.index]
    print("speakers:", df.speaker.nunique())

    # 3) Filter: 1-11 s clips (max_seq_len = 1000 frames = 11.6 s) whose transcript is plain text
    #    (letters and basic punctuation only: drops noise tags, digits and symbols)
    df["text"] = df.text.str.strip()
    df = df[df.duration.between(1, 11) & df.text.str.fullmatch(r"[A-Za-z' ,.?!-]+", na=False)]
    print(f"after filtering: {len(df)} clips, {df.duration.sum() / 3600:.2f} h")

    # 4) Select: shuffle with a fixed seed (same result every run) and keep clips until we have 6 hours
    df = df.sample(frac=1, random_state=0)
    df = df[df.duration.cumsum() <= 6 * 3600].copy()

    # 5) Split by sentence: a fixed hash sends 1 in 6 distinct sentences (~1 h) to validation, the rest to training.
    #    All clips of the same sentence land on the same side, so no clip and no sentence is in both splits.
    sentence = df.text.str.lower().str.replace(r"[^a-z ]", "", regex=True)
    df["split"] = np.where(pd.util.hash_pandas_object(sentence, index=False) % 6 == 0, "val", "train")
    assert set(df.speaker[df.split == "val"]) <= set(df.speaker[df.split == "train"])  # every val speaker is trained on
    print(df.groupby("split").agg(clips=("name", "size"), hours=("duration", lambda d: d.sum() / 3600),
                                  speakers=("speaker", "nunique")))
    os.makedirs(os.path.dirname(SPLIT_CSV), exist_ok=True)
    df[["name", "speaker", "split", "duration", "text"] + SPEAKER_INFO].to_csv(SPLIT_CSV, index=False)

    # 6) For every clip write a 22050 Hz wav and a cleaned .lab transcript, one folder per speaker
    #    (the input format of the aligner, same as ming024's prepare_align)
    words = set()
    for row in df.itertuples():
        wav, sr = sf.read(io.BytesIO(row.audio_filepath["bytes"]))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)  # stereo -> mono
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        wav = wav / np.abs(wav).max() * 32767  # peak-normalize to 16-bit, like ming024
        text = _clean_text(row.text, config["preprocessing"]["text"]["text_cleaners"])  # e.g. lowercase
        speaker_dir = os.path.join(paths["raw_path"], row.speaker)
        os.makedirs(speaker_dir, exist_ok=True)
        wavfile.write(os.path.join(speaker_dir, row.name + ".wav"), SR, wav.astype(np.int16))
        with open(os.path.join(speaker_dir, row.name + ".lab"), "w") as f:
            f.write(text)
        words.update(re.findall(r"[a-z][a-z']*", text))
    print("original sample rate (last clip):", sr)

    # 7) Pronunciation dictionary for the aligner: ming024's librispeech lexicon
    #    + g2p_en guesses for words it does not contain (Indian names, places, ...)
    import nltk
    from g2p_en import G2p
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)  # needed by g2p_en with newer nltk
    lexicon = open(os.path.join(HERE, "..", "FastSpeech2", "lexicon", "librispeech-lexicon.txt")).read().strip()
    known = {line.split()[0].lower() for line in lexicon.splitlines()}
    g2p = G2p()
    new = [(w, [p for p in g2p(w) if p.strip()]) for w in sorted(words - known)]
    new = [f"{w.upper()}  {' '.join(p)}" for w, p in new if p]
    with open(paths["lexicon_path"], "w") as f:
        f.write(lexicon + "\n" + "\n".join(new) + "\n")
    print(f"{len(new)} new words added to the lexicon with g2p_en, e.g. {new[:3]}")


def features():
    # 1) ming024's Preprocessor (unchanged apart from step 2's fixes): for every clip that MFA aligned it computes
    #    the mel spectrogram, pitch (pyworld), energy and phoneme durations (from the TextGrid),
    #    plus stats.json (pitch/energy mean and std) and speakers.json (speaker name -> number)
    from preprocessor.preprocessor import Preprocessor
    Preprocessor(config).build_from_path()

    # 2) With val_size: 0 ming024 wrote every clip into train.txt; split it again using our split.csv from step 4
    split = pd.read_csv(SPLIT_CSV, index_col="name")
    out = paths["preprocessed_path"]
    lines = open(os.path.join(out, "train.txt"), encoding="utf-8").read().splitlines()
    for s in ["train", "val"]:
        keep = [line for line in lines if split.loc[line.split("|")[0], "split"] == s]
        with open(os.path.join(out, f"{s}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(keep) + "\n")
        hours = split.loc[[line.split("|")[0] for line in keep], "duration"].sum() / 3600
        print(f"{s}: {len(keep)} clips, {hours:.2f} h")


if __name__ == "__main__":
    {"prepare": prepare, "features": features}[sys.argv[1]]()
