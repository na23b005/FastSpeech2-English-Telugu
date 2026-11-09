"""Speak the 5 English test sentences with the trained model and plot what the variance adaptor predicted.
Run from the project root:

    python English/inference.py

Writes Results/English/test_generated_audios/<ID>.wav and
Results/English/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png
"""
import json
import os
import re
import sys

import matplotlib
import nltk
import torch
import yaml
from g2p_en import G2p
from scipy.io import wavfile

matplotlib.use("Agg")  # save plots to files, no screen needed
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))  # the English/ folder
sys.path.insert(0, os.path.join(HERE, "..", "FastSpeech2"))  # so we can import ming024's code
from model import FastSpeech2
from text import text_to_sequence
from text.numbers import normalize_numbers
from utils.model import get_vocoder, vocoder_infer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = os.path.join(HERE, "..", "Results", "English")
SPEAKER = "spk029"  # the voice: female, 9.6 min of training audio (2nd most); clearer by ear than spk073 (most audio)
SENTENCES = {  # exactly as given in the assignment
    "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3": "The rhyth m of the orchestra changed unexpectedly.",
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
}

nltk.download("averaged_perceptron_tagger_eng", quiet=True)  # part-of-speech tagger g2p_en needs (nltk 3.9+ name)
g2p = G2p()


def to_phonemes(text):
    """Text -> ARPAbet phonemes. g2p_en does the main work: CMU dictionary, part-of-speech rule for
    words like "conduct" (noun KAA1nduct vs verb conDUH1ct) and a neural guess for unknown words
    ("andaroo", "rhyth"). Before it we fix 3 things it reads wrongly."""
    text = text.replace(":", " ")  # 1) "10:45" -> "10 45", read "ten forty five"
    text = normalize_numbers(text).replace("-", " ")  # 2) digits -> words ("2026" -> "twenty twenty-six"), then
    #                                                    split "twenty-six": g2p_en misreads hyphenated words
    phones = []
    for part in re.split(r"\b(AM)\b", text):  # 3) "AM" is spelled out "A M"; g2p_en lowercases it to the word "am"
        phones += ["EY1", "EH1", "M"] if part == "AM" else g2p(part)
    phones = ["sp" if p in ",.?!;" else p for p in phones if p != " "]  # punctuation -> short pause
    while phones[-1] == "sp":  # no pause at the end (training clips have their end silence trimmed)
        phones.pop()
    return ["sp"] + phones  # start with a short pause: without it the first word ("The") was swallowed


def bar_plot(sid, name, phones, values, ylabel):
    fig, ax = plt.subplots(figsize=(max(6, 0.22 * len(phones)), 3.2))
    ax.bar(range(len(phones)), values, color="0.35")  # dark grey bars
    ax.set_xticks(range(len(phones)), phones, rotation=90, fontsize=7)
    ax.set_xlim(-0.6, len(phones) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{sid}: predicted {name} per phoneme")
    fig.tight_layout()
    os.makedirs(os.path.join(RESULTS, "variance_adaptor_plots", name), exist_ok=True)
    fig.savefig(os.path.join(RESULTS, "variance_adaptor_plots", name, f"{sid}.png"), dpi=150)
    plt.close(fig)


def main():
    config = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
    pre, mod = config["preprocess"], config["model"]
    data = pre["path"]["preprocessed_path"]

    # Model: ming024's FastSpeech2 with our trained weights (saved in half precision, loaded back as float)
    model = FastSpeech2(pre, mod).to(device).eval()
    model.load_state_dict(torch.load(config["train"]["path"]["final_model"], map_location=device)["model"])
    vocoder = get_vocoder(mod, device)  # pretrained HiFi-GAN: mel spectrogram -> waveform

    speaker_id = json.load(open(os.path.join(data, "speakers.json")))[SPEAKER]

    # Pitch and energy were trained as z-scores; stats.json has [min, max, mean, std] to turn them back
    stats = json.load(open(os.path.join(data, "stats.json")))
    sr = pre["preprocessing"]["audio"]["sampling_rate"]
    os.makedirs(os.path.join(RESULTS, "test_generated_audios"), exist_ok=True)

    for sid, sentence in SENTENCES.items():
        phones = to_phonemes(sentence)
        ids = text_to_sequence("{" + " ".join(phones) + "}", pre["preprocessing"]["text"]["text_cleaners"])
        with torch.no_grad():
            out = model(torch.tensor([speaker_id], device=device), torch.tensor([ids], device=device),
                        torch.tensor([len(ids)], device=device), len(ids))
        mel, pitch, energy, duration = out[1], out[2][0], out[3][0], out[5][0]  # see FastSpeech2.forward
        wav = vocoder_infer(mel.transpose(1, 2), vocoder, mod, pre)[0]
        wavfile.write(os.path.join(RESULTS, "test_generated_audios", f"{sid}.wav"), sr, wav)

        bar_plot(sid, "duration", phones, duration.cpu().numpy(), "frames (1 frame = 11.6 ms)")
        bar_plot(sid, "pitch", phones, (pitch * stats["pitch"][3] + stats["pitch"][2]).cpu().numpy(), "F0 (Hz)")
        bar_plot(sid, "energy", phones, (energy * stats["energy"][3] + stats["energy"][2]).cpu().numpy(),
                 "energy (STFT frame norm)")
        print(f"{sid}: {len(wav) / sr:.2f} s | {' '.join(phones)}")


if __name__ == "__main__":
    main()
