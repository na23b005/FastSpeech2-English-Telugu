"""Train FastSpeech 2 on the prepared Svarah data. ming024's train.py, adapted. Run from the project root:

    python English/train.py                     # train from scratch
    python English/train.py --restore_step N    # resume from English/logs/ckpt/N.pth.tar
    python English/train.py --plot              # draw training / validation loss curves from the logs
"""
import argparse
import os
import re
import sys
import time

import matplotlib
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

matplotlib.use("Agg")  # save plots to files, no screen needed
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))  # the English/ folder
sys.path.insert(0, os.path.join(HERE, "..", "FastSpeech2"))  # so we can import ming024's code
from dataset import Dataset
from evaluate import evaluate
from model import FastSpeech2Loss
from utils.model import get_model, get_param_num, get_vocoder
from utils.tools import log, synth_one_sample, to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_NAMES = ["Total", "Mel", "Mel PostNet", "Pitch", "Energy", "Duration"]


def main(args, configs):
    preprocess_config, model_config, train_config = configs
    paths, steps, opt = train_config["path"], train_config["step"], train_config["optimizer"]

    # Data: load 4 batches at a time and sort them by length, so clips in a batch need little padding
    dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    group_size = 4
    loader = DataLoader(dataset, batch_size=opt["batch_size"] * group_size, shuffle=True, collate_fn=dataset.collate_fn)

    # Model (random weights, or a checkpoint with --restore_step), optimizer with warm-up / annealing, loss, vocoder
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)  # uses every GPU Kaggle gives (T4 x2)
    print("Number of FastSpeech2 Parameters:", get_param_num(model))
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    vocoder = get_vocoder(model_config, device)  # only to listen to samples during training

    for p in [paths["ckpt_path"], paths["result_path"], os.path.dirname(paths["final_model"])]:
        os.makedirs(p, exist_ok=True)
    train_logger = SummaryWriter(os.path.join(paths["log_path"], "train"))  # TensorBoard logs + log.txt
    val_logger = SummaryWriter(os.path.join(paths["log_path"], "val"))

    def save_checkpoint():  # full checkpoint (model + optimizer state) so training can resume
        torch.save({"model": model.module.state_dict(), "optimizer": optimizer._optimizer.state_dict()},
                   os.path.join(paths["ckpt_path"], f"{step}.pth.tar"))

    step = args.restore_step + 1
    start = time.time()
    bar = tqdm(total=steps["total_step"], initial=args.restore_step, desc="Training")
    while True:
        for batchs in loader:
            for batch in batchs:
                # One training step: predict, compare with the real mel / pitch / energy / duration, update weights
                batch = to_device(batch, device)
                output = model(*(batch[2:]))
                losses = Loss(batch, output)  # [total, mel, mel postnet, pitch, energy, duration]
                (losses[0] / opt["grad_acc_step"]).backward()
                if step % opt["grad_acc_step"] == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), opt["grad_clip_thresh"])  # avoid exploding gradients
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % steps["log_step"] == 0:  # write the training losses
                    losses = [l.item() for l in losses]
                    message = f"Step {step}/{steps['total_step']}, " + ", ".join(
                        f"{n} Loss: {l:.4f}" for n, l in zip(LOSS_NAMES, losses))
                    with open(os.path.join(paths["log_path"], "train", "log.txt"), "a") as f:
                        f.write(message + "\n")
                    bar.write(message)
                    log(train_logger, step, losses=losses)

                if step % steps["synth_step"] == 0:  # turn one training sample into audio (TensorBoard)
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch, output, vocoder, model_config, preprocess_config)
                    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
                    log(train_logger, fig=fig, tag=f"Training/step_{step}_{tag}")
                    log(train_logger, audio=wav_prediction, sampling_rate=sr, tag=f"Training/step_{step}_{tag}_synthesized")

                if step % steps["val_step"] == 0:  # validation loss on all val clips (ming024's evaluate.py)
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(paths["log_path"], "val", "log.txt"), "a") as f:
                        f.write(message + "\n")
                    bar.write(message)
                    model.train()

                if step % steps["save_step"] == 0:
                    save_checkpoint()

                # Stop at total_step or when the time budget is used up. Save a full checkpoint (to resume)
                # and a small model-only copy in half precision (~70 MB, fits on GitHub) for inference.
                if step == steps["total_step"] or time.time() - start > steps["max_hours"] * 3600:
                    save_checkpoint()
                    state = {k: v.half() if v.is_floating_point() else v for k, v in model.module.state_dict().items()}
                    torch.save({"model": state, "step": step}, paths["final_model"])
                    print(f"Stopped at step {step}, final model saved to {paths['final_model']}")
                    return
                step += 1
                bar.update(1)


def plot_losses(train_config):
    # Each log line looks like "Step N/..., Total Loss: x, Mel Loss: x, ..., Duration Loss: x";
    # the first number is the step and the last 6 numbers are the losses.
    out_dir = os.path.join(HERE, "..", "Results", "English")
    os.makedirs(out_dir, exist_ok=True)
    for split, png in [("train", "training_loss.png"), ("val", "validation_loss.png")]:
        with open(os.path.join(train_config["path"]["log_path"], split, "log.txt")) as f:
            rows = [re.findall(r"\d+\.?\d*", line) for line in f]
        steps = [int(r[0]) for r in rows]
        fig, axes = plt.subplots(2, 3, figsize=(13, 6))
        for i, ax in enumerate(axes.flat):
            ax.plot(steps, [float(r[-6 + i]) for r in rows])
            ax.set_title(f"{LOSS_NAMES[i]} loss")
            ax.set_xlabel("step")
        fig.suptitle(f"English FastSpeech 2 - {'training' if split == 'train' else 'validation'} loss")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, png), dpi=150)
        print("saved", os.path.join(out_dir, png))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))  # one file instead of ming024's three
    configs = (config["preprocess"], config["model"], config["train"])
    if args.plot:
        plot_losses(config["train"])
    else:
        main(args, configs)
