import json
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from paper_src.utils import flatten

matplotlib.rcParams.update({"font.size": 8, "pdf.use14corefonts": True})


def get_results(multirun_dir):
    run_folders = multirun_dir.glob("train_*_*/")

    def get_result(dirpath):
        try:
            # Load the metrics
            metrics = pd.read_csv(dirpath / "metrics.csv")
            # Get last 500 epochs of validation metrics and remove "valid" from name
            metrics = metrics[metrics.epoch > metrics.epoch.max() - 500]
            metrics = metrics[[col for col in metrics if "valid" in col]].dropna()
            metrics = metrics.rename(
                {col: col.replace("valid/", "") for col in metrics}, axis=1
            )
            # Compute medians and return as a dictionary
            metrics = metrics.median().to_dict()
            # Load the hyperparameters
            with open(dirpath / "params.json", "r") as file:
                hps = flatten(json.load(file))

            return {**metrics, **hps}
        except FileNotFoundError:
            return {}

    results = pd.DataFrame([get_result(dirpath) for dirpath in run_folders])
    return results


if __name__ == "__main__":

    RUNS_HOME = Path("../../runs")
    SUBFOLDER = "user_runs/multi/20230201_ablation"

    results = get_results(RUNS_HOME / SUBFOLDER)
    rng = np.random.default_rng(42)
    order = [
        ("ablation0_baseline", "NODE", "mediumseagreen"),
        ("ablation3_rm-gradual", "No incr.", "lightseagreen"),
        ("ablation1_rm-pass", "No p.t.", "mediumturquoise"),
        ("ablation2_rm-mlp", "No MLP", "turquoise"),
        ("ablation4_vanilla", "Vanilla", "blueviolet"),
    ]
    fig, ax = plt.subplots(figsize=(2, 2))
    for i, (name, label, color) in enumerate(order):
        data = results[results.model == name]
        jitter = 0.2 * rng.uniform(-1, 1, len(data))
        ax.scatter(i + jitter, data.r2_observ, s=5, c=color)
        ax.hlines(data.r2_observ.mean(), xmin=i - 0.3, xmax=i + 0.3, colors="k", lw=1)
    ax.set_xticks(range(len(order)), [o[1] for o in order], rotation=45, ha="center")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.ylabel("Rate R2")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("png/ablation.png", dpi=300)
    plt.savefig("pdf/ablation.pdf", transparent=True)
