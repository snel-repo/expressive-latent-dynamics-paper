import json
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

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
    SUBFOLDER_MAP = {
        "Arneodo": (("gru_arneodo", "node_arneodo"), 3),
        "Rossler": (("gru_rossler", "node_rossler"), 3),
        "Lorenz": (("gru_lorenz", "node_lorenz"), 3),
        "Arneodo100": (("arneodo100",), 3),
        "MackeyGlass100": (("mackeyglass100",), 10),
    }

    for sys_name, (subfolder, true_dim) in SUBFOLDER_MAP.items():
        if len(subfolder) == 2:
            gru_subfolder, node_subfolder = subfolder
            # Load results dataframes
            gru_results = get_results(RUNS_HOME / gru_subfolder)
            gru_results["model"] = "GRU"
            node_results = get_results(RUNS_HOME / node_subfolder)
            node_results["model"] = "NODE"
            results = pd.concat([gru_results, node_results])
        else:
            results = get_results(RUNS_HOME / subfolder[0])
            results["model"] = results["model"].str.upper()
        # Fill for runs that used the default seed
        results = results.fillna({"seed": 0})
        # Compute mean and SEM across all seeds
        means = results.groupby(["model", "model.latent_size"]).mean(numeric_only=True).reset_index()
        errors = results.groupby(["model", "model.latent_size"]).std(numeric_only=True).reset_index()
        # Create the figure
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(2, 4))
        panel_plot_kwargs = {
            "loss": {
                "ax": axes[0],
                "legend": True,
                "ylabel": "Spike NLL",
            },
            "r2_observ": {
                "ax": axes[1],
                "legend": False,
                "ylabel": "Rate R2",
                "ylim": (0, 1.05),
            },
            "r2_latent": {
                "ax": axes[2],
                "legend": False,
                "ylabel": "Hidden R2",
                "ylim": (0, 1.05),
            },
        }
        shared_plot_kwargs = dict(
            logx=True,
            xlabel="Hidden size",
            marker="o",
            markersize=4,
            color=["royalblue", "mediumseagreen"],
        )
        for metric, plot_kwargs in panel_plot_kwargs.items():
            # Get the axis
            ax = plot_kwargs["ax"]
            # Plot the averages
            mean = means.pivot("model.latent_size", "model", metric)
            mean.plot(**plot_kwargs, **shared_plot_kwargs)
            # Plot the error
            error = errors.pivot("model.latent_size", "model", metric)
            for dyn_name, color in zip(error, shared_plot_kwargs["color"]):
                dyn_mean, dyn_error = mean[dyn_name], error[dyn_name]
                ax.fill_between(
                    dyn_mean.index,
                    dyn_mean - dyn_error,
                    dyn_mean + dyn_error,
                    color=color,
                    alpha=0.2,
                )

            # Remove title and frame from any legends
            if plot_kwargs["legend"]:
                ax.legend(title=None, frameon=False)
            # Prevent autoscaling after vlines
            ax.set_autoscale_on(False)
            # Plot vertical line at true latent dimensionality
            ax.vlines(true_dim, *ax.get_ylim(), color="k", linestyle="--", zorder=-1)
            # Remove top and right spines
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
        plt.tight_layout()
        # Ensure that the paths exist
        png_path = Path(f"png/{sys_name}")
        pdf_path = Path(f"pdf/{sys_name}")
        png_path.mkdir(exist_ok=True)
        pdf_path.mkdir(exist_ok=True)
        # Save the figures
        plt.savefig(png_path / "metrics.png", dpi=300)
        plt.savefig(pdf_path / "metrics.pdf", transparent=True)
