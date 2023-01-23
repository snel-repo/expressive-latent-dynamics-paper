import shutil
import sys
from datetime import datetime
from pathlib import Path

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator

from paper_src.train import train

# ---------- OPTIONS -----------
LOCAL_MODE = False
OVERWRITE = False
RUN_TAG = datetime.now().strftime("%Y%m%d") + "_my_run"
RUNS_HOME = Path("../../runs/user_runs")
RUN_DIR = RUNS_HOME / "multi" / RUN_TAG
CONFIG_PATH = Path("../configs/multi.yaml")
# ------------------------------

# Initialize the `ray` server in local mode if necessary
if LOCAL_MODE:
    ray.init(local_mode=True)
# Overwrite the directory if necessary
if RUN_DIR.exists() and OVERWRITE:
    shutil.rmtree(RUN_DIR)
RUN_DIR.mkdir(parents=True, exist_ok=True)
# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
# Run the hyperparameter search
tune.run(
    tune.with_parameters(
        train,
        config_path=CONFIG_PATH,
    ),
    metric="loss",
    mode="min",
    name=RUN_DIR.name,
    config={
        "datamodule": tune.grid_search(["arneodo", "lorenz", "rossler", "arneodo100", "mackeyglass100"]),
        "model": tune.grid_search(["gru", "node"]),
        "model.latent_size": tune.grid_search([2, 3, 5, 10, 25, 50, 100]),
        "seed": tune.grid_search([0, 1, 2, 3, 4]),
    },
    resources_per_trial=dict(cpu=2, gpu=0.3),
    num_samples=1,
    local_dir=RUN_DIR.parent,
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=FIFOScheduler(),
    verbose=1,
    progress_reporter=CLIReporter(
        metric_columns=["r2_latent", "r2_observ", "loss", "training_iteration"],
        sort_by_metric=True,
    ),
)
