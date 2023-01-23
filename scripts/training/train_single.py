import os
import sys
import shutil
from datetime import datetime
from pathlib import Path

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator

from paper_src.train import train

# ---------- OPTIONS -----------
OVERWRITE = True
RUN_TAG = datetime.now().strftime("%Y%m%d-%H%M%S")
RUNS_HOME = Path("../../runs/user_runs")
RUN_DIR = RUNS_HOME / "single" / RUN_TAG
CONFIG_PATH = Path("../configs/single.yaml")
# ------------------------------

# Overwrite the directory if necessary
if RUN_DIR.exists() and OVERWRITE:
    shutil.rmtree(RUN_DIR)
RUN_DIR.mkdir(parents=True)
# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
# Switch to the `RUN_DIR` and train the model
os.chdir(RUN_DIR)
train(
    overrides={
        "datamodule": "mackeyglass100",
        "model": "node",
        "model.latent_size": 10,
        "seed": 0
    },
    config_path=CONFIG_PATH,
)
