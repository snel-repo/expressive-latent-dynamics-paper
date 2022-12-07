import logging
from typing import List, Optional
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch

from .utils import flatten

log = logging.getLogger(__name__)


def train(overrides: dict, config_path: str) -> Optional[float]:
    # Format the overrides so they can be used by hydra
    config_path = Path(config_path)
    overrides = [f"{k}={v}" for k, v in flatten(overrides).items()]
    # Compose the train config
    with hydra.initialize(config_path=config_path.parent, job_name="train"):
        config = hydra.compose(config_name=config_path.name, overrides=overrides)
    # Set seed for pytorch, numpy, and python.random
    if "seed" in config:
        pl.seed_everything(config.seed, workers=True)
    # Instantiate datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        config.datamodule, _convert_="all"
    )
    # Instantiate model
    log.info(f"Instantiating model <{config.model._target_}")
    model: pl.LightningModule = hydra.utils.instantiate(config.model, _convert_="all")
    # Instantiate callbacks
    callbacks: List[pl.Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf, _convert_="all"))
    # Instantiate loggers
    logger: List[pl.LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))
    # Instantiate trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=logger,
        callbacks=callbacks,
        gpus=int(torch.cuda.is_available()),
        _convert_="all",
    )
    # Begin training
    trainer.fit(model=model, datamodule=datamodule)
