import logging
import os
from pathlib import Path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from dysts import flows

logger = logging.getLogger(__name__)


def to_tensor(array):
    return torch.tensor(array, dtype=torch.float)


class ChaoticDataModule(pl.LightningDataModule):
    def __init__(
        self,
        system: str = "Arneodo",
        obs_dim: int = 10,
        n_samples: int = 2000,
        n_timesteps: int = 70,
        pts_per_period: int = 35,
        seed: int = 0,
        batch_size: int = 650,
        data_home: str = Path(__file__).parent / "../datasets",
    ):
        super().__init__()
        self.save_hyperparameters()
        # Instantiate the dynamical system
        self.model = getattr(flows, system)()
        # Generate the dataset tag
        self.name = (
            f"{system}{obs_dim}_{n_samples}S_{n_timesteps}T_{pts_per_period}P"
            f"_{seed}seed_poisson"
        )

    def prepare_data(self):
        hps = self.hparams

        filename = f"{self.name}.h5"
        fpath = os.path.join(hps.data_home, filename)
        if os.path.isfile(fpath):
            logger.info(f"Loading dataset {self.name}")
            return
        logger.info(f"Generating dataset {self.name}")
        # Simulate the trajectories: Standardize them only if the data is to be warped
        time, latents = self.model.make_trajectory(
            n=hps.n_samples * hps.n_timesteps,
            resample=True,
            pts_per_period=hps.pts_per_period,
            return_times=True,
            standardize=False,
        )
        # Project into observation space
        latent_dim = self.model.embedding_dimension
        # Randomly sample, normalize, and sort readout
        rng = np.random.default_rng(hps.seed)
        readout = rng.uniform(-0.5, 0.5, (latent_dim, hps.obs_dim))
        readout = readout[:, np.argsort(readout[0])]
        # Project the system into the observation space
        activity = latents @ readout
        # Standardize and record original mean and standard deviations
        orig_mean = np.mean(activity, axis=0, keepdims=True)
        orig_std = np.std(activity, axis=0, keepdims=True)
        activity = (activity - orig_mean) / orig_std
        # Add noise to the observations
        activity = np.exp(activity)
        data = rng.poisson(activity).astype(float)
        # Reshape into samples
        times = time.reshape(hps.n_samples, hps.n_timesteps)
        latents = latents.reshape(hps.n_samples, hps.n_timesteps, -1)
        activity = activity.reshape(hps.n_samples, hps.n_timesteps, -1)
        data = data.reshape(hps.n_samples, hps.n_timesteps, -1)
        # Perform data splits
        inds = np.arange(hps.n_samples)
        train_inds, test_inds = train_test_split(
            inds, test_size=0.2, random_state=hps.seed
        )
        train_inds, valid_inds = train_test_split(
            train_inds, test_size=0.2, random_state=hps.seed
        )
        # Save the trajectories
        with h5py.File(fpath, "w") as h5file:
            h5file.create_dataset("train_data", data=data[train_inds])
            h5file.create_dataset("valid_data", data=data[valid_inds])
            h5file.create_dataset("test_data", data=data[test_inds])
            h5file.create_dataset("train_activity", data=activity[train_inds])
            h5file.create_dataset("valid_activity", data=activity[valid_inds])
            h5file.create_dataset("test_activity", data=activity[test_inds])
            h5file.create_dataset("train_latents", data=latents[train_inds])
            h5file.create_dataset("valid_latents", data=latents[valid_inds])
            h5file.create_dataset("test_latents", data=latents[test_inds])
            h5file.create_dataset("train_times", data=times[train_inds])
            h5file.create_dataset("valid_times", data=times[valid_inds])
            h5file.create_dataset("test_times", data=times[test_inds])
            h5file.create_dataset("train_inds", data=train_inds)
            h5file.create_dataset("valid_inds", data=valid_inds)
            h5file.create_dataset("test_inds", data=test_inds)
            h5file.create_dataset("readout", data=readout)
            h5file.create_dataset("orig_mean", data=orig_mean)
            h5file.create_dataset("orig_std", data=orig_std)

    def setup(self, stage=None):
        # Load data arrays from file
        data_path = os.path.join(self.hparams.data_home, f"{self.name}.h5")
        with h5py.File(data_path, "r") as h5file:
            # Load the data
            train_data = to_tensor(h5file["train_data"][()])
            valid_data = to_tensor(h5file["valid_data"][()])
            test_data = to_tensor(h5file["test_data"][()])
            # Load the activity
            train_activity = to_tensor(h5file["train_activity"][()])
            valid_activity = to_tensor(h5file["valid_activity"][()])
            test_activity = to_tensor(h5file["test_activity"][()])
            # Load the latents
            train_latents = to_tensor(h5file["train_latents"][()])
            valid_latents = to_tensor(h5file["valid_latents"][()])
            test_latents = to_tensor(h5file["test_latents"][()])
            # Load the indices
            train_inds = to_tensor(h5file["train_inds"][()])
            valid_inds = to_tensor(h5file["valid_inds"][()])
            test_inds = to_tensor(h5file["test_inds"][()])
            # Load other parameters
            self.orig_mean = h5file["orig_mean"][()]
            self.orig_std = h5file["orig_std"][()]
            self.readout = h5file["readout"][()]
        # Store datasets
        self.train_ds = TensorDataset(
            train_data, train_activity, train_latents, train_inds
        )
        self.valid_ds = TensorDataset(
            valid_data, valid_activity, valid_latents, valid_inds
        )
        self.test_ds = TensorDataset(
            test_data, test_activity, test_latents, test_inds
        )

    def train_dataloader(self, shuffle=True):
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
        )
        return train_dl

    def val_dataloader(self):
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
        )
        return valid_dl
