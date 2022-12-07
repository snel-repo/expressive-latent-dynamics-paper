# derived from lfads-torch (arsedler9)
import io

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.decomposition import PCA

from .metrics import linear_regression

plt.switch_backend("Agg")


def has_image_loggers(loggers):
    """Checks whether any image loggers are available.

    Parameters
    ----------
    loggers : obj or list[obj]
        An object or list of loggers to search.
    """
    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.TensorBoardLogger):
            return True
        elif isinstance(logger, pl.loggers.WandbLogger):
            return True
    return False


def log_figure(loggers, name, fig, step):
    """Logs a figure image to all available image loggers.

    Parameters
    ----------
    loggers : obj or list[obj]
        An object or list of loggers
    name : str
        The name to use for the logged figure
    fig : matplotlib.figure.Figure
        The figure to log
    step : int
        The step to associate with the logged figure
    """
    # Save figure image to in-memory buffer
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    image = Image.open(img_buf)
    # Distribute image to all image loggers
    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.TensorBoardLogger):
            logger.experiment.add_figure(name, fig, step)
        elif isinstance(logger, pl.loggers.WandbLogger):
            logger.log_image(name, [image], step)
    img_buf.close()


class RasterPlot(pl.Callback):
    """Plots validation spiking data side-by-side with
    inferred inputs and rates and logs to tensorboard.
    """

    def __init__(self, n_samples=2, log_every_n_epochs=20):
        """Initializes the callback.
        Parameters
        ----------
        n_samples : int, optional
            The number of samples to plot, by default 2
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 20
        """
        self.n_samples = n_samples
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.
        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return
        # Get data samples
        dataloader = trainer.datamodule.val_dataloader()
        spikes, rates, latents, _ = next(iter(dataloader))
        spikes = spikes.to(pl_module.device)
        # Compute model output
        pred_logrates, pred_latents = pl_module(spikes)
        # Convert everything to numpy
        spikes = spikes.detach().cpu().numpy()
        rates = rates.detach().cpu().numpy()
        pred_rates = torch.exp(pred_logrates).detach().cpu().numpy()
        # Create subplots
        plot_arrays = [spikes, rates, pred_rates]
        fig, axes = plt.subplots(
            len(plot_arrays),
            self.n_samples,
            sharex=True,
            sharey="row",
            figsize=(3 * self.n_samples, 10),
        )
        for i, ax_col in enumerate(axes.T):
            for ax, array in zip(ax_col, plot_arrays):
                ax.imshow(array[i].T, interpolation="none", aspect="auto")
        plt.tight_layout()
        # Log the figure
        log_figure(
            trainer.loggers,
            "raster_plot",
            fig,
            trainer.global_step,
        )


class TrajectoryPlot(pl.Callback):
    """Plots the top-3 PC's of the latent trajectory for
    all samples in the validation set and logs to tensorboard.
    """

    def __init__(self, log_every_n_epochs=100):
        """Initializes the callback.
        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.
        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return
        # Get the validation dataset
        val_dataloader = trainer.datamodule.val_dataloader()

        # Pass data through the model
        def batch_fwd(pl_module, batch):
            return pl_module(batch[0].to(pl_module.device))

        latents = [batch_fwd(pl_module, batch)[1] for batch in val_dataloader]
        latents = torch.cat(latents).detach().cpu().numpy()
        # Reduce dimensionality if necessary
        n_samp, n_step, n_lats = latents.shape
        if n_lats > 3:
            latents_flat = latents.reshape(-1, n_lats)
            pca = PCA(n_components=3)
            latents = pca.fit_transform(latents_flat)
            latents = latents.reshape(n_samp, n_step, 3)
            explained_variance = np.sum(pca.explained_variance_ratio_)
        else:
            explained_variance = 1.0
        # Create figure and plot trajectories
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for traj in latents:
            ax.plot(*traj.T, alpha=0.2, linewidth=0.5)
        ax.scatter(*latents[:, 0, :].T, alpha=0.1, s=10, c="g")
        ax.scatter(*latents[:, -1, :].T, alpha=0.1, s=10, c="r")
        ax.set_title(f"explained variance: {explained_variance:.2f}")
        plt.tight_layout()
        # Log the figure
        log_figure(
            trainer.loggers,
            "trajectory_plot",
            fig,
            trainer.global_step,
        )


class LatentRegressionPlot(pl.Callback):
    def __init__(self, n_dims=10, log_every_n_epochs=100):
        self.n_dims = n_dims
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.
        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return
        # Get the validation dataset
        val_dataloader = trainer.datamodule.val_dataloader()

        # Pass data through the model
        def batch_fwd(pl_module, batch):
            return pl_module(batch[0].to(pl_module.device))

        true_latents = torch.cat([batch[2] for batch in val_dataloader])
        true_latents = true_latents.to(pl_module.device)
        pred_latents = [batch_fwd(pl_module, batch)[1] for batch in val_dataloader]
        pred_latents = torch.cat(pred_latents)
        regr_latents = linear_regression(true_latents, pred_latents)
        # Convert latents to numpy
        pred_latents = pred_latents.detach().cpu().numpy()
        regr_latents = regr_latents.detach().cpu().numpy()

        # Reduce dimensionality if necessary
        n_samp, n_step, n_lats = pred_latents.shape
        if n_lats > 3:
            pred_latents_flat = pred_latents.reshape(-1, n_lats)
            regr_latents_flat = regr_latents.reshape(-1, n_lats)
            pca = PCA(n_components=3)
            pred_latents = pca.fit_transform(pred_latents_flat)
            explained_variance = np.sum(pca.explained_variance_ratio_)
            regr_latents = pca.transform(regr_latents_flat)
            pred_latents = pred_latents.reshape(n_samp, n_step, 3)
            regr_latents = regr_latents.reshape(n_samp, n_step, 3)
        else:
            regr_latents = regr_latents.reshape(n_samp, n_step, -1)
            explained_variance = 1.0
        # Create figure and plot trajectories
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for color, latents in zip([None, "b"], [pred_latents, regr_latents]):
            for traj in latents:
                ax.plot(*traj.T, color=color, alpha=0.2, linewidth=0.5)
            ax.scatter(*latents[:, 0, :].T, alpha=0.1, s=10, c="g")
            ax.scatter(*latents[:, -1, :].T, alpha=0.1, s=10, c="r")
        ax.set_title(f"explained variance: {explained_variance:.2f}")
        plt.tight_layout()
        # Log the figure
        log_figure(
            trainer.loggers,
            "latent_regression_plot",
            fig,
            trainer.global_step,
        )
