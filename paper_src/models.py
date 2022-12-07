import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchdyn.core import NeuralODE

from .metrics import r2_score, regression_r2_score


class RNN(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, input, h_0):
        hidden = h_0
        states = []
        for input_step in input.transpose(0, 1):
            hidden = self.cell(input_step, hidden)
            states.append(hidden)
        states = torch.stack(states, dim=1)
        return states, hidden


class AbstractLatentSAE(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        encoder_size: int,
        latent_size: int,
        learning_rate: float,
        weight_decay: float,
        dropout: float,
    ):
        super().__init__()
        # Instantiate bidirectional GRU encoder
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=encoder_size,
            batch_first=True,
            bidirectional=True,
        )
        # Instantiate linear mapping to initial conditions
        self.ic_linear = nn.Linear(2 * encoder_size, latent_size)
        # Instantiate linear readout
        self.readout = nn.Linear(
            in_features=latent_size,
            out_features=input_size,
        )
        # Instantiate dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        # Pass data through the model
        _, h_n = self.encoder(data)
        # Combine output from fwd and bwd encoders
        h_n = torch.cat([*h_n], -1)
        # Compute initial condition with dropout
        h_n_drop = self.dropout(h_n)
        ic = self.ic_linear(h_n_drop)
        ic_drop = self.dropout(ic)
        return ic_drop

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def training_step(self, batch, batch_ix):

        spikes, rates, latents, _ = batch
        # Pass data through the model
        pred_logrates, pred_latents = self.forward(spikes)
        # Prepare the data for metrics
        pred_rates = torch.exp(pred_logrates)
        early_rates, mid_rates, late_rates = torch.chunk(rates, 3, dim=1)
        early_pred_rates, mid_pred_rates, late_pred_rates = torch.chunk(
            pred_rates, 3, dim=1
        )
        early_latents, mid_latents, late_latents = torch.chunk(latents, 3, dim=1)
        early_pred_latents, mid_pred_latents, late_pred_latents = torch.chunk(
            pred_latents, 3, dim=1
        )
        # Compute the results
        results = {
            "train/r2_observ": r2_score(pred_rates, rates),
            "train/r2_observ/early": r2_score(early_pred_rates, early_rates),
            "train/r2_observ/middle": r2_score(mid_pred_rates, mid_rates),
            "train/r2_observ/late": r2_score(late_pred_rates, late_rates),
            "train/r2_latent": regression_r2_score(latents, pred_latents),
            "train/r2_latent/early": regression_r2_score(
                early_latents, early_pred_latents
            ),
            "train/r2_latent/middle": regression_r2_score(
                mid_latents, mid_pred_latents
            ),
            "train/r2_latent/late": regression_r2_score(
                late_latents, late_pred_latents
            ),
        }
        self.log_dict(results)
        # Compute the weighted loss
        loss_all = F.poisson_nll_loss(
            pred_logrates, spikes, full=True, reduction="none"
        )
        return loss_all

    def validation_step(self, batch, batch_ix):

        spikes, rates, latents, _ = batch
        # Pass data through the model
        pred_logrates, pred_latents = self.forward(spikes)
        # Prepare the data for metrics
        pred_rates = torch.exp(pred_logrates)
        early_rates, mid_rates, late_rates = torch.chunk(rates, 3, dim=1)
        early_pred_rates, mid_pred_rates, late_pred_rates = torch.chunk(
            pred_rates, 3, dim=1
        )
        early_latents, mid_latents, late_latents = torch.chunk(latents, 3, dim=1)
        early_pred_latents, mid_pred_latents, late_pred_latents = torch.chunk(
            pred_latents, 3, dim=1
        )
        # Compute the results
        results = {
            "valid/r2_observ": r2_score(pred_rates, rates),
            "valid/r2_observ/early": r2_score(early_pred_rates, early_rates),
            "valid/r2_observ/middle": r2_score(mid_pred_rates, mid_rates),
            "valid/r2_observ/late": r2_score(late_pred_rates, late_rates),
            "valid/r2_latent": regression_r2_score(latents, pred_latents),
            "valid/r2_latent/early": regression_r2_score(
                early_latents, early_pred_latents
            ),
            "valid/r2_latent/middle": regression_r2_score(
                mid_latents, mid_pred_latents
            ),
            "valid/r2_latent/late": regression_r2_score(
                late_latents, late_pred_latents
            ),
        }
        self.log_dict(results)
        # Compute the weighted loss
        loss_all = F.poisson_nll_loss(
            pred_logrates, spikes, full=True, reduction="none"
        )
        return loss_all


class GRULatentSAE(AbstractLatentSAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        latent_size = self.hparams.latent_size
        self.decoder = RNN(nn.GRUCell(input_size=1, hidden_size=latent_size))

    def forward(self, data):
        ic_drop = super().forward(data)
        # Create an empty input tensor
        B, T, _ = data.shape
        input_placeholder = torch.zeros((B, T, 1), device=self.device)
        # Unroll the decoder
        latents, _ = self.decoder(input_placeholder, ic_drop)
        # Map decoder state to data dimension
        logrates = self.readout(latents)
        return logrates, latents

    def training_step(self, batch, batch_ix):
        loss_all = super().training_step(batch, batch_ix)
        loss = torch.mean(loss_all)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_ix):
        loss_all = super().validation_step(batch, batch_ix)
        loss = torch.mean(loss_all)
        self.log("valid/loss", loss)
        return loss


class NODELatentSAE(AbstractLatentSAE):
    def __init__(
        self,
        vf_hidden_size: int,
        points_per_group: int,
        epochs_per_group: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        latent_size = self.hparams.latent_size
        # Define the vector field
        vector_field = nn.Sequential(
            nn.Linear(latent_size, vf_hidden_size),
            nn.Tanh(),
            nn.Linear(vf_hidden_size, latent_size),
        )
        # Define the NeuralODE decoder and readout network
        self.decoder = NeuralODE(vector_field)

    def forward(self, data):
        ic_drop = super().forward(data)
        # Evaluate the NeuralODE
        t_span = torch.linspace(0, 1, data.shape[1])
        _, latents = self.decoder(ic_drop, t_span)
        latents = latents.transpose(0, 1)
        # Map decoder state to data dimension
        logrates = self.readout(latents)
        return logrates, latents

    def training_step(self, batch, batch_ix):
        loss_all = super().training_step(batch, batch_ix)
        total_points = loss_all.shape[1]
        # Incrementally consider more points in the loss
        group_number = int(self.current_epoch / self.hparams.epochs_per_group) + 1
        num_points = min(group_number * self.hparams.points_per_group, total_points)
        self.log("train/num_points", float(num_points))
        weight = torch.ones(total_points)
        weight[num_points:] = 0
        # Ensure the sum of the weights does not change
        weight = weight / weight.sum() * total_points
        weight = weight[None, :, None].to(self.device)
        # Compute weighted loss
        loss = torch.mean(loss_all * weight)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_ix):
        loss_all = super().validation_step(batch, batch_ix)
        total_points = loss_all.shape[1]
        # Incrementally consider more points in the loss
        group_number = int(self.current_epoch / self.hparams.epochs_per_group) + 1
        num_points = min(group_number * self.hparams.points_per_group, total_points)
        self.log("valid/num_points", float(num_points))
        weight = torch.ones(total_points)
        weight[num_points:] = 0
        # Ensure the sum of the weights does not change
        weight = weight / weight.sum() * total_points
        weight = weight[None, :, None].to(self.device)
        # Compute weighted loss
        loss = torch.mean(loss_all * weight)
        self.log("valid/loss", loss)
        return loss
