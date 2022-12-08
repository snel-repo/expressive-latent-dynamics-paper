import hydra
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams.update({"font.size": 6, "pdf.use14corefonts": True})

CMAP = "Greys"
DATA_IX = 6

# Compose the train config
with hydra.initialize(config_path="../../configs", job_name="train"):
    config = hydra.compose(config_name="single.yaml", overrides=["datamodule=arneodo"])
# Instantiate and setup the datamodule
datamodule = hydra.utils.instantiate(config.datamodule, _convert_="all")
datamodule.setup()
sys_name = datamodule.model.name
# Ensure that the paths exist
png_path = Path(f"png/{sys_name}")
pdf_path = Path(f"pdf/{sys_name}")
png_path.mkdir(exist_ok=True)
pdf_path.mkdir(exist_ok=True)
# Get the data tensors
spikes, rates, latents, _ = next(iter(datamodule.val_dataloader()))
# Plot the latent states
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(0.5, 0.75))
for ax, trace in zip(axes, latents[DATA_IX].T):
    ax.plot(trace, color="deepskyblue")
    ax.axis("off")
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(png_path / "latents.png", dpi=300)
plt.savefig(pdf_path / "latents.pdf", transparent=True)
# Set up `imshow` args
vmax = max(rates[DATA_IX].max(), spikes[DATA_IX].max())
imshow_args = dict(cmap=CMAP, vmin=0, vmax=vmax, interpolation="none", aspect="auto")
# Plot the firing rates
fig, ax = plt.subplots(figsize=(1, 1))
ax.imshow(rates[DATA_IX].T, **imshow_args)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(png_path / "rates.png", dpi=300)
plt.savefig(pdf_path / "rates.pdf", transparent=True)
# Plot the spikes
fig, ax = plt.subplots(figsize=(1.2, 1))
ax_img = ax.imshow(spikes[DATA_IX].T, **imshow_args)
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(ax_img)
plt.tight_layout()
plt.savefig(png_path / "spikes.png", dpi=300)
plt.savefig(pdf_path / "spikes.pdf", transparent=True)