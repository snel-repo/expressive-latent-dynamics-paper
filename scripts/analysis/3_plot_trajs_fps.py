import pickle
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from paper_src.fixedpoints import find_fixed_points, get_sys_fps

matplotlib.rcParams.update({"font.size": 6, "pdf.use14corefonts": True})
# Use Axes3D to avoid "not used" formatting error
assert Axes3D

RUNS_HOME = Path("../../runs")
CHECKPOINT_MAPS = {
    "Arneodo": {
        "LOWD_GRU": RUNS_HOME / "gru_arneodo" / "train_d7694_00008_8_latent_size=3,seed=1_2022-05-09_15-54-40/last.ckpt",
        "HIGHD_GRU": RUNS_HOME / "gru_arneodo" / "train_d7694_00003_3_latent_size=10,seed=0_2022-05-09_15-54-40/last.ckpt",
        "LOWD_NODE": RUNS_HOME / "node_arneodo" / "train_27524_00029_29_latent_size=3,seed=4_2022-05-09_20-05-59/last.ckpt",
        "HIGHD_NODE": RUNS_HOME / "node_arneodo" / "train_27524_00010_10_latent_size=10,seed=1_2022-05-09_15-56-50/last.ckpt",
    },
    "Rossler": {
        "LOWD_GRU": RUNS_HOME / "gru_rossler" / "train_d8fd6_00001_1_latent_size=3,seed=0_2022-05-04_21-01-13/last.ckpt",
        "HIGHD_GRU": RUNS_HOME / "gru_rossler" / "train_d8fd6_00004_4_latent_size=25,seed=0_2022-05-04_21-01-18/last.ckpt",
        "LOWD_NODE": RUNS_HOME / "node_rossler" / "train_a2b99_00001_1_latent_size=3,seed=0_2022-05-04_20-59-42/last.ckpt",
        "HIGHD_NODE": RUNS_HOME / "node_rossler" / "train_a2b99_00003_3_latent_size=10,seed=0_2022-05-04_20-59-42/last.ckpt",
    },
    "Lorenz": {
        "LOWD_GRU": RUNS_HOME / "gru_lorenz" / "train_90d5c_00001_1_latent_size=3_2022-03-16_15-59-22/last.ckpt",
        "HIGHD_GRU": RUNS_HOME / "gru_lorenz" / "train_90d5c_00004_4_latent_size=25_2022-03-16_15-59-22/last.ckpt",
        "LOWD_NODE": RUNS_HOME / "node_lorenz" / "train_81833_00001_1_latent_size=3_2022-03-16_15-58-56/last.ckpt",
        "HIGHD_NODE": RUNS_HOME / "node_lorenz" / "train_81833_00003_3_latent_size=10_2022-03-16_15-58-56/last.ckpt",
    },
}

for sys_name, ckpt_map in CHECKPOINT_MAPS.items():
    # Compose the train configs
    with hydra.initialize(config_path="../../configs"):
        config_gru = hydra.compose(
            config_name="single.yaml",
            overrides=["model=gru", f"datamodule={sys_name.lower()}"],
        )
        config_node = hydra.compose(
            config_name="single.yaml",
            overrides=["model=node", f"datamodule={sys_name.lower()}"],
        )
    # Instantiate the models
    model_gru = hydra.utils.instantiate(config_gru.model, _convert_="all")
    model_node = hydra.utils.instantiate(config_node.model, _convert_="all")
    # Load the checkpoints
    model_gru_lowd = model_gru.load_from_checkpoint(ckpt_map["LOWD_GRU"])
    model_gru_highd = model_gru.load_from_checkpoint(ckpt_map["HIGHD_GRU"])
    model_node_lowd = model_node.load_from_checkpoint(ckpt_map["LOWD_NODE"])
    model_node_highd = model_node.load_from_checkpoint(ckpt_map["HIGHD_NODE"])
    # Instantiate and setup the datamodule
    datamodule = hydra.utils.instantiate(config_gru.datamodule, _convert_="all")
    datamodule.prepare_data()
    datamodule.setup()
    # Get the data tensors
    spikes, rates, latents, _ = next(iter(datamodule.val_dataloader()))

    titles = ["Truth", "GRU (3D)", "GRU (25D)", "NODE (3D)", "NODE (10D)"]
    model_types = [None, "rnn", "rnn", "node", "node"]
    all_models = [None, model_gru_lowd, model_gru_highd, model_node_lowd, model_node_highd]
    colors = ["deepskyblue", "royalblue", "royalblue", "mediumseagreen", "mediumseagreen"]
    fig = plt.figure(figsize=(6, 2.5))
    nrows, ncols = 2, len(all_models)
    eigvals = {}
    for col, (model, fp_mode, color, title) in enumerate(
        zip(all_models, model_types, colors, titles)
    ):
        # Create axis for plotting state space
        subplot_ix = 1 + col
        ax = fig.add_subplot(nrows, ncols, subplot_ix, projection="3d")
        ax.axis("off")
        ax.set_title(title)

        if model is None:
            plot_latents = latents.detach().numpy()
            plot_fps, plot_eigvals = get_sys_fps(datamodule.model)
        else:
            model.eval()
            _, pred_latents = model(spikes)
            pred_latents = pred_latents.detach().numpy()
            B, T, N = pred_latents.shape

            fps = find_fixed_points(
                model,
                torch.tensor(pred_latents),
                mode=fp_mode,
                noise_scale=1e0,
                tol_q=1e-10,
                tol_unique=1e-1,
                device="cuda",
            )
            print(f"{fps.n} fixed points found.")

            eigvals[title] = fps.eigval_J_xstar

            if N > 3:
                pca = PCA(n_components=3)
                plot_latents = pca.fit_transform(pred_latents.reshape(-1, N))
                plot_latents = plot_latents.reshape(B, T, -1)
                plot_fps = pca.transform(fps.xstar)
                print(pca.explained_variance_ratio_)
            else:
                plot_latents = pred_latents
                plot_fps = fps.xstar

        for latent in plot_latents:
            ax.plot(*latent.T, color=color, alpha=0.2, linewidth=0.25)
        ax.set_autoscale_on(False)
        ax.scatter(*plot_fps.T, s=2, alpha=0.7, color="darkmagenta")

        if model is not None:
            # Create axis for latent projection into state space
            subplot_ix = (1 + col) + ncols
            ax = fig.add_subplot(nrows, ncols, subplot_ix, projection="3d")
            ax.axis("off")

            lin_reg = LinearRegression()
            true_latents = latents.detach().numpy().reshape(-1, 3)
            lin_reg.fit(true_latents, pred_latents.reshape(-1, N))
            mapped_latents = lin_reg.predict(true_latents)

            if N > 3:
                mapped_latents = pca.transform(mapped_latents)

            mapped_latents = mapped_latents.reshape(B, T, -1)

            for latent in plot_latents:
                ax.plot(*latent.T, color=color, alpha=0.2, linewidth=0.25)

            for latent in mapped_latents:
                ax.plot(*latent.T, color="deepskyblue", alpha=0.2, linewidth=0.25)

    # Ensure that the path exists
    int_path = Path(f"intermediate/{sys_name}")
    int_path.mkdir(exist_ok=True)
    with open(int_path / "eigvals.pkl", "wb") as file:
        pickle.dump(eigvals, file)

    plt.subplots_adjust(wspace=0, hspace=0)
    # Ensure that the paths exist
    png_path = Path(f"png/{sys_name}")
    pdf_path = Path(f"pdf/{sys_name}")
    png_path.mkdir(exist_ok=True)
    pdf_path.mkdir(exist_ok=True)
    # Save the figures
    plt.savefig(png_path / "trajs_fps.png", dpi=300)
    plt.savefig(pdf_path / "trajs_fps.pdf", transparent=True)
