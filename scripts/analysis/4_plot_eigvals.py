import pickle

import matplotlib
import matplotlib.pyplot as plt
from dysts import flows

from paper_src.fixedpoints import get_sys_fps

matplotlib.rcParams.update(
    {
        "font.size": 8,
        "pdf.use14corefonts": True,
        "axes.unicode_minus": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)

# --- MANUALLY / ITERATIVELY ENSURE THAT THIS BLOCK IS CORRECT ---

# For cols
systems = ["Arneodo", "Lorenz", "Rossler"]
heights = [3, 2, 4]
true_ixs = [[1, 0], [1, 0], [1, 0]]
lowd_gru_ixs = [[0, 1], [0], [0]]
highd_gru_ixs = [[0, 1], [1, 0], [0]]
lowd_node_ixs = [[0, 1], [0, 2], [0]]
highd_node_ixs = [[1, 2], [0, 2], [0]]

# For rows
titles = ["GRU (3D)", "GRU (25D)", "NODE (3D)", "NODE (10D)"]
all_ixs = [lowd_gru_ixs, highd_gru_ixs, lowd_node_ixs, highd_node_ixs]
colors = ["royalblue", "royalblue", "mediumseagreen", "mediumseagreen"]
# Create axes
for i, (sys_name, height) in enumerate(zip(systems, heights)):
    nrows, ncols = 2, len(titles)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5, height), sharex=True, sharey=True
    )
    # Load eigenvalues
    with open(f"intermediate/{sys_name}/eigvals.pkl", "rb") as file:
        eigvals = pickle.load(file)
    eigvals["True"] = get_sys_fps(getattr(flows, sys_name)())[1]
    # Hacky because high-d GRU is only 10D for Rossler
    if "GRU (25D)" in eigvals:
        eigvals["GRU (10D)"] = eigvals["GRU (25D)"]
    else:
        eigvals["GRU (25D)"] = eigvals["GRU (10D)"]

    for j, (title, color, eig_ixs) in enumerate(zip(titles, colors, all_ixs)):
        eigs = eigvals[title]
        for k in range(2):
            ax = axes[k, j]
            ax.add_artist(plt.Circle((0, 0), 1.0, color="k", fill=False))
            ax.margins(y=0.2, x=0.2)
            ax.set_aspect("equal")
            # ax.set_xlim(-0.1, 1.25)
            # ax.set_ylim(-0.35, 0.35)
            true_eigs = get_sys_fps(getattr(flows, sys_name)())[1]
            eig_ix = true_ixs[i][k]
            ax.scatter(
                true_eigs[eig_ix].real,
                true_eigs[eig_ix].imag,
                color="deepskyblue",
                marker="x",
                s=30,
                linewidths=0.5,
                alpha=0.7,
            )
            if k + 1 <= len(eig_ixs[i]):
                eig_ix = eig_ixs[i][k]
                ax.scatter(
                    eigs[eig_ix].real,
                    eigs[eig_ix].imag,
                    color=color,
                    s=10,
                    edgecolors="k",
                    linewidths=0.5,
                    alpha=0.7,
                )
    plt.tight_layout()
    plt.savefig(f"png/{sys_name}/eigvals.png", dpi=600)
    plt.savefig(f"pdf/{sys_name}/eigvals.pdf", transparent=True)