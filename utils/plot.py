import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os


def plot_distributio_ood(
    args,
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    tag: str,
    ood_method: str,
    score=None,
):
    sns.set(style="white", palette="muted")
    palette = ["#A8BAE3", "#55AB83"]

    data = {
        "OOD": [-1 * ood_score for ood_score in ood_scores],
        "ID": [-1 * id_score for id_score in id_scores],
    }
    sns.displot(data, label="id", kind="kde", palette=palette, fill=True, alpha=0.8)
    filename_base = f"{tag}_{ood_method}"
    if score is not None:
        plt.savefig(
            os.path.join(args.OUTPUT_DIR, f"{filename_base}_{score}.png"),
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            os.path.join(args.OUTPUT_DIR, f"{filename_base}.png"),
            bbox_inches="tight",
        )
