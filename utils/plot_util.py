import seaborn as sns
from matplotlib import pyplot as plt
from  matplotlib.ticker import FormatStrFormatter
import numpy as np
import os
import pandas as pd
import torch
import seaborn as sns
import torch.nn.functional as F


def resample_to_length(arr, target_len):
    arr_sorted = np.sort(arr)
    idxs = np.linspace(0, len(arr_sorted)-1, target_len).astype(int)
    return arr_sorted[idxs]

def plot_distribution(args, id_scores, ood_scores, out_dataset, score=None):

    # target_len = min(len(id_scores), len(ood_scores))
    # id_scores = resample_to_length(id_scores, target_len)
    # ood_scores = resample_to_length(ood_scores, target_len)

    sns.set(style="white", palette="muted")
    palette = ['#A8BAE3', '#55AB83']

    data = {
        "ID": [-1 * id_score for id_score in id_scores],
        "OOD": [-1 * ood_score for ood_score in ood_scores]
    }

    # x_formatter = FormatStrFormatter('%.3f')
    plt.rcParams["xtick.labelsize"] = 12
    ax = sns.displot(data, label="id", kind="hist", palette=palette, fill=True, alpha=0.8 )
    # ax = sns.displot(data, label="id", kind="kde", palette=palette, fill=True, alpha=0.8, ax = x_formatter )
    
    
    if score is not None:
        plt.savefig(os.path.join(args.output_dir, f"{out_dataset}_{args.T}_{score}.svg"), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(args.output_dir, f"{out_dataset}_{args.T}.svg"), bbox_inches='tight')


def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center", fontsize=9) 
    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
