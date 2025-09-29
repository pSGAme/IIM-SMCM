import argparse
import torch
from sklearn import manifold

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import numpy as np
from utils.train_eval_util import set_val_loader, set_ood_loader_ImageNet, set_model_clip
from utils.detection_util import get_and_print_results
from utils.plot_util import plot_distribution
import trainers.localprompt
import trainers.prosimo
import trainers.zeroshot
# import trainers.prosimohyper
import datasets.imagenet
import os
import seaborn as sns
from matplotlib import pyplot as plt
def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.num_neg_prompts:
        cfg.num_neg_prompts = args.num_neg_prompts

    if args.top_k:
        cfg.topk = args.top_k

    if args.T:
        cfg.T = args.T


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.LOCALPROMPT = CN()
    cfg.TRAINER.LOCALPROMPT.N_CTX = 16  # number of context vectors
    cfg.TRAINER.LOCALPROMPT.V_CTX = 8  # number of context vectors
    cfg.TRAINER.LOCALPROMPT.CSC = False  # class-specific context
    cfg.TRAINER.LOCALPROMPT.CTX_INIT = ""  # initialization words
    cfg.TRAINER.LOCALPROMPT.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.LOCALPROMPT.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.ProSimO = CN()
    cfg.TRAINER.ProSimO.N_CTX = 16  # number of context vectors
    cfg.TRAINER.ProSimO.V_CTX = 8  # number of context vectors
    cfg.TRAINER.ProSimO.CSC = False  # class-specific context
    cfg.TRAINER.ProSimO.CTX_INIT = ""  # initialization words
    cfg.TRAINER.ProSimO.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.ProSimO.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.ProSimOHyper = CN()
    cfg.TRAINER.ProSimOHyper.N_CTX = 16  # number of context vectors
    cfg.TRAINER.ProSimOHyper.V_CTX = 8  # number of context vectors
    cfg.TRAINER.ProSimOHyper.CSC = False  # class-specific context
    cfg.TRAINER.ProSimOHyper.CTX_INIT = ""  # initialization words
    cfg.TRAINER.ProSimOHyper.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.ProSimOHyper.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


to_np = lambda x: x.data.cpu().numpy()

def inter_sims(data):
    sims = []
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            sim = np.dot(data[i], data[j])
            sims.append(sim)
    return sims


def plot_imagenet1k(scores1, scores2, scores3, out_dataset="imagenet1k"):
    # plt.figure(figsize=(4, 4), dpi=200)
    plt.rcParams['font.family'] ='Times New Roman'
    sns.set(style="white", palette="muted", rc={'font.family':'Arial'})
    palette = ['#229489', '#F0AC31', "#A59DF5", "#FFD700"]

    data = {
        "ImageNet-500-Easy": scores1,
        "ImageNet-500-Hard": scores2,
        "ImageNet-1k": scores3,
    }

    # x_formatter = FormatStrFormatter('%.3f')

    plt.rcParams["xtick.labelsize"] = 12
    ax = sns.displot(data, label="id", kind="kde", palette=palette, fill=True, alpha=0.3, legend=False) # kde is approximate, you can use hist for a better xx
    # plt.grid(axis="x", color="grey", linestyle="--", linewidth=0.5, zorder=0)
    # plt.grid(axis="y", color="grey", linestyle="--", linewidth=0.5, zorder=0)
    plt.savefig(os.path.join(args.output_dir, f"{out_dataset}_microsoft_kde.svg"), bbox_inches='tight')


def tsne_plot(data, domain_name, marker, color):

    plt.scatter(data[:, 0], data[:, 1], 45, marker=marker, color=color, alpha=0.9, linewidths=0.05,
                edgecolors="white",
                label=domain_name)
    plt.grid(axis="x", color="grey", linestyle="--", linewidth=1)
    plt.grid(axis="y", color="grey", linestyle="--", linewidth=1)


def myshow():

    plt.tight_layout()
    # plt.legend(loc="upper left")
    plt.savefig(os.path.join(args.output_dir, f"t_sne.svg"), format="svg")
    plt.show()


def main(args):
    cfg = setup_cfg(args)
    # print(cfg)
    net, preprocess = set_model_clip(cfg)
    net.eval()
    net.float()

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)
    trainer.model.training = False

    outputs_val_image,  outputs_text = trainer.extract()
    # path = "/home/user/Code/Local-Prompt/output/zero_shot_part_base"
    # torch.save(outputs_val_image, os.path.join(path, "outputs_val_image"))
    # torch.save(outputs_val_image, os.path.join(path, "outputs_test_image"))
    # torch.save(outputs_val_image, os.path.join(path, "outputs_val_text"))
    # torch.save(outputs_val_image, os.path.join(path, "outputs_test_image"))

    base_text = to_np(outputs_text[0:500])
    new_text = to_np(outputs_text[500:])
    all_text = to_np(outputs_text)
    sim_base = inter_sims(base_text)
    sim_new = inter_sims(new_text)
    sim_all = inter_sims(all_text)
    print("Base Sim:")
    print(np.mean(sim_base))
    print("New Sim:")
    print(np.mean(sim_new))
    print("ALL Sim:")
    print(np.mean(sim_all))

    sim_all = sorted(sim_all)
    sim_all = sim_all[::4]
    print("ALL Sim:")
    print(np.mean(sim_all))
    print(len(sim_base), len(sim_new), len(sim_all))

    plot_imagenet1k(sim_base, sim_new, sim_all)

    tsne = manifold.TSNE(n_components=2, init='pca', n_iter=2000, random_state=42).fit_transform(all_text)
    tsne_base, tsne_new = tsne[:500], tsne[500:]
    fig = plt.figure(figsize=(4, 4), dpi=200)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    palette = ['#229489', '#F0AC31', "#A59DF5", "#FFD700"]
    tsne_plot(tsne_new, "ImageNet-500-Hard", "o", '#EE874C')
    tsne_plot(tsne_base, "ImageNet-500-Easy", "<", "#229489")

    myshow()


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument('--in_dataset', default="", type=str, help='in-distribution dataset')
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        help='mini-batch size')
    parser.add_argument('--num_neg_prompts', type=int, default=300,
                        help='number of negative local prompts')
    parser.add_argument('--T', type=float, default=1,
                        help='temperature parameter')
    parser.add_argument('--top_k', type=int, default=10,
                        help='top_k selection of regions')
    args = parser.parse_args()
    main(args)
    # os._exit(0)