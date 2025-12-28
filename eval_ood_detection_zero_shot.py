import argparse
import torch
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
import datasets.fgvc_aircraft
import os


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

    if args.in_dataset in ['imagenet', 'imagenet100', 'fgvc_aircraft']:
        out_datasets = ['iNaturalist', 'SUN', 'places365', 'Texture']
    elif args.in_dataset in ['imagenet10']:
        out_datasets = ['imagenet20']
    elif args.in_dataset in ['imagenet20']:
        out_datasets = ['imagenet10']
    else:
        raise NotImplementedError('dataset not implement yet')

    trainer = build_trainer(cfg)
    trainer.model.training = False

    score_names = trainer.Info()
    num_score = len(score_names)

    auroc_list = [[] for _ in range(num_score)]
    aupr_list = [[] for _ in range(num_score)]
    fpr_list = [[] for _ in range(num_score)]

    auroc_list_mcm, aupr_list_mcm, fpr_list_mcm = [], [], []
    auroc_list_gl_mcm, aupr_list_gl_mcm, fpr_list_gl_mcm = [], [], []
    auroc_list_r_mcm, aupr_list_r_mcm, fpr_list_r_mcm = [], [], []
    auroc_list_slcm, aupr_list_slcm, fpr_list_slcm = [], [], []

    if cfg.DATASET.SUBSAMPLE_CLASSES != "all":
        in_scores = trainer.test_ood_imagenet(trainer.test_loader, args.top_k, args.T)

        print(f"Evaluating fine-grained OOD dataset Imagenet")
        out_scores = trainer.test_ood_imagenet(trainer.val_loader, args.top_k, args.T)
        # in our implementation, val_loader has different class names with the test_loader and the train_loader

        for i in range(num_score):
            print(score_names[i])
            get_and_print_results(args, in_scores[i], out_scores[i],
                                  auroc_list[i], aupr_list[i], fpr_list[i])
    else:
        id_data_loader = set_val_loader(args, preprocess)
        in_scores = trainer.test_ood(id_data_loader, args.top_k, args.T)

    for out_dataset in out_datasets:

        print(f"Evaluating course-grained OOD dataset {out_dataset}")
        ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess)

        out_scores = trainer.test_ood(ood_loader, args.top_k, args.T)

        for i in range(num_score):
            print(score_names[i])
            get_and_print_results(args, in_scores[i], out_scores[i],
                                  auroc_list[i], aupr_list[i], fpr_list[i])
            # plot_distribution(args, in_scores[i], out_scores[i], out_dataset, score=score_names[i])

    for i in range(num_score):
        print("{} avg. FPR:{}, AUROC:{}, AUPR:{}".format(score_names[i], np.mean(fpr_list[i]), np.mean(auroc_list[i]),
                                                         np.mean(aupr_list[i])))


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
