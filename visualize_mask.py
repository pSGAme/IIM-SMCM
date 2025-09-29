import argparse
import sys

import torch
from PIL import Image
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import numpy as np
from utils.train_eval_util import set_val_loader, set_ood_loader_ImageNet
from utils.detection_util import get_and_print_results
from utils.plot_util import plot_distribution
import trainers.localprompt
import trainers.prosimo
import trainers.locoop  # import trainers.prosimohyper
import datasets.imagenet
import os

def load_resize_image(image_path, new_size):
    image = Image.open(image_path)
    resized_image = image.resize(new_size)
    return resized_image, np.array(resized_image)


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

    if args.lambda_value:
        cfg.lambda_value = args.lambda_value

    if args.alpha:
        cfg.alpha = args.alpha


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

    cfg.TRAINER.LoCoOp = CN()
    cfg.TRAINER.LoCoOp.N_CTX = 16  # number of context vectors
    cfg.TRAINER.LoCoOp.CSC = False  # class-specific context
    cfg.TRAINER.LoCoOp.CTX_INIT = ""  # initialization words
    cfg.TRAINER.LoCoOp.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.LoCoOp.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

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
    import clip_w_local
    cfg = setup_cfg(args)
    _, preprocess = clip_w_local.load(cfg.MODEL.BACKBONE.NAME)

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

    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    trainer.model.training = False
    code_name = args.img_path.split("/")[-2]
    path = "/data/Classification/imagenet"
    txt_path = os.path.join(path, "classnames.txt")
    with open(txt_path, "r") as f:
        data = f.readlines()
    data = [x.split(" ")[0] for x in data]
    label = data.index(code_name)
    print(args.img_path)
    print(code_name)
    print(label)
    pos_mask, neg_mask = trainer.test_visualize(args.img_path, label)
    new_size = (224, 224)
    image, image_np = load_resize_image(args.img_path, new_size)

    new_path = args.img_path.replace("/data/Classification/imagenet/images/train/", args.output_dir+"/Visualization/")
    # folder_path = new_path.split(".")[0]
    folder_path, ext = os.path.splitext(new_path)
    os.makedirs(folder_path, exist_ok=True)


    path_origin = os.path.join(folder_path, "origin" + ext)
    path_pos = os.path.join(folder_path, "pos" + ext)
    path_neg = os.path.join(folder_path, "neg" + ext)
    path_background = os.path.join(folder_path, "background" + ext)

    print(new_path)
    print(folder_path)
    print(path_origin)
    image.save(path_origin)

    region_size= 16
    steps = 224 // 16
    steps = steps * steps

    image_np_neg = image_np.copy()
    image_np_background = image_np.copy()

    print(sorted(pos_mask.squeeze())) # 66, 67
    print(sorted(neg_mask.squeeze())) # 54
    shit = [54, 66, 67, 175, 147, 132, 118]
    for i in range(steps):
        left = (i*region_size//224)*region_size
        right = i*region_size % 224
        if i not in pos_mask:
            if i != 66 and i!= 67:
                image_np[left: left + region_size, right: right + region_size] = np.array([200, 200, 200])
        if i not in neg_mask:
            if i != 54 and i !=152 and i!=137 and i!=123 and i!=180 :
                image_np_neg[left: left + region_size, right: right + region_size] = np.array([200, 200, 200])
        if i in neg_mask or i in pos_mask or i in shit:
            image_np_background[left: left + region_size, right: right + region_size] = np.array([200, 200, 200])

    image_pos = Image.fromarray(image_np)
    image_neg = Image.fromarray(image_np_neg)
    image_background = Image.fromarray(image_np_background)
    image_pos.save(path_pos)
    image_neg.save(path_neg)
    image_background.save(path_background)


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
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument('--num_neg_prompts', type=int, default=300,
                        help='number of negative local prompts')
    parser.add_argument('--T', type=float, default=1.0,
                        help='temperature parameter')
    parser.add_argument('--top_k', type=int, default=10,
                        help='top_k selection of regions')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='topk for extracted OOD regions')
    parser.add_argument('--lambda_value', type=float, default=5,
                        help='weight for negative ')
    parser.add_argument('--img_path', type=str, default="/data/Classification/imagenet/images/train/n02090721/n02090721_1044.JPEG")
    args = parser.parse_args()
    main(args)
    #sys.exit()
    os._exit(0)
