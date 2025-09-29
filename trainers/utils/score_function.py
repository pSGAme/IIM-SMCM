import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip_w_local import clip
from clip_w_local.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
from tqdm import tqdm
from PIL import Image
from einops import repeat

to_np = lambda x: x.data.cpu().numpy()


def MCM_Score(output, T):
    smax_global = to_np(F.softmax(output / T, dim=-1))
    mcm_global_score = -np.max(smax_global, axis=1)
    return mcm_global_score


def R_MCM_Score(output, output_local, T, topk=1, neg_output_local=None):

    mcm_score = MCM_Score(output, T)
    N, C = output_local.shape[1:]
    if neg_output_local is not None:
        smax_local = torch.topk((torch.exp(output_local / T) /
                                 torch.sum(torch.exp(torch.cat((output_local, neg_output_local), dim=-1) / T), dim=-1,
                                           keepdim=True)).reshape(-1, N * C), k=topk, dim=-1)[0]
    else:
        smax_local = torch.topk((torch.exp(output_local / T) /
                                 torch.sum(torch.exp(output_local / T), dim=-1,
                                           keepdim=True)).reshape(-1, N * C), k=topk, dim=-1)[0]
    r_mcm_score = -to_np(torch.mean(smax_local, dim=1))
    r_mcm_score = r_mcm_score + mcm_score
    return r_mcm_score


def SLCM_Score(output, output_local, T, lamda, topk=1, neg_output_local=None):
    # slcm_local_score_neg = SLCM_Score(output, output_local, T, lamda, topk=top_k,
    #                                   neg_output_local=neg_output_local)
    #
    r_mcm_score = R_MCM_Score(output, output_local, T, topk, neg_output_local)

    output_local_t = output_local.permute(0, 2, 1)
    N, C = output_local_t.shape[1:]
    S1 = torch.sum(torch.exp(output_local_t / T), dim=-1, keepdim=True)
    glcm_score = torch.topk((torch.exp(output_local_t / T) / S1).reshape(-1, N * C), k=topk, dim=-1)[0]
    glcm_score = -to_np(torch.mean(glcm_score, dim=1))
    glcm_score = lamda * glcm_score + r_mcm_score

    return glcm_score

