import copy
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
import math
from functools import reduce
from operator import mul
import os
from .utils.score_function import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

_tokenizer = _Tokenizer()
softmax = nn.Softmax(dim=1).cuda()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model.cuda().eval()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _, _, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        # (classnames)
        classnames = [name.replace("_", " ") for name in classnames]
        print(classnames)
        global_prompts = ["a photo of a" + " " + name + "" for name in classnames]
        global_tokenized_prompts = torch.cat([clip.tokenize(p).cuda() for p in global_prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(global_tokenized_prompts).type(dtype)

        self.global_embedding = embedding
        self.global_tokenized_prompts = global_tokenized_prompts  # torch.Tensor  #1000,77

    def forward(self):
        return self.global_embedding


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.global_tokenized_prompts = self.prompt_learner.global_tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def simple_extract(self, images):
        global_prompts = self.prompt_learner()
        global_tokenized_prompts = self.global_tokenized_prompts

        global_text_features = self.text_encoder(global_prompts, global_tokenized_prompts)
        image_features, local_image_features = self.image_encoder(images.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # print(image_features[0][0:10])

        local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)
        global_text_features = global_text_features / global_text_features.norm(dim=-1, keepdim=True)
        return image_features, global_text_features

    def forward(self, images):

        global_prompts = self.prompt_learner()
        global_tokenized_prompts = self.global_tokenized_prompts

        global_text_features = self.text_encoder(global_prompts, global_tokenized_prompts)
        image_features, local_image_features = self.image_encoder(images.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
       # print(image_features[0][0:10])

        local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)
        global_text_features = global_text_features / global_text_features.norm(dim=-1, keepdim=True)
        # print(global_text_features[0][0:10])

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ global_text_features.t()
        logits_local = logit_scale * local_image_features @ global_text_features.t()


        return logits, logits_local


@TRAINER_REGISTRY.register()
class ZeroShot(TrainerX):
    """
    Extensible Local Prompts for Few-Shot Out-of-Distribution Detection (LOCAL-PROMPT).
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.ProSimO.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.top_k = cfg.topk
        self.T = cfg.T

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.ProSimO.PREC == "fp32" or cfg.TRAINER.ProSimO.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.model.to(self.device)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    # be sure to use it in the visualize_imagenet1k.py
    # be sure the part be base or new
    # by default, we set part = base
    @torch.no_grad()
    def extract(self):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        outputs_val_image = []
        outputs_test_image = []
        outputs_text = None

        print("Extracting on the *val (class different)* set")
        for batch_idx, batch in enumerate(tqdm(self.val_loader)): # whatever loader is ok, i just want the text
            input, label = self.parse_batch_test(batch)
            output_val_image, outputs_text = self.model.simple_extract(input)
            outputs_val_image.append(output_val_image)
            break

        return torch.cat(outputs_val_image, dim=0),  outputs_text

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT  # test

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        list_correct = []
        outputs = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output_global, output_local, _, _, _ = self.model_inference(input)
            # output_global, output_local, _ = self.model.forward_custom(input)

            output_global /= 100.0
            output_local /= 100.0

            local_score = torch.topk(torch.exp(output_local / self.T), k=self.top_k, dim=1)[0]
            output = torch.exp(output_global) * torch.mean(local_score, dim=1)

            outputs.append(F.softmax(output, dim=-1).data.cpu().numpy())
            pred = output.max(dim=1)[1]
            for j in range(len(pred)):
                if pred[j] == label[j]:
                    cor = 1
                else:
                    cor = 0
                list_correct.append(cor)

            if len(output) == 2:
                output = output[0]
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0], np.concatenate(outputs, axis=0), list_correct


    @torch.no_grad()
    def test_ood(self, data_loader, top_k, T):
        """Test-time OOD detection pipeline."""
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)

        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        mcm_score = []
        gl_mcm_score = []
        r_mcm_score = []
        slcm_score = []

        for batch_idx, (images, labels, *id_flag) in enumerate(tqdm(data_loader)):
            images = images.cuda()

            output, output_local = self.model_inference(images)
            output /= 100.0
            output_local /= 100.0


            mcm_global_score = MCM_Score(output, T)
            mcm_score.append(mcm_global_score)

            #gl-mcm
            N, C = output_local.shape[1:]
            S1 = torch.sum(torch.exp(output_local / T), dim=-1, keepdim=True)
            smax_local = torch.topk((torch.exp(output_local / T) / (S1)).reshape(-1, N * C), k=1, dim=-1)[0]
            gl_mcm_local_score = -to_np(torch.mean(smax_local, dim=1))
            gl_mcm_score.append(mcm_global_score + gl_mcm_local_score)


            # r-mcm
            N, C = output_local.shape[1:]
            S1 = torch.sum(torch.exp(output_local / T), dim=-1, keepdim=True)
            smax_local = torch.topk((torch.exp(output_local / T) / (S1)).reshape(-1, N * C), k=top_k, dim=-1)[0]
            r_mcm_local_score = -to_np(torch.mean(smax_local, dim=1))
            r_mcm_score.append(mcm_global_score + r_mcm_local_score)

            #glcm
            output_local_t = output_local.permute(0, 2, 1)
            N, C = output_local_t.shape[1:]
            S1 = torch.sum(torch.exp(output_local_t / T), dim=-1, keepdim=True)
            smax_local_t = torch.topk((torch.exp(output_local_t / T) / S1).reshape(-1, N * C), k=top_k, dim=-1)[0]
            mcm_local_score_t = -to_np(torch.mean(smax_local_t, dim=1))
            slcm_score.append(mcm_global_score + 1.0 * gl_mcm_local_score + 0.2 * mcm_local_score_t)

        return concat(mcm_score)[:len(data_loader.dataset)].copy(), \
               concat(gl_mcm_score)[:len(data_loader.dataset)].copy(), \
               concat(r_mcm_score)[:len(data_loader.dataset)].copy(), \
               concat(slcm_score)[:len(data_loader.dataset)].copy()

    @torch.no_grad()
    def test_ood_imagenet(self, data_loader, top_k, T):
        """Test-time OOD detection pipeline."""
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)

        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        mcm_score = []
        gl_mcm_score = []
        r_mcm_score = []
        slcm_score = []

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            images, labels = self.parse_batch_test(batch)

            output, output_local = self.model_inference(images)
            output /= 100.0
            output_local /= 100.0

            smax_global = to_np(F.softmax(output / T, dim=-1))
            mcm_global_score = -np.max(smax_global, axis=1)
            mcm_score.append(mcm_global_score)

            # gl-mcm
            N, C = output_local.shape[1:]
            S1 = torch.sum(torch.exp(output_local / T), dim=-1, keepdim=True)
            smax_local = torch.topk((torch.exp(output_local / T) / (S1)).reshape(-1, N * C), k=1, dim=-1)[0]
            gl_mcm_local_score = -to_np(torch.mean(smax_local, dim=1))
            gl_mcm_score.append(mcm_global_score + gl_mcm_local_score)

            # r-mcm
            N, C = output_local.shape[1:]
            S1 = torch.sum(torch.exp(output_local / T), dim=-1, keepdim=True)
            smax_local = torch.topk((torch.exp(output_local / T) / (S1)).reshape(-1, N * C), k=top_k, dim=-1)[0]
            r_mcm_local_score = -to_np(torch.mean(smax_local, dim=1))
            r_mcm_score.append(mcm_global_score + r_mcm_local_score)

            # glcm
            output_local_t = output_local.permute(0, 2, 1)
            N, C = output_local_t.shape[1:]
            S1 = torch.sum(torch.exp(output_local_t / T), dim=-1, keepdim=True)
            smax_local_t = torch.topk((torch.exp(output_local_t / T) / S1).reshape(-1, N * C), k=top_k, dim=-1)[0]
            mcm_local_score_t = -to_np(torch.mean(smax_local_t, dim=1))
            slcm_score.append(mcm_global_score + 1.0 * gl_mcm_local_score + 0.2 * mcm_local_score_t)

        return concat(mcm_score)[:len(data_loader.dataset)].copy(), \
               concat(gl_mcm_score)[:len(data_loader.dataset)].copy(), \
               concat(r_mcm_score)[:len(data_loader.dataset)].copy(), \
               concat(slcm_score)[:len(data_loader.dataset)].copy()