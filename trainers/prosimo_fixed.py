import copy
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

import clip_w_local
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
        n_cls = len(classnames)
        self.num_neg_prompts = cfg.num_neg_prompts
        self.num_local_prompts = n_cls
        n_ctx = cfg.TRAINER.ProSimO.N_CTX
        ctx_init = cfg.TRAINER.ProSimO.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # for global prompt initialization: frozen and hand-crafted 'a photo of {c}'
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            global_ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.ProSimO.CSC:
                print("Initializing class-specific contexts")
                global_ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                global_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(global_ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial global prompt context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        global_prompts = ["a photo of a" + " " + name + "." for name in classnames]

        global_tokenized_prompts = torch.cat([clip.tokenize(p).cuda() for p in global_prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(global_tokenized_prompts).type(dtype)

        self.global_embedding = embedding
        self.global_tokenized_prompts = global_tokenized_prompts  # torch.Tensor  #1000,77
        self.class_token_position = cfg.TRAINER.ProSimO.CLASS_TOKEN_POSITION

        # for local prompt initialization: learnable
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            local_ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.ProSimO.CSC:
                print("Initializing class-specific contexts")
                local_ctx_vectors = torch.empty(self.num_local_prompts, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(local_ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial local context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.local_ctx = nn.Parameter(local_ctx_vectors)  # to be optimized

        local_prompts = [prompt_prefix + " " + name + "." for name in classnames]
        local_tokenized_prompts = torch.cat([clip.tokenize(p).cuda() for p in local_prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(local_tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.local_tokenized_prompts = local_tokenized_prompts

        # for local prompt initialization: learnable and random initialization
        print("Initializing negative local contexts")
        neg_ctx_vectors = torch.empty(self.num_neg_prompts, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(neg_ctx_vectors, std=0.02)
        neg_prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial local context: "{neg_prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.neg_ctx = nn.Parameter(neg_ctx_vectors)  # to be optimized

        neg_prompts = [neg_prompt_prefix + " " + "." for _ in range(self.num_neg_prompts)]
        neg_tokenized_prompts = torch.cat([clip.tokenize(p).cuda() for p in neg_prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(neg_tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names

        self.register_buffer("neg_token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("neg_token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.neg_tokenized_prompts = neg_tokenized_prompts

        ## newly appended by ckx
        self.v_ctx = cfg.TRAINER.ProSimO.V_CTX
        self.visual_ctx = None
        self.visual_ctx_neg = None
        if self.v_ctx > 0:
            conv1 = clip_model.visual.conv1
            width = conv1.out_channels
            patch_size = conv1.kernel_size
            prompt_dim = width
            scale = width ** -0.5
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
            self.visual_ctx = nn.Parameter(scale * torch.randn(1, self.v_ctx, width))
            self.visual_ctx_neg = nn.Parameter(scale * torch.randn(1, self.v_ctx, width))

    def forward(self):
        assert self.class_token_position == 'end', 'not expected class token position.'

        local_ctx = self.local_ctx  # 100,16,512
        if local_ctx.dim() == 2:
            local_ctx = local_ctx.unsqueeze(0).expand(self.num_ood_prompts, -1, -1)

        prefix = self.token_prefix  # 100,1,512
        suffix = self.token_suffix  # 1000,60,512

        local_prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                local_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        if local_ctx.dim() == 2:
            local_ctx = local_ctx.unsqueeze(0).expand(self.num_neg_prompts, -1, -1)

        neg_prefix = self.neg_token_prefix  # 100,1,512
        neg_suffix = self.neg_token_suffix  # 1000,60,512

        neg_prompts = torch.cat(
            [
                neg_prefix,  # (n_cls, 1, dim)
                self.neg_ctx,  # (n_cls, n_ctx, dim)
                neg_suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return self.visual_ctx, self.visual_ctx_neg, self.global_embedding, local_prompts, neg_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.global_tokenized_prompts = self.prompt_learner.global_tokenized_prompts
        self.local_tokenized_prompts = self.prompt_learner.local_tokenized_prompts
        self.neg_tokenized_prompts = self.prompt_learner.neg_tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.zs_clip_model = copy.deepcopy(clip_model)
        self.zs_text_encoder = TextEncoder(self.zs_clip_model)

    def multi_loader_select(self, images, label):
        '''
        Global Prompt Guided Negative Augmentation
        select images according to similarity between global image features label texts.
        '''

        # global_text_features = self.zs_text_encoder(global_prompts, global_tokenized_prompts)
        # local_text_features = self.text_encoder(local_prompts, local_tokenized_prompts)
        # neg_text_features = self.text_encoder(neg_prompts, neg_tokenized_prompts)
        #
        # image_features, _ = self.zs_clip_model.visual(images.type(self.dtype))
        # _, local_image_features = self.image_encoder(images.type(self.dtype))

        with torch.no_grad():
            image_features, local_image_features = [], []
            similarity_list = []

            for image in images:
                image_feature, local_image_feature = self.image_encoder(image.type(self.dtype))
                image_features.append(image_feature)
                local_image_features.append(local_image_feature)

            visual_prompts, _, global_prompts, _, _ = self.prompt_learner()
            global_tokenized_prompts = self.global_tokenized_prompts
            global_text_features = self.text_encoder(global_prompts, global_tokenized_prompts)
            global_text_features = global_text_features / global_text_features.norm(dim=-1, keepdim=True)

            image_features = [image_feature / image_feature.norm(dim=-1, keepdim=True) for image_feature in
                              image_features]

            # print(len(image_features), image_features[0].shape)  # 16, [256, 512], [256, 196, 512]

            global_text_selected_label = global_text_features.gather(0, label[..., None].expand_as(image_features[0]))

            for image_feature in image_features:
                similarity_list.append(
                    torch.nn.functional.cosine_similarity(image_feature, global_text_selected_label, dim=-1))

            image_features = torch.stack(image_features, dim=0)  # 16, 256, 512
            local_image_features = torch.stack(local_image_features, dim=0)  # 16, 256, 196, 512
            similarity_list = torch.stack(similarity_list, dim=0)  # 16 256
            # print(image_features.shape, local_image_features.shape, similarity_list.shape)
            return similarity_list, image_features, local_image_features

    def forward(self, images, image_features=None, local_image_features=None, max_list=None, min_list=None):
        # image_features = torch.stack(image_features, dim=0)  # 16, 256, 512
        # local_image_features = torch.stack(local_image_features, dim=0)  # 16, 256, 196, 512
        # similarity_list = torch.stack(similarity_list, dim=0)  # 16 256
        if self.training:
            num_region, dimension = local_image_features.shape[-2:]

            visual_prompts, neg_visual_prompts, _, local_prompts, neg_prompts = self.prompt_learner()

            local_tokenized_prompts = self.local_tokenized_prompts
            neg_tokenized_prompts = self.neg_tokenized_prompts

            local_text_features = self.text_encoder(local_prompts, local_tokenized_prompts)
            neg_text_features = self.text_encoder(neg_prompts, neg_tokenized_prompts)

            # print(max_list.shape) # 1, 256
            with torch.no_grad():
                pos_local_image_features = local_image_features.gather(0,
                                                                       repeat(max_list, 'q b -> q b n c', n=num_region,
                                                                              c=dimension)).squeeze()
                # neg_local_image_features = local_image_features.gather(0,
                #                                                        repeat(min_list, 'q b -> q b n c', n=num_region,
                #                                                               c=dimension)).squeeze()

            # B, C, H, W = images[0].shape
            # device = images[0].device
            # pos = torch.empty(B, C, H, W, device=device)
            # # neg = torch.empty(B, C, H, W, device=device)

            # for i in range(B):
            #     # image_features = torch.stack(image_features, dim=0)  # 16, 256, 512
            #
            #     # max_list : 1, 256
            #     # iamges: 16, 256, c, h, w\
            #     pos[i] = images[max_list[0][i]][i]
               # neg[i] = images[min_list[0][i]][i]
                # print(pos.shape)
            # with torch.no_grad():
            #     _, pos_local_image_features = self.image_encoder(pos.type(self.dtype), visual_prompts)
            # neg_global_image_features, neg_local_image_features = self.image_encoder(neg.type(self.dtype), neg_visual_prompts)


            pos_local_image_features = pos_local_image_features / pos_local_image_features.norm(dim=-1, keepdim=True)
            # neg_local_image_features = neg_local_image_features / neg_local_image_features.norm(dim=-1, keepdim=True)
            neg_local_image_features = pos_local_image_features

            local_text_features = local_text_features / local_text_features.norm(dim=-1, keepdim=True)
            neg_text_features = neg_text_features / neg_text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits_local = logit_scale * pos_local_image_features @ local_text_features.t()
            p2n_logits_local = logit_scale * neg_local_image_features @ local_text_features.t()
            n2p_logits_local = logit_scale * pos_local_image_features @ neg_text_features.t()
            neg_logits_local = logit_scale * neg_local_image_features @ neg_text_features.t()

            # for diversity regularization
            loss_div = torch.nn.functional.cosine_similarity(neg_text_features[None, :, :],
                                                             neg_text_features[:, None, :], dim=-1)

            loss_div = torch.sum(loss_div, dim=-1) / self.prompt_learner.num_neg_prompts
            loss_div = torch.sum(loss_div, dim=-1) / (self.prompt_learner.num_neg_prompts - 1)

            return logits_local, p2n_logits_local, n2p_logits_local, neg_logits_local, loss_div

        else:  # for inference
            visual_prompts, neg_visual_prompts, global_prompts, local_prompts, neg_prompts = self.prompt_learner()

            global_tokenized_prompts = self.global_tokenized_prompts
            local_tokenized_prompts = self.local_tokenized_prompts
            neg_tokenized_prompts = self.neg_tokenized_prompts

            global_text_features = self.zs_text_encoder(global_prompts, global_tokenized_prompts)
            # global_text_features = self.text_encoder(global_prompts, global_tokenized_prompts)
            local_text_features = self.text_encoder(local_prompts, local_tokenized_prompts)
            neg_text_features = self.text_encoder(neg_prompts, neg_tokenized_prompts)

            image_features, _ = self.zs_clip_model.visual(images.type(self.dtype))
            # image_features, local_image_features = self.image_encoder(images.type(self.dtype))
            _, local_image_features = self.image_encoder(images.type(self.dtype))
            # _, local_image_features_vp = self.image_encoder(images.type(self.dtype), visual_prompts)
            # local image features are much more robust with the effect of the visual prompts.

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)
            # local_image_features_vp = local_image_features_vp / local_image_features_vp.norm(dim=-1, keepdim=True)

            global_text_features = global_text_features / global_text_features.norm(dim=-1, keepdim=True)
            local_text_features = local_text_features / local_text_features.norm(dim=-1, keepdim=True)
            neg_text_features = neg_text_features / neg_text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()

            logits = logit_scale * image_features @ global_text_features.t()
            logits_local = logit_scale * local_image_features @ local_text_features.t()
            # logits_local_vp = logit_scale * local_image_features_vp @ local_text_features.t()
            neg_logits_local = logit_scale * local_image_features @ neg_text_features.t()
            # neg_logits_local_vp = logit_scale * local_image_features_vp @ neg_text_features.t()

            return logits, logits_local, neg_logits_local


@TRAINER_REGISTRY.register()
class ProSimO(TrainerX):
    """
    ProSimO
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

        print("Turning off gradients in both the image and the text encoder")
        train_parameters = []
        train_parameters.append('text_encoder.text_projection')
        train_parameters.append('image_encoder.proj')
        train_parameters.append('prompt_learner')

        flag = 0
        for name, param in self.model.named_parameters():
            # if "prompt_learner" not in name and name not in train_parameters:
            #     param.requires_grad_(False)
            for str in train_parameters:
                flag = 0
                if name.startswith(str):
                    param.requires_grad_(True)
                    flag = 1
                    print(name)
                    break
            if flag == 0:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.ProSimO.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def calculate_loss_local(self, output_local, p2n_output_local, label):
        # print(output_local.shape)

        total_local_prompts = self.model.prompt_learner.num_local_prompts + self.model.prompt_learner.num_neg_prompts
        # batch,n,c ->  batch,n,1 -> batch,k,1 -> batch,k
        label_output_local = output_local.gather(2, repeat(label, 'b -> b n 1', n=output_local.shape[1]))  # batch,n,1
        label_topk_output_local_pos = label_output_local.topk(k=self.top_k, dim=1)[1]  # batch,k,1 :index
        label_topk_output_local = label_output_local.gather(1, label_topk_output_local_pos).squeeze()  # batch, k

        # the above is more general!
        # here is a more precise style:
        # label_topk_output_local = label_output_local.topk(k=self.top_k, dim=1)[0]

        pos_topk_ouput_local = output_local.topk(k=self.top_k, dim=1)[0]  # batch, k, c
        neg_topk_output_local = p2n_output_local.topk(k=self.top_k, dim=1)[0]  # batch, k, c_neg

        # To prevent overflow. for simplicity, divide both the numerator and denominator by the first number
        if len(label_topk_output_local.shape) == 1:  # avoid the case k=1
            label_topk_output_local = label_topk_output_local.unsqueeze(1)

        common_factor = label_topk_output_local[:, 0] # batch

        label_topk_output_local[:, 0:1].expand_as(label_topk_output_local)

        local_contrastive = torch.sum(
            torch.exp((label_topk_output_local - repeat(common_factor, 'b-> b k', k=self.top_k)) / self.T), dim=-1) / \
                            torch.sum(torch.exp((torch.cat((pos_topk_ouput_local, neg_topk_output_local),
                                                           dim=-1).reshape(-1,
                                                                           self.top_k * total_local_prompts) - repeat(
                                common_factor, 'b-> b k', k=self.top_k * total_local_prompts)) / self.T), dim=-1)

        loss_local = -torch.mean(torch.log(local_contrastive))
        return loss_local


    def calculate_loss_local_bad(self, output_local, p2n_output_local, label):

        total_local_prompts = self.model.prompt_learner.num_local_prompts
        # batch,n,c ->  batch,n,1 -> batch,k,1 -> batch,k
        label_output_local = output_local.gather(2, repeat(label, 'b -> b n 1', n=output_local.shape[1]))  # batch,n,1
        label_topk_output_local_pos = label_output_local.topk(k=self.top_k, dim=1)[1]  # batch,k,1 :index
        label_topk_output_local = label_output_local.gather(1, label_topk_output_local_pos).squeeze()  # batch, k

        # the above is more general!
        # here is a more precise style:
        # label_topk_output_local = label_output_local.topk(k=self.top_k, dim=1)[0]

        pos_topk_ouput_local = output_local.topk(k=self.top_k, dim=1)[0]  # batch, k, c
        neg_topk_output_local = p2n_output_local.topk(k=self.top_k, dim=1)[0]  # batch, k, c_neg

        # To prevent overflow. for simplicity, divide both the numerator and denominator by the first number
        if len(label_topk_output_local.shape) == 1:  # avoid the case k=1
            label_topk_output_local = label_topk_output_local.unsqueeze(1)

        common_factor = label_topk_output_local[:, 0] # batch

        label_topk_output_local[:, 0:1].expand_as(label_topk_output_local)

        local_contrastive = torch.sum(
            torch.exp((label_topk_output_local - repeat(common_factor, 'b-> b k', k=self.top_k)) / self.T), dim=-1) / \
                            torch.sum(torch.exp((pos_topk_ouput_local.reshape(-1,
                                                                           self.top_k * total_local_prompts) - repeat(
                                common_factor, 'b-> b k', k=self.top_k * total_local_prompts)) / self.T), dim=-1)

        loss_local = -torch.mean(torch.log(local_contrastive))
        return loss_local

    def calculate_loss_local_neg(self, output_local, p2n_output_local, label):
        #
        # loss_local_negative = self.calculate_loss_local_neg(neg_output_local, p2n_output_local, label)
        '''
        output_local is now local_negative_prompts, so the topk positional should be determined by p2n_output_local
        '''

        #print(output_local.shape, p2n_output_local.shape, len(label))
        # 4, 196, 300; 4, 196, 500; 5
        total_local_prompts = self.model.prompt_learner.num_local_prompts + self.model.prompt_learner.num_neg_prompts

        label_output_local = p2n_output_local.gather(2, repeat(label, 'b -> b n 1', n=output_local.shape[1]))
        #print(label_output_local.shape)
        # 4, 196, 1
        label_topk_output_local_pos = label_output_local.topk(k=self.top_k * 2, dim=1)[1]
        #print(label_topk_output_local_pos.shape)  # indices
        label_topk_output_local_pos = label_topk_output_local_pos[:, self.top_k:2 * self.top_k]
        #print(label_topk_output_local_pos.shape) # indices
        # 4, 50, 1
        pos_topk_ouput_local = output_local.gather(1, repeat(label_topk_output_local_pos, 'b k 1 -> b k n',
                                                             n=self.model.prompt_learner.num_neg_prompts))
        #print(pos_topk_ouput_local.shape)
        # 4, 50, 300
        neg_topk_output_local = p2n_output_local.topk(k=self.top_k * 2, dim=1)[0]
        neg_topk_output_local = neg_topk_output_local[:, self.top_k:2 * self.top_k]
        # print(neg_topk_output_local.shape)
        # 4, 50, 500

        # To prevent overflow. for simplicity, divide both the numerator and denominator by the first number
        t = torch.cat((pos_topk_ouput_local, neg_topk_output_local),
                                                           dim=-1)

        common_factor = pos_topk_ouput_local[:, 0, 0]
        common_factor = torch.amax(t, dim=[1,2], keepdim=True) # b, k, 1
        # print(common_factor.shape)

        local_contrastive = torch.sum(torch.exp((pos_topk_ouput_local - repeat(common_factor, 'b 1 1 -> b k n',
                                                                               k=self.top_k,
                                                                               n=self.model.prompt_learner.num_neg_prompts)) / self.T),
                                      dim=(1, 2)) / \
                            torch.sum(torch.exp((torch.cat((pos_topk_ouput_local, neg_topk_output_local),
                                                           dim=-1) - repeat(common_factor, 'b 1 1 -> b k n ', k=self.top_k,
                                                                            n=total_local_prompts)) / self.T),
                                      dim=(1, 2))

        local_contrastive = torch.log(local_contrastive)
        nan_mask = local_contrastive.isfinite()
        if nan_mask.any():
            loss_local_neg = -torch.mean(local_contrastive[nan_mask])
        else:
            return None

        #loss_local_neg = -torch.mean(local_contrastive)

        return loss_local_neg

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.ProSimO.PREC
        num_pos = self.cfg.num_pos
        self.lambda_value = self.cfg.lambda_value
        self.div_value = self.cfg.div_value
        if prec == "amp":

            # image_features = torch.stack(image_features, dim=0)  # 16, 256, 512
            # local_image_features = torch.stack(local_image_features, dim=0)  # 16, 256, 196, 512
            # similarity_list = torch.stack(similarity_list, dim=0)  # 16 256
            #
            similarity_list, image_features, local_image_features = self.model.multi_loader_select(image, label)

            #
            max_list, min_list = torch.topk(similarity_list, k=num_pos, dim=0)[1], \
                                 torch.topk(similarity_list, k=num_pos, largest=False, dim=0)[1]
            # print(max_list.shape, min_list.shape)
            # 8 256, 8 256
            for i in range(num_pos):
                with autocast():
                    output_local, p2n_output_local, n2p_output_local, neg_output_local, loss_div = self.model(image,
                                                                                                              image_features,
                                                                                                              local_image_features,
                                                                                                              max_list[
                                                                                                              i:i + 1,
                                                                                                              :],
                                                                                                              min_list[
                                                                                                              0:1, :])


                    loss_local = self.calculate_loss_local(output_local, n2p_output_local, label)
                    loss_local_negative = self.calculate_loss_local_neg(neg_output_local, p2n_output_local, label)

                    loss = loss_local + self.lambda_value * loss_local_negative + self.div_value * loss_div
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
        else:
            raise NotImplementedError('fp32 easily falls into oom and fp16 suffers from nan loss. Should be amp')

        loss_summary = {
            "loss": loss.item(),
            "loss_local": loss_local.item(),
            "loss_local_negative": loss_local_negative.item() if loss_local_negative else 0,
            "loss_div": loss_div.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, total_batch):
        num = 1
        # get number of random_crop
        for key in total_batch.keys():
            if 'img' in key:
                num = max(num, int(key[3:])) if num else 1

        inputs = []
        for i in range(num):
            inputs.append(total_batch["img" + str(i + 1)].to(self.device))
        label = total_batch["label"].to(self.device)
        return inputs, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            # print(state_dict.keys())
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print('Successfully load global weights from pretrained model.')

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


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
            output_global, output_local, _,  = self.model_inference(input)
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
        r_mcm_score_neg = []
        slcm_score_neg = []

        for batch_idx, (images, labels, *id_flag) in enumerate(tqdm(data_loader)):
            images = images.cuda()

            output, output_local, neg_output_local = self.model_inference(images)
            output /= 100.0
            output_local /= 100.0
            neg_output_local /= 100.0

            # mcm
            mcm_global_score = MCM_Score(output, T)
            mcm_score.append(mcm_global_score)

            # gl_mcm
            gl_mcm_local_score = R_MCM_Score(output, output_local, T, topk=1)
            gl_mcm_score.append(gl_mcm_local_score)

            # r_mcm
            r_mcm_local_score = R_MCM_Score(output, output_local, T, topk=top_k)
            r_mcm_score.append(r_mcm_local_score)

            # r_mcm_neg
            r_mcm_local_score_neg = R_MCM_Score(output, output_local, T, topk=top_k, neg_output_local=neg_output_local)
            r_mcm_score_neg.append(r_mcm_local_score_neg)

            # slcm
            alpha = self.cfg.alpha
            lamda = alpha * 196 / (self.model.prompt_learner.num_local_prompts + self.model.prompt_learner.num_neg_prompts / 3)

            slcm_local_score = SLCM_Score(output, output_local, T, lamda, topk=top_k)
            slcm_score.append(slcm_local_score)

            # slcm_neg
            slcm_local_score_neg = SLCM_Score(output, output_local, T, lamda, topk=top_k,
                                              neg_output_local=neg_output_local)
            slcm_score_neg.append(slcm_local_score_neg)

        return concat(mcm_score)[:len(data_loader.dataset)].copy(), \
               concat(gl_mcm_score)[:len(data_loader.dataset)].copy(), \
               concat(r_mcm_score)[:len(data_loader.dataset)].copy(), \
               concat(r_mcm_score_neg)[:len(data_loader.dataset)].copy(), \
               concat(slcm_score)[:len(data_loader.dataset)].copy(), \
               concat(slcm_score_neg)[:len(data_loader.dataset)].copy()

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
        r_mcm_score_neg = []
        slcm_score_neg = []

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            images, labels = self.parse_batch_test(batch)

            output, output_local, neg_output_local = self.model_inference(images)
            output /= 100.0
            output_local /= 100.0
            neg_output_local /= 100.0

            # mcm
            mcm_global_score = MCM_Score(output, T)
            mcm_score.append(mcm_global_score)

            # gl_mcm
            gl_mcm_local_score = R_MCM_Score(output, output_local, T, topk=1)
            gl_mcm_score.append(gl_mcm_local_score)

            # r_mcm
            r_mcm_local_score = R_MCM_Score(output, output_local, T, topk=top_k)
            r_mcm_score.append(r_mcm_local_score)

            # r_mcm_neg
            r_mcm_local_score_neg = R_MCM_Score(output, output_local, T, topk=top_k, neg_output_local=neg_output_local)
            r_mcm_score_neg.append(r_mcm_local_score_neg)

            # slcm
            alpha = self.cfg.alpha
            lamda = alpha * 196 / (self.model.prompt_learner.num_local_prompts + self.model.prompt_learner.num_neg_prompts / 3)

            slcm_local_score = SLCM_Score(output, output_local, T, lamda, topk=top_k)
            slcm_score.append(slcm_local_score)

            # slcm_neg
            slcm_local_score_neg = SLCM_Score(output, output_local, T, lamda, topk=top_k,
                                              neg_output_local=neg_output_local)
            slcm_score_neg.append(slcm_local_score_neg)

        return concat(mcm_score)[:len(data_loader.dataset)].copy(), \
               concat(gl_mcm_score)[:len(data_loader.dataset)].copy(), \
               concat(r_mcm_score)[:len(data_loader.dataset)].copy(), \
               concat(r_mcm_score_neg)[:len(data_loader.dataset)].copy(), \
               concat(slcm_score)[:len(data_loader.dataset)].copy(), \
               concat(slcm_score_neg)[:len(data_loader.dataset)].copy()

    @torch.no_grad()
    def test_visualize(self, image_path, label):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        device="cuda"
        model, preprocess=clip_w_local.load("ViT-B/16", device=device)

        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        output, output_local, neg_output_local= self.model_inference(image)

        num_regions = output_local.shape[1]  # 1,

        label = torch.tensor(label).cuda()
        label = label.unsqueeze(0)

        label_output_local = output_local.gather(2, repeat(label, 'b -> b n 1', n=num_regions))
        # print(label_output_local.shape)
        # 4, 196, 1
        index = label_output_local.topk(k=100, dim=1)[1]
        # print(label_topk_output_local_pos.shape)  # indices
        neg_index = index[:, self.top_k:2 * self.top_k]
        pos_index = index[:, :self.top_k]
        # print(label_topk_output_local_pos.shape) # indices
        # 4, 50, 1

        return pos_index, neg_index

    def Info(self):
        names = ["MCM score", "GL-MCM score", "R-MCM score", "R-MCM score with neg", "SLCM score",
                 "SLCM score with neg"]
        return names