import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import clip_w_local
from clip_w_local import clip

def set_model_clip(cfg):
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

    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    val_preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    return model.cuda().eval(), val_preprocess



def set_val_loader(args, preprocess=None):
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    kwargs = {'num_workers': 2, 'pin_memory': True}
    if args.in_dataset == "imagenet":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.root, 'imagenet/val'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        # 50000 print(len(datasets.ImageFolder(os.path.join(args.root, 'imagenet/val'), transform=preprocess)))
        # 128 print(args.batch_size)
    elif args.in_dataset == "imagenet100":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.root, 'imagenet100/val'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "imagenet10":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.root, 'imagenet10/val'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "imagenet20":
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(args.root, 'imagenet20/val'), transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        raise NotImplementedError
    return val_loader


def set_ood_loader_ImageNet(args, out_dataset, preprocess=None):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    # print(preprocess)
    if preprocess is None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    if out_dataset == 'iNaturalist':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'Places'), transform=preprocess)
    elif out_dataset == 'Texture':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'dtd', 'images'),
                                          transform=preprocess)
    elif out_dataset == 'imagenet20':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'imagenet20', 'val'),
                                          transform=preprocess)
    elif out_dataset == 'imagenet10':
        testsetout = datasets.ImageFolder(root=os.path.join(args.root, 'imagenet10', 'train'),
                                          transform=preprocess)
    # print(len(testsetout))
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                                shuffle=False, num_workers=2)
    return testloaderOut
