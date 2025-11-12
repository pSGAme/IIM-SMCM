# Intra-Image Mining and Symmetric Maximum Concept Matching for Few Shot Out-of-Distribution Detection  (AAAI 2026)


> Intra-Image Mining and Symmetric Maximum Concept Matching for Few Shot Out-of-Distribution Detection
>
> Kaixiang Chen, Pengfei Fang and Hui Xue

[//]: # ()
[//]: # ([![arXiv]&#40;https://img.shields.io/badge/Arxiv-2409.04796-b31b1b.svg?logo=arXiv&#41;]&#40;https://arxiv.org/abs/2409.04796&#41;)

[//]: # ([![ICLR]&#40;https://img.shields.io/badge/OpenReview-Paper-orange.svg&#41;]&#40;https://openreview.net/pdf?id=Ew3VifXaxZ&#41;)

[//]: # ()
[//]: # (**Key words: Vision-language model, Out-of-distribution detection, Open-environment recognition, Few-shot learning, Prompt learning.**)

[//]: # ()
[//]: # (## :newspaper: News)

[//]: # ()
[//]: # (- **[2025.02.20]** We release [training]&#40;scripts/localprompt/train.sh&#41; and [evaluation]&#40;scripts/localprompt/eval.sh&#41; script for LocalPrompt. Try it now! :fire:)

[//]: # (- **[2025.02.15]** We release camera-ready submission on [openreview]&#40;https://openreview.net/forum?id=Ew3VifXaxZ&noteId=I6rrHj9ExE&#41;. :cake:)

[//]: # (- **[2025.01.23]** **Local-Prompt** has been accepted by **ICLR 2025**! :tada:)

[//]: # (- **[2024.09.07]** [Local-Prompt]&#40;https://arxiv.org/abs/2409.04796&#41; is available on Arxiv. :candy:)

[//]: # ()
[//]: # (## :sparkles: Motivation)

[//]: # ()
[//]: # (### **What is the challenging problem for out-of-distribution detection?**)

[//]: # ()
[//]: # (![overall]&#40;figures/overall.png&#41;)

[//]: # ()
[//]: # (The most challenging scene for OOD detection is that one hard OOD sample is **similar to a known class on the whole** and only **has subtle differences locally**, which naturally requires the detector to identify outliers through local outlier regions. However, existing research falls short of refining OOD task via rich local information when subtle OOD samples are exhibited in certain regions. Some methods merely focus on utilizing global features only &#40;blue boxes&#41;, which ignores local features &#40;red boxes&#41; and inevitably brings about coarse description. Others use the same prompts to match both global and local image features, so the gap between them may lead to inaccurate local outlier identification. Consequently, it is straightforward that enhancing regional information to **empower the model with local outlier knowledge could be significant to OOD detection**.)

[//]: # ()
[//]: # (## :open_book: Overview)

[//]: # ()
[//]: # (### **What do we do to overcome current problem?**)

[//]: # ()
[//]: # (![structure]&#40;figures/structure.png&#41;)

[//]: # ()
[//]: # (We introduce **Local-Prompt**, a novel coarse-to-fine tuning paradigm to emphasize regional enhancement with local prompts. Our method comprises two integral components: **global prompt guided negative augmentation** and **local prompt enhanced regional regularization**. The former utilizes frozen, coarse global prompts as guiding cues to incorporate negative augmentation, thereby leveraging local outlier knowledge. The latter employs trainable local prompts and a regional regularization to capture local information effectively, aiding in outlier identification. We also propose regional-related metric to empower the enrichment of OOD detection. )

[//]: # ()
[//]: # (Comprehensive experiments demonstrate the effectiveness and potential of our method. Notably, our method reduces average FPR95 by 5.17% against state-of-the-art method in 4-shot tuning on challenging ImageNet-1k dataset, even outperforming 16-shot results of previous method.)

[//]: # ()
[//]: # (## :rocket: Quick Start)

[//]: # (### Environment and Package )

[//]: # (```bash)

[//]: # (# Create a conda environment)

[//]: # (conda create -n localprompt python=3.8)

[//]: # ()
[//]: # (# Activate the environment)

[//]: # (conda activate localprompt)

[//]: # ()
[//]: # (# Clone this repo)

[//]: # (git clone https://github.com/AuroraZengfh/Local-Prompt.git)

[//]: # (cd Local-Prompt)

[//]: # ()
[//]: # (# Install necessary packages)

[//]: # (pip install -r requirements.txt)

[//]: # (```)

[//]: # ()
[//]: # (### Install Dassl)

[//]: # (```bash)

[//]: # (cd Dassl)

[//]: # ()
[//]: # (# Install dependencies)

[//]: # (pip install -r requirements.txt)

[//]: # ()
[//]: # (# Install this library &#40;no need to re-build if the source code is modified&#41;)

[//]: # (python setup.py develop)

[//]: # (```)

[//]: # ()
[//]: # (### Data Preparation)

[//]: # (Please create `data` folder and download the following ID and OOD datasets to `data`.)

[//]: # ()
[//]: # (#### In-distribution Datasets)

[//]: # (We use ImageNet-1K as the ID dataset.)

[//]: # (- Create a folder named `imagenet/` under `data` folder.)

[//]: # (- Create `images/` under `imagenet/`.)

[//]: # (- Download the dataset from the [official website]&#40;https://image-net.org/index.php&#41; and extract the training and validation sets to `$DATA/imagenet/images`.)

[//]: # ()
[//]: # (Besides, we need to put `classnames.txt` under`imagenet/data` folder. This .txt file can be downloaded via https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view)

[//]: # ()
[//]: # (For more ID dataset like ImageNet100, ImageNet10 and ImageNet20 in the paper, follow the same procedure to construct 'classnames.txt' in the same format and refer to [MCM]&#40;https://github.com/deeplearning-wisc/MCM&#41; to find detailed category names.)

[//]: # ()
[//]: # (#### Out-of-distribution Datasets)

[//]: # (We use the large-scale OOD datasets [iNaturalist]&#40;https://arxiv.org/abs/1707.06642&#41;, [SUN]&#40;https://vision.princeton.edu/projects/2010/SUN/&#41;, [Places]&#40;https://arxiv.org/abs/1610.02055&#41;, and [Texture]&#40;https://arxiv.org/abs/1311.3618&#41;. You can download the subsampled datasets following instructions from [this repository]&#40;https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset&#41;.)

[//]: # ()
[//]: # (The overall file structure of file is as follows:)

[//]: # (```)

[//]: # (Local-Prompt)

[//]: # (|-- data)

[//]: # (    |-- imagenet)

[//]: # (        |-- classnames.txt)

[//]: # (        |-- train/ # contains 1,000 folders like n01440764, n01443537, etc.)

[//]: # (        |-- val/ # contains 1,000 folders like n01440764, n01443537, etc.)

[//]: # (    |-- imagenet100)

[//]: # (        |-- classnames.txt)

[//]: # (        |-- train/)

[//]: # (        |-- val/)

[//]: # (    |-- imagenet10)

[//]: # (        |-- classnames.txt)

[//]: # (        |-- train/)

[//]: # (        |-- val/)

[//]: # (    |-- imagenet20)

[//]: # (        |-- classnames.txt)

[//]: # (        |-- train/)

[//]: # (        |-- val/)

[//]: # (    |-- iNaturalist)

[//]: # (    |-- SUN)

[//]: # (    |-- Places)

[//]: # (    |-- Texture)

[//]: # (    ...)

[//]: # (```)

[//]: # ()
[//]: # (### Training)

[//]: # (The training script is in `Local-Prompt/scripts/train.sh`, you can alter the parameters in the script file.)

[//]: # ()
[//]: # (e.g., 4-shot training with ViT-B/16)

[//]: # (``` )

[//]: # (CUDA_VISIBLE_DEVICES=0 sh scripts/train.sh data imagenet vit_b16_ep30 end 16 4 True 5 0.5 50)

[//]: # (```)

[//]: # ()
[//]: # (### Inference)

[//]: # (The evaluation script is in Local-Prompt/scripts/train.sh, you can alter the parameters in the script file.)

[//]: # ()
[//]: # (e.g., evaluate 4-shot model of seed1 obtained from training scripts above)

[//]: # ()
[//]: # (```)

[//]: # (CUDA_VISIBLE_DEVICES=0 sh scripts/eval.sh data imagenet vit_b16_ep30 16 10 output/imagenet/LOCALPROMPT/vit_b16_ep30_4shots/nctx16_cscTrue_ctpend_topk50/seed1)

[//]: # (```)

[//]: # ()
[//]: # (## :blue_book: Citation)

[//]: # (If you find this work useful, consider giving this repository a star :star: and citing :bookmark_tabs: our paper as follows:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@inproceedings{zeng2025local,)

[//]: # (  title={Local-Prompt: Extensible Local Prompts for Few-Shot Out-of-Distribution Detection},)

[//]: # (  author={Zeng, Fanhu and Cheng, Zhen and Zhu, Fei and Wei, Hongxin and Zhang, Xu-Yao},)

[//]: # (  booktitle={The Thirteenth International Conference on Learning Representations},)

[//]: # (  year={2025},)

[//]: # (  url={https://openreview.net/forum?id=Ew3VifXaxZ})

[//]: # (})

[//]: # (```)

## Acknowledgememnt

The code is based on  [CoOp](https://github.com/KaiyangZhou/CoOp), [LoCoOp](https://github.com/AtsuMiyai/LoCoOp). Thanks for these great works and open sourcing! 

If you find them helpful, please consider citing them as well. 

## Contact

If you have any questions, feel free to contact [kxchen@seu.edu.cn](kxchen@seu.edu.cn).
