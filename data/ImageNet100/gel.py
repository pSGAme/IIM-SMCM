with open("/data/ood/imagenet/classnames.txt") as f:
    all_classes = dict(line.strip().split(' ', 1) for line in f if line.strip())


with open("/home/user/Code/Local-Prompt/data/ImageNet100/class_list.txt") as f:
    subset_ids = set(line.strip() for line in f if line.strip())



subset_classes = {cls_id: all_classes[cls_id] for cls_id in subset_ids if cls_id in all_classes}

with open("classnames.txt", "w") as f:
    for cls_id in subset_ids:
        if cls_id in subset_classes:
            f.write(f"{cls_id} {subset_classes[cls_id]}\n")