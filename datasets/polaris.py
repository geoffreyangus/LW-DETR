# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Polaris dataset in COCO format which returns image_id for evaluation.
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision

import datasets.transforms as T
from datasets.coco import CocoDetection, make_coco_transforms, make_coco_transforms_square_div_64

IMAGE_FOLDER = "/mnt/e/data/pohang-canal-dataset/pohang00/stereo/left_images"

class PolarisDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(PolarisDetection, self).__init__(img_folder, ann_file, transforms)


def build(image_set, args):

    polaris_path = Path(args.polaris_path)

    PATHS = {
        "train": (IMAGE_FOLDER, polaris_path / "train_annotations.json"),
        "val": (IMAGE_FOLDER, polaris_path / "val_annotations.json"),
        "test": (IMAGE_FOLDER, polaris_path / "test_annotations.json"),
    }
    
    img_folder, ann_file = PATHS[image_set.split("_")[0]]
    
    try:
        square_resize = args.square_resize
    except:
        square_resize = False
    
    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False

    
    if square_resize_div_64:
        dataset = PolarisDetection(img_folder, ann_file, transforms=make_coco_transforms_square_div_64(image_set))
    else:
        dataset = PolarisDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set))
    return dataset
