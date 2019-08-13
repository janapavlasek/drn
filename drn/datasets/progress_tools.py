#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class ProgressTools(Dataset):

    """Progress full tools dataset."""

    CLASS_LABELS = {
        "__background__": 0,
        "clamp": 1,
        "flashlight": 2,
        "grey_pliers": 3,
        "hammer": 4,
        "knife": 5,
        "longnose_pliers": 6,
        "red_pliers": 7,
        "screwdriver": 8
    }
    CLASSES = [k for k in CLASS_LABELS.keys()]

    def __init__(self, data_dir, split, transforms=None, always_transforms=None,
                 num_transforms=0, out_name=False, labels=True, background=True):
        self.data_dir = data_dir
        self.labels = labels
        self.split = split
        self.transforms = transforms
        self.always_transforms = always_transforms
        self.num_transforms = num_transforms
        self.out_name = out_name
        self.background = background

        self.image_list = []
        self.mask_list = []

        self.read_lists()

    def __getitem__(self, index):
        list_idx = index % len(self.image_list)
        is_transform = index >= len(self.image_list)

        img = Image.open(self.image_list[list_idx]).convert("RGB")
        data = [img]

        mask = None
        if self.labels:
            mask_path = self.mask_list[list_idx]
            if not self.background:
                mask_path = mask_path.replace("labels.png", "labels_train.png")
            mask = Image.open(mask_path)
            data.append(mask)

        if self.transforms is not None and is_transform:
            data = list(self.transforms(*data))

        if self.always_transforms is not None:
            data = list(self.always_transforms(*data))

        if self.out_name:
            data.append(self.image_list[list_idx])

        # print("--------------")
        # for ele in data:
        #     print("\t",ele.shape)
        return tuple(data)

    def __len__(self):
        return len(self.image_list) * (self.num_transforms + 1)

    def read_lists(self):
        image_path = os.path.join(self.data_dir, "{}_images.txt".format(self.split))
        mask_path = os.path.join(self.data_dir, "{}_masks.txt".format(self.split))

        assert os.path.exists(image_path)

        self.image_list = [line.strip() for line in open(image_path, 'r')]

        if os.path.exists(mask_path):
            self.mask_list = [line.strip() for line in open(mask_path, 'r')]
            assert len(self.image_list) == len(self.mask_list)

    def get_class_distribution(self):
        for m in self.mask_list:
            mask = Image.open(m)
            mask = np.array(mask)


class ProgressToolParts(ProgressTools):

    CLASS_LABELS = {"__background__": 0,
                    "clamp_bar": 1,
                    "clamp_bottom_clamp": 3,
                    "clamp_handle": 4,
                    "clamp_top_clamp": 2,
                    "flashlight_base": 5,
                    "flashlight_light": 6,
                    "flashlight_stand": 7,
                    "screwdriver_handle": 8,
                    "screwdriver_tip": 9}
    CLASSES = [k for k in CLASS_LABELS.keys()]

    def __init__(self, *args, **kwargs):
        super(ProgressToolParts, self).__init__(*args, **kwargs)
