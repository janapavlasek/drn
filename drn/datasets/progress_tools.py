#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as TF


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

    def __init__(self, data_dir, split, transforms=None, num_transforms=1,
                 out_name=False, labels=True, background=True, normalize=True):

        self.CLASSES = sorted(self.CLASSES, key=lambda x: self.CLASS_LABELS[x])

        self.data_dir = data_dir
        self.labels = labels
        self.split = split
        self.transforms = transforms
        self.num_transforms = num_transforms
        self.out_name = out_name
        self.background = background
        self.normalize = normalize

        self.image_list = []
        self.mask_list = []

        self.read_lists()

        self.info = json.load(open(os.path.join(data_dir, 'info.json'), 'r'))

    def __getitem__(self, index):
        # Don't allow indices greater than the length of our data.
        if index >= self.__len__():
            raise IndexError()

        # Wrap around to allow for duplicate transforms.
        list_idx = index % len(self.image_list)

        # Load image as PIL.
        img = Image.open(self.image_list[list_idx]).convert("RGB")

        # Get the mask.
        mask = None
        if self.labels:
            mask_path = self.mask_list[list_idx]
            mask = Image.open(mask_path)
            if not self.background:
                # Background is the 0 element so remove this.
                mask = Image.fromarray(np.array(mask) - 1)

        # Apply the transforms.
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        # Apply these transforms always.
        img, mask = self.to_tensor(img, mask)

        # Normalize if necessary.
        if self.normalize:
            img = self.normalize_image(img)

        # Return the name of the file if requested.
        if self.out_name:
            return img, mask, self.image_list[list_idx]

        # If there is no mask, return only the image
        if mask is None:
            return img

        # Default return.
        return img, mask

    def __len__(self):
        return len(self.image_list) * self.num_transforms

    def read_lists(self):
        image_path = os.path.join(self.data_dir, "{}_images.txt".format(self.split))
        mask_path = os.path.join(self.data_dir, "{}_masks.txt".format(self.split))

        assert os.path.exists(image_path), "Path does not exist: {}".format(image_path)

        self.image_list = [line.strip() for line in open(image_path, 'r')]

        if os.path.exists(mask_path):
            self.mask_list = [line.strip() for line in open(mask_path, 'r')]
            assert len(self.image_list) == len(self.mask_list)

    def normalize_image(self, image):
        image = TF.normalize(image, mean=self.info["mean"], std=self.info["std"])
        return image

    def to_tensor(self, image, mask):
        t = TF.to_tensor(image)

        if mask is None:
            return t, mask

        return t, torch.LongTensor(np.array(mask, dtype=np.int))


class ProgressToolParts(ProgressTools):

    def __init__(self, *args, **kwargs):
        super(ProgressToolParts, self).__init__(*args, **kwargs)

        self.CLASS_LABELS = {"__background__": 0,
                             "clamp_bar": 1,
                             "clamp_top_clamp": 2,
                             "clamp_bottom_clamp": 3,
                             "clamp_handle": 4,
                             "flashlight_base": 5,
                             "flashlight_light": 6,
                             "grey_pliers_handle": 7,
                             "grey_pliers_pinch": 8,
                             "hammer_handle": 9,
                             "hammer_head": 10,
                             "knife_blade": 11,
                             "knife_handle": 12,
                             "longnose_pliers_handle": 13,
                             "longnose_pliers_pinch": 14,
                             "red_pliers_handle": 15,
                             "red_pliers_pinch": 16,
                             "screwdriver_handle": 17,
                             "screwdriver_tip": 18}

        self.CLASSES = [k for k in self.CLASS_LABELS.keys()]
        self.CLASSES = sorted(self.CLASSES, key=lambda x: self.CLASS_LABELS[x])
