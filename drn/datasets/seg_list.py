#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)


class SegList(Dataset):

    CLASSES = ['road',
               'sidewalk',
               'building',
               'wall',
               'fence',
               'pole',
               'trafficlight',
               'trafficsign',
               'vegetation',
               'terrain',
               'sky',
               'person',
               'rider',
               'car',
               'truck',
               'bus',
               'train',
               'motorcycle',
               'bicycle']

    def __init__(self, data_dir, phase, trans, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = trans
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(os.path.join(self.data_dir, self.image_list[index]))]
        if self.label_list is not None:
            data.append(Image.open(
                os.path.join(self.data_dir, self.label_list[index])))
        data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = os.path.join(self.list_dir, self.phase + '_images.txt')
        label_path = os.path.join(self.list_dir, self.phase + '_labels.txt')
        assert os.path.exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if os.path.exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)


class SegListMS(Dataset):

    CLASSES = ['road',
               'sidewalk',
               'building',
               'wall',
               'fence',
               'pole',
               'trafficlight',
               'trafficsign',
               'vegetation',
               'terrain',
               'sky',
               'person',
               'rider',
               'car',
               'truck',
               'bus',
               'train',
               'motorcycle',
               'bicycle']

    def __init__(self, data_dir, phase, trans, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = trans
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()
        self.scales = scales

    def __getitem__(self, index):
        data = [Image.open(os.path.join(self.data_dir, self.image_list[index]))]
        w, h = data[0].size
        if self.label_list is not None:
            data.append(Image.open(
                os.path.join(self.data_dir, self.label_list[index])))
        # data = list(self.transforms(*data))
        out_data = list(self.transforms(*data))
        ms_images = [self.transforms(data[0].resize((int(w * s), int(h * s)),
                                                    Image.BICUBIC))[0]
                     for s in self.scales]
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        return tuple(out_data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = os.path.join(self.list_dir, self.phase + '_images.txt')
        label_path = os.path.join(self.list_dir, self.phase + '_labels.txt')
        assert os.path.exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if os.path.exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)
