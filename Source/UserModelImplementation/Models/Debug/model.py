# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.utils.data
import torch.nn.functional as F

import numpy as np
from typing import TypeVar, Generic

import JackFramework as jf
import time

from .submodule import FeatureExtraction, CostVolume
from .submodule import FeatureMatching_v2, DispRegression

tensor = TypeVar('tensor')


def norm_layer_func(out_channels: int) -> object:
    return nn.BatchNorm2d(out_channels)


class Model(nn.Module):
    """docstring for Model"""

    def __init__(self, in_channels: int, start_disp: int, disp_num: int,
                 out_full_channels: int = 64, out_reduce_channels: int = 16,
                 scale_size: int = 4) -> object:
        super().__init__()
        # jf.layer.set_norm_layer_func(norm_layer_func)

        self.__start_disp = start_disp
        self.__disp_num = disp_num
        # out_channel = 64, 16
        self.feature_extraction = FeatureExtraction(
            in_channels, out_full_channels, out_reduce_channels)
        # out_channel = 64 + 16 *2 = 96
        self.cost_volume = CostVolume(
            self.__disp_num // scale_size, out_full_channels, out_reduce_channels)
        # outchannel = 1
        self.feature_matching = FeatureMatching_v2(
            out_full_channels + 2 * out_reduce_channels,
            self.__start_disp, self.__disp_num)

        # torch.autograd.set_detect_anomaly(True)

    def forward(self, left_img: tensor, right_img: tensor) -> tensor:
        _, _, height, width = left_img.size()

        left_img, reduce_left_img = self.feature_extraction(left_img)
        right_img, reduce_right_img = self.feature_extraction(right_img)

        cost = self.cost_volume(left_img, reduce_left_img, right_img, reduce_right_img)
        disp_0 = self.feature_matching(cost, self.__disp_num, height, width)

        return disp_0
