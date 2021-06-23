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

tensor = TypeVar('tensor')


class FeatureExtraction(nn.Module):
    """docstring for Model"""

    def __init__(self, in_channels: int, out_full_channels: int,
                 out_reduce_channels: int, hidden_channels: int = 32)-> None:
        super().__init__()

        self.block_1 = self.__first_layer(in_channels, hidden_channels)
        # res-block
        self.block_2 = self.__make_block(
            hidden_channels, hidden_channels, 3, 1)
        self.block_3 = self.__make_block(
            hidden_channels, 2*hidden_channels, 8, 1)
        self.block_4 = self.__make_block(
            2*hidden_channels, 4*hidden_channels, 2, 1)

        # dilation block
        self.block_5 = self.__make_block(
            4*hidden_channels, 4*hidden_channels, 3, 2)
        self.block_6 = self.__make_block(
            4*hidden_channels, 4*hidden_channels, 3, 4)

        # aspp
        self.aspp = jf.block.ASPPBlock(
            4*hidden_channels, hidden_channels)

        # fusion
        self.fusion = jf.layer.conv_2d_layer(
            16*hidden_channels, out_full_channels, 3,
            bias=True, norm=False, act=False)
        self.reduce_channels = jf.layer.conv_2d_layer(
            64, out_reduce_channels, 1,
            padding=0, bias=True, norm=False, act=False)

    def __first_layer(self, in_channels: int, out_channels: int) -> object:
        layer = [jf.layer.conv_2d_layer(in_channels, out_channels, 3, 2)]
        layer.append(jf.layer.conv_2d_layer(
            out_channels, out_channels, 3))
        layer.append(jf.layer.conv_2d_layer(
            out_channels, out_channels, 3))
        layer.append(jf.layer.conv_2d_layer(
            out_channels, out_channels, 3, 2))
        return nn.Sequential(*layer)

    def __make_block(self, in_channels: int, out_channels: int,
                     block_num: int, dilation: int) -> object:
        layer = [
            jf.layer.conv_2d_layer(
                in_channels, out_channels, 3, padding=dilation, dilation=dilation
            )
        ]


        for _ in range(block_num):
            layer.append(jf.block.Res2DBlock(
                out_channels, out_channels, 3,
                padding=dilation, dilation=dilation))

        return nn.Sequential(*layer)

    def forward(self, x: tensor)->tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        level_1 = self.block_4(x)
        level_2 = self.block_5(level_1)
        level_3 = self.block_6(level_2)
        x = self.aspp(level_3)
        x = torch.cat((level_1, level_2, level_3, x), 1)
        x = self.fusion(x)
        reduce_x = self.reduce_channels(x)
        return x, reduce_x


class CostVolume_v2(nn.Module):
    def __init__(self, maxdisp):
        super().__init__()
        self.maxdisp = maxdisp + 1

    def forward(self, x, y):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            num, channels, height, width = x.size()
            cost = x.new().resize_(num, channels * 2, self.maxdisp, height, width).zero_()

            for i in range(self.maxdisp):
                if i > 0:
                    cost[:, :x.size()[1], i, :, i:] = x[:, :, :, i:]
                    cost[:, x.size()[1]:, i, :, i:] = y[:, :, :, :-i]
                else:
                    cost[:, :x.size()[1], i, :, :] = x
                    cost[:, x.size()[1]:, i, :, :] = y

            cost = cost.contiguous()
        return cost


class CostVolume(nn.Module):
    """docstring for CostVolume"""

    def __init__(self, max_disp: int,  out_full_channels: int, out_reduce_channels: int):
        super().__init__()
        self.__max_disp = int(max_disp)
        self.__out_full_channels = out_full_channels
        self.__out_reduce_channels = out_reduce_channels

    def _new_space(self, left_img: tensor, reduce_left_img: tensor):
        # new the var volume
        batch_size, channels, height, width = left_img.size()
        cost_var = left_img.new().resize_(
            batch_size, channels, self.__max_disp, height, width).zero_()

        # to computer the var
        tmp_left_img = reduce_left_img.new().resize_(
            batch_size, channels, height, width).zero_()
        tmp_right_img = reduce_left_img.new().resize_(
            batch_size, channels, height, width).zero_()

        # new the car volume
        _, reduce_channels, _, _ = reduce_left_img.size()
        cost_cat = reduce_left_img.new().resize_(
            batch_size, reduce_channels*2, self.__max_disp, height, width).zero_()

        return cost_var, cost_cat, tmp_left_img, tmp_right_img

    def forward(self, left_img: tensor, reduce_left_img: tensor,
                right_img: tensor, reduce_right_img: tensor) -> tensor:
        with torch.cuda.device_of(left_img):
            cost_var, cost_cat, tmp_left_img, tmp_right_img = self._new_space(
                left_img, reduce_left_img)
            reduce_channels = self.__out_reduce_channels

            for i in range(self.__max_disp):
                tmp_left_img.zero_()
                tmp_right_img.zero_()
                if i > 0:
                    cost_cat[:, :reduce_channels, i, :, i:] = reduce_left_img[:, :, :, i:]
                    cost_cat[:, reduce_channels:, i, :, i:] = reduce_right_img[:, :, :, :-i]
                    tmp_left_img[:, :, :, i:] = left_img[:, :, :, i:]
                    tmp_right_img[:, :, :, i:] = right_img[:, :, :, :-i]
                else:
                    cost_cat[:, :reduce_channels, i, :, :] = reduce_left_img
                    cost_cat[:, reduce_channels:, i, :, :] = reduce_right_img
                    tmp_left_img[:, :, :, i:] = left_img
                    tmp_right_img[:, :, :, i:] = right_img

                ave_feature = (tmp_left_img + tmp_right_img) / 2
                tmp_left_img = tmp_left_img ** 2
                tmp_right_img = tmp_right_img ** 2
                ave_feature2 = (tmp_left_img + tmp_right_img) / 2
                feature_var = ave_feature2 - ave_feature**2
                cost_var[:, :, i, :, i:] = feature_var[:, :, :, i:]

            cost = torch.cat((cost_var, cost_cat), 1)
            # cost = cost_cat
            cost = cost.contiguous()
        return cost


class FeatureMatching(nn.Module):
    """docstring for FeatureMatching"""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 32)->object:
        super().__init__()

        self.first_layer = self.__first_layer(in_channels, hidden_channels)
        self.down_sampling_block_1 = self.__down_sampling_block(hidden_channels, 2*hidden_channels)
        self.down_sampling_block_2 = self.__down_sampling_block(
            2*hidden_channels, 3*hidden_channels)
        self.down_sampling_block_3 = self.__down_sampling_block(
            3*hidden_channels, 4*hidden_channels)
        self.down_sampling_block_4 = self.__down_sampling_block(
            4*hidden_channels, 4*hidden_channels)

        self.up_sampling_block_4 = self.__up_sampling_block(
            4*hidden_channels, 4*hidden_channels)
        self.up_sampling_block_3 = self.__up_sampling_block(
            4*hidden_channels, 3*hidden_channels)
        self.up_sampling_block_2 = self.__up_sampling_block(
            3*hidden_channels, 2*hidden_channels)
        self.up_sampling_block_1 = self.__up_sampling_block(
            2*hidden_channels, hidden_channels)

        self.last_layer = self.__last_layer(hidden_channels, out_channels)

    def __first_layer(self, in_channels: int, out_channels: int) -> object:
        layer = [jf.layer.conv_3d_layer(in_channels, out_channels*2, 3)]
        layer.append(jf.layer.conv_3d_layer(out_channels*2, out_channels, 3))
        layer.append(jf.block.Res3DBlock(out_channels, out_channels, 3))
        return nn.Sequential(*layer)

    def __down_sampling_block(self, in_channels: int, out_channels: int) -> tensor:
        layer = [jf.layer.conv_3d_layer(in_channels, out_channels, 3, 2)]
        layer.append(jf.block.Res3DBlock(out_channels, out_channels, 3))
        return nn.Sequential(*layer)

    def __up_sampling_block(self, in_channels: int, out_channels: int) -> tensor:
        layer = [jf.layer.deconv_3d_layer(in_channels, out_channels, 3, 2, 1, 1)]
        layer.append(jf.block.Res3DBlock(out_channels, out_channels, 3))
        return nn.Sequential(*layer)

    def __last_layer(self, in_channels: int, out_channels: int) -> tensor:
        layer = [
            jf.layer.deconv_3d_layer(
                in_channels,
                in_channels // 2,
                3,
                2,
                padding=1,
                bias=True,
                norm=False,
                act=False,
            )
        ]

        layer.append(jf.layer.deconv_3d_layer(
            in_channels // 2, out_channels, 3, 2,
            padding=1, bias=True, norm=False, act=False))
        return nn.Sequential(*layer)

    def _feature2prob(self, x: tensor, disp_num: int,
                      height: int, width: int) -> tensor:
        x = F.interpolate(x, [disp_num, height, width], mode='trilinear')
        x = torch.squeeze(x, 1)
        x = F.softmax(x, dim=1)
        return x

    def forward(self, x: tensor, disp_num: int,
                height: int, width: int)->tensor:
        x = self.first_layer(x)
        level_1 = self.down_sampling_block_1(x)
        level_2 = self.down_sampling_block_2(level_1)
        level_3 = self.down_sampling_block_3(level_2)
        level_4 = self.down_sampling_block_4(level_3)
        level_3 = level_3 + self.up_sampling_block_4(level_4)
        level_2 = level_2 + self.up_sampling_block_3(level_3)
        level_1 = level_1 + self.up_sampling_block_2(level_2)
        x = x + self.up_sampling_block_1(level_1)
        x = self.last_layer(x)
        x = self._feature2prob(x, disp_num, height, width)
        return x


class FeatureMatching_v2(nn.Module):
    def __init__(self, in_channels: int, start_disp: int,
                 disp_num: int, hidden_channels: int = 32)->object:
        super().__init__()
        self.__patch_height = 64
        self.__patch_width = 128

        self.reduce_channels_layers = jf.layer.conv_3d_layer(
            in_channels, hidden_channels, 1, padding=0)
        self.transformer_l1 = jf.vp.TimeSformer_v2(
            bottleneck_channels=128, image_height=self.__patch_height,
            image_width=self.__patch_width, num_frames=48, in_channels=32,
            patch_size=4, depth=5, heads=8, dim_head=16,
            attn_dropout=0, ff_dropout=0)

        self.last_layers_l1 = jf.layer.conv_3d_layer(
            hidden_channels, 1, 1, padding=0,
            bias=True, norm=False, act=False)

        self.regression = DispRegression(start_disp, disp_num)

    def _feature2prob(self, x: tensor, disp_num: int,
                      height: int, width: int) -> tensor:
        x = F.interpolate(x, [disp_num, height, width],
                          mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        x = F.softmax(x, dim=1)
        return x

    def _patch_transfomer(self, x: tensor, transformer_func: object)->tensor:
        _, _, _, height, width = x.size()
        assert (height % self.__patch_height == 0) and \
            (width % self.__patch_width == 0)

        path_num_h = height // self.__patch_height
        path_num_w = width // self.__patch_width

        off_set = 1
        least_patch_num = 1
        patch_cat_h = []
        for i in range(path_num_h):
            start_h = i * self.__patch_height
            end_h = (i + off_set) * self.__patch_height
            patch_cat_w = []
            for j in range(path_num_w):
                start_w = j * self.__patch_width
                end_w = (j + off_set) * self.__patch_width
                patch_img = x[:, :, :, start_h:end_h, start_w:end_w]
                patch_img = transformer_func(patch_img)
                patch_cat_w.append(patch_img)
            if path_num_w > least_patch_num:
                patch_cat_w = torch.cat(tuple(patch_cat_w), 4)  # concat w
            else:
                patch_cat_w = patch_cat_w[0]
            patch_cat_h.append(patch_cat_w)

        if path_num_h > least_patch_num:
            x = torch.cat(tuple(patch_cat_h), 3)  # concat h
        else:
            x = patch_cat_h[0]

        return x

    def forward(self, x: tensor, disp_num: int,
                height: int, width: int) -> tensor:

        x = self.reduce_channels_layers(x)
        x = self._patch_transfomer(x, self.transformer_l1)
        cost_0 = self.last_layers_l1(x)
        cost_0 = self._feature2prob(
            cost_0, disp_num, height, width)
        cost_0 = self.regression(cost_0)

        return cost_0


class DispRegression(nn.Module):
    """docstring for DispRegression"""

    def __init__(self, start_disp: int, disp_num: int)->object:
        super().__init__()
        #self.softmax = nn.Softmin(dim=1)
        self.__start_disp = start_disp
        self.__disp_num = disp_num

    def _disp_regression(self, x: tensor) -> tensor:
        assert len(x.shape) == 4
        disp_values = torch.arange(
            self.__start_disp, self.__start_disp+self.__disp_num,
            dtype=x.dtype, device=x.device)
        disp_values = disp_values.view(1, self.__disp_num, 1, 1)
        return torch.sum(x * disp_values, 1, keepdim=False)

    def forward(self, x: tensor) -> tensor:
        return self._disp_regression(x)
