# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import JackFramework as jf
import UserModelImplementation.user_define as user_def


class Debug(jf.UserTemplate.ModelHandlerTemplate):
    """docstring for DeepLabV3Plus"""

    def __init__(self, args: object) -> object:
        super().__init__(args)
        self.__args = args
        self.__criteria = nn.CrossEntropyLoss()
        self.__acc = None

    def get_model(self) -> list:
        args = self.__args
        # return output
        model = jf.sm.GwcNet(args.dispNum)
        # model = Model(user_def.RGB_CHANNELS_NUM,
        #              args.startDisp, args.dispNum)
        return [model]

    def optimizer(self, model: list, lr: float) -> list:
        # args = self.__args
        # return opt
        opt = optim.AdamW(model[0].parameters(), lr=lr, weight_decay=0.05)

        max_warm_up_epoch = 5
        convert_epoch = 30
        off_set = 1

        def lr_lambda(epoch): return ((epoch + off_set) / max_warm_up_epoch * lr) \
            if epoch < max_warm_up_epoch else lr if (
            epoch >= max_warm_up_epoch and epoch < convert_epoch) else lr * 0.1

        # sch = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5,
        #                                           threshold=0.01,
        #                                           cooldown=0, verbose=True)
        sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return [opt], [sch]

    def lr_scheduler(self, sch: object, ave_loss: list, sch_id: int) -> None:
        model_id = 0
        if model_id == sch_id:
            sch.step()

    def inference(self, model: list, input_data: list, model_id: int) -> list:
        pred0, pred1, pred2, pred3 = model(input_data[0], input_data[1])
        return [pred0, pred1, pred2, pred3]

    def accuary(self, output_data: list, label_data: list, model_id: int) -> list:
        # return acc
        args = self.__args
        acc = jf.Accuracy.d_1(output_data[0], label_data[0])
        return [acc[1]]

    def loss(self, output_data: list, label_data: list, model_id: int) -> list:
        # return loss
        args = self.__args

        l0 = jf.Loss.smooth_l1(
            output_data[0], label_data[0], args.startDisp, args.startDisp + args.dispNum)
        l1 = jf.Loss.smooth_l1(
            output_data[1], label_data[0], args.startDisp, args.startDisp + args.dispNum)
        l2 = jf.Loss.smooth_l1(
            output_data[2], label_data[0], args.startDisp, args.startDisp + args.dispNum)
        l3 = jf.Loss.smooth_l1(
            output_data[3], label_data[0], args.startDisp, args.startDisp + args.dispNum)
        total_loss = l0 + l1 + l2 + l3

        return [total_loss, l0]
