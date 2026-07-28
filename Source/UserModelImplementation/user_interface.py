# -*- coding: utf-8 -*-
"""Entry-point interface handed to ``jf.Application``.

交给 ``jf.Application`` 的入口接口。
"""

import argparse

import JackFramework as jf
# import UserModelImplementation.user_define as user_def

from UserModelImplementation import Models
from UserModelImplementation import Dataloaders


class UserInterface(jf.UserTemplate.NetWorkInferenceTemplate):
    """Wire the model zoo and dataloader zoo into JackFramework.

    把模型注册表与数据集注册表接进 JackFramework。
    """

    def inference(self, args: object) -> tuple:
        """Instantiate the (model, dataloader) pair selected by the CLI flags.

        按命令行参数实例化 (模型, dataloader) 二元组。
        """
        dataloader = Dataloaders.dataloaders_zoo(args, args.dataset)
        model = Models.model_zoo(args, args.modelName)
        return model, dataloader

    def user_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add project-specific CLI flags.

        添加项目自定义命令行参数。

        Note:
            **Must return the parser.** JackFramework keeps the returned
            object only if ``isinstance(x, argparse.ArgumentParser)``;
            returning None silently discards every flag added here.
            **必须把 parser 返回。** JackFramework 只在
            ``isinstance(x, argparse.ArgumentParser)`` 成立时采用返回值；
            返回 None 会让这里加的参数被静默丢弃。
        """
        # parser.add_argument('--startDisp', type=int, default=0,
        #                     help='start disparity')
        return parser

    # NOTE: do not re-implement _str2bool here — the base class provides it.
    # 注意：不要重复实现 _str2bool，基类已经提供。
