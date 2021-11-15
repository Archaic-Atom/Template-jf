# -*- coding: utf-8 -*-
# import UserModelImplementation.user_define as user_def
import JackFramework as jf
import argparse

# model
from UserModelImplementation.Models.Your_Model.inference import YourModel

# dataloader
from UserModelImplementation.Dataloaders.your_dataloader import YourDataloader


class UserInterface(jf.UserTemplate.NetWorkInferenceTemplate):
    """docstring for UserInterface"""

    def __init__(self) -> object:
        super().__init__()

    def inference(self, args: object) -> object:
        name = args.modelName
        for case in jf.Switch(name):
            if case('YourModel'):
                jf.log.warning("Enter the YourModel model!")
                model = YourModel(args)
                dataloader = YourDataloader(args)
                break
            if case():
                model = None
                dataloader = None
                jf.log.error("The model's name is error!!!")

        return model, dataloader

    def user_parser(self, parser: object) -> object:
        # parser.add_argument('--startDisp', type=int,
        #                    default=user_def.START_DISP,
        #                    help='start disparity')
        #
        # return parser
        return None

    @staticmethod
    def __str2bool(arg: str) -> bool:
        if arg.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
