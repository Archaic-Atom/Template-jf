# -*- coding: utf-8 -*-
"""Dataloader zoo — maps ``--dataset`` to a data-handler class.

数据集注册表 —— 把 ``--dataset`` 映射到数据处理类。

The key here MUST match what the launcher passes to ``--dataset``.

这里的 key **必须**和启动脚本传给 ``--dataset`` 的值一致。
"""

import JackFramework as jf

from .your_dataloader import YourDataloader


def dataloaders_zoo(args: object, name: str) -> object:
    """Return the dataloader registered under ``name``.

    返回以 ``name`` 注册的 dataloader。
    """
    dataloader = None
    for case in jf.Switch(name):
        if case('YourDataloader'):
            jf.log.info("Enter the YourDataloader")
            dataloader = YourDataloader(args)
            break
        if case():
            jf.log.error("The dataloader's name is error!!! got: %s" % name)
    return dataloader
