# -*- coding: utf-8 -*-
"""Model zoo — maps ``--modelName`` to a model-interface class.

模型注册表 —— 把 ``--modelName`` 映射到模型接口类。

The key here MUST match what the launcher passes to ``--modelName``.
A mismatch surfaces as a bare AssertionError with no hint.

这里的 key **必须**和启动脚本传给 ``--modelName`` 的值一致；
对不上时只会抛一个没有提示的 AssertionError。
"""

import JackFramework as jf

from .your_model.inference import YourModelInterface


def model_zoo(args: object, name: str) -> object:
    """Return the model interface registered under ``name``.

    返回以 ``name`` 注册的模型接口。
    """
    model = None
    for case in jf.Switch(name):
        if case('YourModel'):
            jf.log.info("Enter the YourModel interface")
            model = YourModelInterface(args)
            break
        if case():
            jf.log.error("The model's name is error!!! got: %s" % name)
    return model
