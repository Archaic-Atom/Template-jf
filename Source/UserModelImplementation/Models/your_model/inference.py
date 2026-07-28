# -*- coding: utf-8 -*-
"""Model-interface skeleton for a JackFramework project.

JackFramework 项目的模型接口骨架。

This file is what JackFramework dispatches into. Every method below is
called **by name** by the framework, so the signatures must match the
base class exactly. As shipped it trains a tiny MLP on synthetic data so
that a freshly cloned template runs end-to-end; replace the parts marked
``REPLACE`` with your own network.

本文件是 JackFramework 的调用入口。下面每个方法都由框架**按名字**调用，
签名必须与基类完全一致。模板出厂自带一个极小的 MLP + 合成数据，保证
clone 下来就能跑通全流程；把标了 ``REPLACE`` 的部分换成你自己的网络即可。
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import JackFramework as jf
# import UserModelImplementation.user_define as user_def


class _TinyNet(nn.Module):
    """REPLACE — placeholder network so the template is runnable.

    REPLACE —— 占位网络，仅为让模板开箱可跑。

    Args:
        in_dim: Flattened input dimension. 展平后的输入维度。
        out_dim: Output dimension. 输出维度。
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self._body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._body(x)


class YourModelInterface(jf.UserTemplate.ModelHandlerTemplate):
    """User-side model contract consumed by JackFramework.

    JackFramework 使用的用户侧模型契约。

    Note:
        ``inference`` MUST return a **list** even for a single model, and
        ``loss`` / ``accuracy`` receive that list — unwrap with
        ``output_data[self.ID_PRED]`` (index 0) before touching the tensor.

        ``inference`` 即使只有一个模型也**必须返回 list**；``loss`` /
        ``accuracy`` 收到的就是这个 list，取张量前先用
        ``output_data[self.ID_PRED]``（下标 0）解包。
    """

    ID_PRED = 0            # index of the prediction inside the output list
    _IN_DIM = 3 * 32 * 32  # REPLACE
    _OUT_DIM = 10          # REPLACE

    def __init__(self, args: object) -> None:
        super().__init__(args)
        self.__args = args

    def get_model(self) -> Sequence[object]:
        """Build every model replica JackFramework should manage.

        构建所有交给 JackFramework 托管的模型副本。

        Returns:
            A list of ``nn.Module``. 模型副本列表。
        """
        # REPLACE: return [YourNetwork(self.__args)]
        return [_TinyNet(self._IN_DIM, self._OUT_DIM)]

    def optimizer(self, model: Sequence[object], lr: float) -> Tuple[Sequence, Sequence]:
        """Create one optimizer (and optional scheduler) per model replica.

        为每个模型副本创建优化器（调度器可选）。

        Returns:
            ``(opt_list, sch_list)``, both the same length as ``model``.
            ``(优化器列表, 调度器列表)``，长度与 ``model`` 一致。
        """
        opt = optim.Adam(model[0].parameters(), lr=lr)
        # Return [None] for the scheduler slot if you do not use one.
        # 不用调度器时，调度器槽位返回 [None]。
        sch = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
        return [opt], [sch]

    def lr_scheduler(self, sch: object, ave_loss: float, sch_id: int) -> None:
        """Advance the scheduler once per epoch.

        每个 epoch 推进一次调度器。
        """
        if sch is not None:
            sch.step()

    def inference(self, model: object, input_data: List, model_id: int) -> List:
        """Forward pass for one model replica.

        单个模型副本的前向。

        Returns:
            A **list** of outputs — never a bare tensor.
            输出的 **list** —— 绝不能直接返回裸张量。
        """
        # REPLACE: unpack however your split_data() packed the inputs.
        # REPLACE：按你的 split_data() 打包方式解包输入。
        return [model(input_data[0])]

    def accuracy(self, output_data: List, label_data: List, model_id: int) -> List:
        """Compute metrics shown on the progress bar and in the epoch summary.

        计算进度条与 epoch 汇总里显示的指标。
        """
        pred, gt = output_data[self.ID_PRED], label_data[0]
        # REPLACE with your own metric. 换成你自己的指标。
        return [1.0 - torch.mean(torch.abs(pred - gt))]

    def loss(self, output_data: List, label_data: List, model_id: int) -> List:
        """Compute the loss tensors used for backprop.

        计算用于反向传播的损失张量。

        Returns:
            ``[total_loss, *components]`` — element 0 is what gets backwarded.
            ``[总损失, *分量]`` —— 下标 0 是实际反传的那个。
        """
        pred, gt = output_data[self.ID_PRED], label_data[0]
        total = F.mse_loss(pred, gt)
        return [total]

    # ------------------------------------------------------------------
    # Optional hooks below. Names are validated by JackFramework at class
    # definition time — a typo raises TypeError instead of being skipped.
    # 以下为可选 hook。名字由 JackFramework 在类定义时校验 ——
    # 拼错会直接抛 TypeError，而不是被静默跳过。
    # ------------------------------------------------------------------
    def pretreatment(self, epoch: int, rank: object) -> None:
        """Run before each training epoch. 每个训练 epoch 之前执行。"""

    def post_process(self, epoch: int, rank: object,
                     ave_tower_loss: List, ave_tower_acc: List) -> None:
        """Run after each epoch. 每个 epoch 之后执行。

        Warning:
            The name is ``post_process`` with an underscore. ``postprocess``
            is rejected at class-definition time.
            方法名是带下划线的 ``post_process``；写成 ``postprocess``
            会在类定义时被直接拒绝。
        """

    def load_model(self, model: object, checkpoint: dict, model_id: int) -> bool:
        """Return True to take over loading; False lets JackFramework do it.

        返回 True 表示自己接管加载；False 交回框架默认逻辑。
        """
        return False

    def load_opt(self, opt: object, checkpoint: dict, model_id: int) -> bool:
        """Same contract as load_model, for the optimizer state.

        与 load_model 同约定，针对优化器状态。
        """
        return False

    def save_model(self, epoch: int, model_list: Sequence, opt_list: Sequence) -> dict:
        """Return a dict to override what gets serialised; None uses defaults.

        返回 dict 可覆盖序列化内容；返回 None 走框架默认。
        """
        return None
