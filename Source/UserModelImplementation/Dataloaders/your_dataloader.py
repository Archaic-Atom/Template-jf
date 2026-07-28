# -*- coding: utf-8 -*-
"""Dataloader skeleton for a JackFramework project.

JackFramework 项目的 dataloader 骨架。

As shipped this yields synthetic tensors so the template runs end-to-end
without any dataset on disk. Replace ``_SyntheticDataset`` and
``split_data`` with your own data pipeline.

模板出厂产出合成张量，因此不需要任何磁盘数据集就能跑通全流程。
把 ``_SyntheticDataset`` 和 ``split_data`` 换成你自己的数据管线即可。
"""

import time
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

import JackFramework as jf
# import UserModelImplementation.user_define as user_def


class _SyntheticDataset(Dataset):
    """REPLACE — random tensors so a fresh clone is runnable.

    REPLACE —— 随机张量，保证新 clone 下来即可运行。
    """

    def __init__(self, length: int, in_shape: Tuple[int, int, int], out_dim: int) -> None:
        super().__init__()
        self._length = max(int(length), 1)
        self._in_shape = in_shape
        self._out_dim = out_dim

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        return (torch.randn(self._in_shape, dtype=torch.float32),
                torch.randn(self._out_dim, dtype=torch.float32))


class YourDataloader(jf.UserTemplate.DataHandlerTemplate):
    """User-side data contract consumed by JackFramework.

    JackFramework 使用的用户侧数据契约。
    """

    _IN_SHAPE = (3, 32, 32)   # REPLACE
    _OUT_DIM = 10             # REPLACE

    def __init__(self, args: object) -> None:
        super().__init__(args)
        self.__args = args
        self.__result_str = jf.ResultStr()
        self.__train_dataset = None
        self.__val_dataset = None
        self.__start_time = 0

    def get_train_dataset(self, path: str, is_training: bool = True) -> object:
        """Build the training dataset.

        构建训练集。

        Note:
            JackFramework also calls this with ``is_training=False`` to build
            the **test** set — the name is misleading, honour the flag.
            JackFramework 也会用 ``is_training=False`` 调它来构建**测试集** ——
            方法名有误导性，务必按这个标志分支。
        """
        # REPLACE: read `path` (the --trainListPath csv) and build your dataset.
        # REPLACE：读取 `path`（即 --trainListPath 指向的 csv）构建数据集。
        self.__train_dataset = _SyntheticDataset(
            getattr(self.__args, 'imgNum', 64), self._IN_SHAPE, self._OUT_DIM)
        return self.__train_dataset

    def get_val_dataset(self, path: str) -> object:
        """Build the validation dataset. 构建验证集。"""
        self.__val_dataset = _SyntheticDataset(
            getattr(self.__args, 'valImgNum', 16), self._IN_SHAPE, self._OUT_DIM)
        return self.__val_dataset

    def split_data(self, batch_data: Tuple, is_training: bool) -> List:
        """Split one dataloader batch into the lists the model will receive.

        把一个 batch 拆成模型将要收到的两个 list。

        Returns:
            Training: ``(input_list, label_list)``.
            Test: ``(input_list, supplement_list)`` — supplement carries
            whatever ``save_result`` needs (file names, sizes, ...).
            训练模式返回 ``(输入 list, 标签 list)``；测试模式返回
            ``(输入 list, 补充信息 list)``，补充信息给 ``save_result`` 用
            （文件名、尺寸等）。
        """
        self.__start_time = time.time()
        x, y = batch_data
        if is_training:
            return [x], [y]
        return [x], [y]

    def show_train_result(self, epoch: int, loss: List, acc: List, duration: float) -> None:
        """Log the aggregated training result of one epoch.

        输出一个 epoch 的训练汇总。
        """
        assert len(loss) == len(acc)   # same model number / 模型数量一致
        info_str = self.__result_str.training_result_str(
            epoch, loss[0], acc[0], duration, True)
        jf.log.info(info_str)

    def show_val_result(self, epoch: int, loss: List, acc: List, duration: float) -> None:
        """Log the aggregated validation result of one epoch.

        输出一个 epoch 的验证汇总。
        """
        assert len(loss) == len(acc)
        info_str = self.__result_str.training_result_str(
            epoch, loss[0], acc[0], duration, False)
        jf.log.info(info_str)

    def save_result(self, output_data: List, supplement: List,
                    img_id: int, model_id: int) -> None:
        """Persist one batch of predictions (``--mode test`` only).

        落盘一个 batch 的预测结果（仅 ``--mode test``）。

        Note:
            ``output_data`` is the **list** returned by ``inference`` —
            unwrap it before use. When computing a global sample index, use
            the configured ``args.batchSize``, not the actual tensor length,
            or the partial last batch will collide with earlier indices.
            ``output_data`` 是 ``inference`` 返回的 **list**，先解包再用。
            算全局样本下标时用配置的 ``args.batchSize``，不要用张量实际长度，
            否则最后一个不满的 batch 会和前面的下标撞车。
        """
        # REPLACE: write predictions to --resultImgDir / a csv.
        # REPLACE：把预测写到 --resultImgDir 或 csv。

    def show_intermediate_result(self, epoch: int, loss: List, acc: List) -> str:
        """Return the one-line string shown on the progress bar.

        返回进度条上显示的一行摘要。
        """
        assert len(loss) == len(acc)
        return self.__result_str.training_intermediate_result(epoch, loss[0], acc[0])
