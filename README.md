# Template-jf

![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg?style=plastic)
![PyTorch 2.x](https://img.shields.io/badge/PyTorch%202.x-%23EE4C2C.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

Project template for **[JackFramework](https://github.com/Archaic-Atom/JackFramework)**.
The framework owns the training loop, DDP launcher, checkpoint I/O, logging, progress
bar and resume; you supply a model interface, a dataloader and a launcher script.

**[JackFramework](https://github.com/Archaic-Atom/JackFramework) 的项目模板。**
框架负责训练循环、DDP 启动、checkpoint 读写、日志、进度条和续训；
你只需提供模型接口、dataloader 和启动脚本。

> **This template runs as-is.** Out of the box it trains a tiny MLP on synthetic
> tensors, so you can verify your environment before writing any of your own code.
> Everything you need to replace is marked `REPLACE`.
>
> **本模板开箱即跑。** 出厂状态会用合成张量训练一个极小的 MLP —— 在动手写自己的
> 代码之前就能先验证环境是否正常。所有需要替换的地方都标了 `REPLACE`。

---

## 1. Quick start / 快速开始

```bash
# Install JackFramework  /  安装框架
git clone https://github.com/Archaic-Atom/JackFramework
cd JackFramework && ./install.sh && cd ..

# Smoke-test this template (CPU is enough)  /  冒烟测试（CPU 即可）
cd Template-jf
bash Scripts/start_train_dataset_model.sh    # 10 epochs on synthetic data
bash Scripts/start_test_dataset_model.sh     # load last checkpoint, run test
```

**EN** — If both end with `The Application has finished successfully.`, your
environment is correct and you can start replacing the placeholder code.

**中文** —— 两条命令都以 `The Application has finished successfully.` 结束，
就说明环境没问题，可以开始替换占位代码了。

---

## 2. What to edit / 需要改哪些文件

| File / 文件 | Role / 作用 |
|---|---|
| `Source/UserModelImplementation/Models/your_model/inference.py` | your network, loss, metric / 网络、损失、指标 |
| `Source/UserModelImplementation/Models/__init__.py` | register it under a `--modelName` key / 注册 `--modelName` 的 key |
| `Source/UserModelImplementation/Dataloaders/your_dataloader.py` | your dataset and batch splitting / 数据集与 batch 拆分 |
| `Source/UserModelImplementation/Dataloaders/__init__.py` | register it under a `--dataset` key / 注册 `--dataset` 的 key |
| `Source/UserModelImplementation/user_interface.py` | extra CLI flags / 自定义命令行参数 |
| `Source/UserModelImplementation/user_define.py` | shared constants / 共用常量 |
| `Scripts/start_*_dataset_model.sh` | launch parameters / 启动参数 |
| `Scripts/generate_path.sh` | build the dataset list CSV / 生成数据清单 CSV |

---

## 3. The contract / 用户侧契约

### Model interface / 模型接口
`ModelHandlerTemplate` subclass. Required: `get_model`, `inference`, `optimizer`,
`lr_scheduler`, `loss`, `accuracy`. Optional: `pretreatment`, `post_process`,
`load_model`, `load_opt`, `save_model`.

`ModelHandlerTemplate` 的子类。必须实现：`get_model`、`inference`、`optimizer`、
`lr_scheduler`、`loss`、`accuracy`。可选：`pretreatment`、`post_process`、
`load_model`、`load_opt`、`save_model`。

### Dataloader interface / 数据接口
`DataHandlerTemplate` subclass. Required: `get_train_dataset`, `get_val_dataset`,
`split_data`, `show_train_result`, `show_val_result`, `save_result`,
`show_intermediate_result`.

`DataHandlerTemplate` 的子类。必须实现：`get_train_dataset`、`get_val_dataset`、
`split_data`、`show_train_result`、`show_val_result`、`save_result`、
`show_intermediate_result`。

---

## 4. Rules that actually bite / 真正会坑到人的约定

**① `inference()` must return a LIST / 必须返回 list**

**EN** — Even with a single model, return `[output]`. `loss()` and `accuracy()`
receive that *list*, not the tensor — unwrap with `output_data[self.ID_PRED]`
(index 0) first. If a check like `'key' in output_data` is always False, this is why.

**中文** —— 哪怕只有一个模型也要返回 `[output]`。`loss()` / `accuracy()` 收到的是
这个 **list** 而非张量，先用 `output_data[self.ID_PRED]`（下标 0）解包。
如果 `'key' in output_data` 恒为 False，原因就在这里。

**② Hook names are validated at class-definition time / hook 名字在类定义时就会校验**

**EN** — `post_process` has an underscore. Writing `postprocess` now raises
`TypeError` immediately. Older JackFramework silently skipped the method instead,
which was far harder to diagnose.

**中文** —— `post_process` 带下划线。写成 `postprocess` 现在会立刻抛 `TypeError`。
旧版 JackFramework 是**静默跳过**该方法，那种情况排查起来困难得多。

**③ Zoo keys must match the launcher / zoo 的 key 必须和启动脚本一致**

**EN** — `--modelName` must equal a key in `model_zoo`, `--dataset` a key in
`dataloaders_zoo`. A mismatch fails with a bare `AssertionError` and no hint.

**中文** —— `--modelName` 要等于 `model_zoo` 里的 key，`--dataset` 要等于
`dataloaders_zoo` 里的 key。对不上只会抛一个没有任何提示的 `AssertionError`。

**④ `user_parser()` must return the parser / 必须把 parser 返回**

**EN** — JackFramework keeps the return value only when it is an
`argparse.ArgumentParser`. Returning `None` silently discards every flag you added.

**中文** —— JackFramework 只在返回值是 `argparse.ArgumentParser` 时才采用它。
返回 `None` 会让你添加的所有参数被静默丢弃。

**⑤ `--gpu` is a COUNT, not a device list / `--gpu` 是数量不是设备列表**

**EN** — Select devices with `CUDA_VISIBLE_DEVICES`. `--port` is the DDP rendezvous
port; vary it across concurrent runs or they collide.

**中文** —— 设备用 `CUDA_VISIBLE_DEVICES` 选。`--port` 是 DDP 的会合端口，
并发多个任务时必须错开，否则会撞。

**⑥ Always debug single-process first / 永远先单进程调试**

**EN** — Use `--dist False --gpu 0`. DDP swallows tracebacks; an empty
`ProcessRaisedException` almost always means a registered parameter received no
gradient (a frozen backbone is the usual cause), not a C-level crash.

**中文** —— 先用 `--dist False --gpu 0`。DDP 会吞掉 traceback；空的
`ProcessRaisedException` 几乎总是"有注册参数没拿到梯度"（通常是冻结的 backbone），
而不是 C 层崩溃。

**⑦ `get_train_dataset` also builds the TEST set / 它也用来构建测试集**

**EN** — It is called with `is_training=False` for the test set. The name is
misleading; branch on the flag.

**中文** —— 测试集是通过 `is_training=False` 调同一个方法构建的。
方法名有误导性，务必按这个标志分支。

**⑧ `--modelDir` takes a directory OR a `.pth` / 可给目录也可给文件**

**EN** — A directory must contain `checkpoint.list`, otherwise you get
`Checkpoint list file not found` and nothing loads. The **first line** of that file
is what loads next — not the highest-numbered file in the directory.

**中文** —— 给目录时里面必须有 `checkpoint.list`，否则会报
`Checkpoint list file not found` 且什么都不加载。该文件的**第一行**才是下次加载的
对象，不是目录里编号最大的那个。

**⑨ Directory names are lowercase / 目录名一律小写**

**EN** — `your_model`, not `Your_Model`. macOS is case-insensitive and will hide a
mismatch that breaks the import on Linux.

**中文** —— 用 `your_model` 而非 `Your_Model`。macOS 不区分大小写会掩盖这个问题，
但在 Linux 服务器上会直接 import 失败。

---

## 5. File structure / 目录结构

```
Template-jf
├── Datasets/                            list CSV / 数据清单
├── Scripts/
│   ├── clean.sh                         clean artifacts / 清理产物
│   ├── generate_path.sh                 build the list CSV / 生成清单
│   ├── kill_process.sh                  kill by port / 按端口杀进程
│   ├── start_train_dataset_model.sh     train / 训练
│   └── start_test_dataset_model.sh      test / 测试
├── Source/
│   ├── UserModelImplementation/
│   │   ├── Models/
│   │   │   ├── __init__.py              model_zoo      (--modelName)
│   │   │   └── your_model/inference.py  model interface  <- edit / 待改
│   │   ├── Dataloaders/
│   │   │   ├── __init__.py              dataloaders_zoo (--dataset)
│   │   │   └── your_dataloader.py       data interface   <- edit / 待改
│   │   ├── user_define.py               shared constants / 共用常量
│   │   └── user_interface.py            wires the two zoos / 接合两个注册表
│   ├── Tools/generate_train_list.py     list generator / 清单生成器
│   └── main.py                          entry point / 入口
├── LICENSE
└── README.md
```

---

## 6. Update log / 更新日志

### 2026-07-28

**EN** — The template dated from 2022 and no longer imported, let alone ran,
against the current JackFramework. Thirteen defects were found by running it:

**中文** —— 模板停留在 2022 年，在当前版本的 JackFramework 下**连导入都无法通过**，
更谈不上运行。通过实际运行共发现 13 处缺陷：

1. `postprocess` → `post_process`. JackFramework now validates hook names in
   `__init_subclass__`, so the old spelling raised `TypeError` at class definition —
   the template failed at import, before any training could start.
   JackFramework 现在会在 `__init_subclass__` 里校验 hook 名，旧拼写在类定义阶段
   就抛 `TypeError`，模板连导入都过不了。
2. Zoo keys disagreed with the launchers in three places
   (`YourMode` / `your_model` / `dataset_name`). Standardised on `YourModel` and
   `YourDataloader`.
   zoo 的 key 与启动脚本有三处不一致，已统一。
3. The `case('')` default branch never matched, so a wrong `--modelName` left
   `model` unassigned and raised `UnboundLocalError` instead of the intended error
   log. Replaced with the no-arg `case()` default.
   `case('')` 默认分支永不命中，名字传错时是 `UnboundLocalError` 而非预期的报错。
4. `Models/Your_Model/` vs the git-tracked `Models/your_model/` — a case clash
   invisible on macOS that breaks the import on Linux. Unified to lowercase.
   大小写冲突在 macOS 上看不出来，但在 Linux 上会 import 失败，已统一为小写。
5. `user_parser()` returned `None`, so every user-added CLI flag was silently
   discarded.
   `user_parser()` 返回 `None`，导致自定义参数被静默丢弃。
6. Removed the duplicated private `__str2bool`; the base class provides `_str2bool`.
   删除重复的 `__str2bool`，基类已提供 `_str2bool`。
7. Launchers hardcoded `CUDA_VISIBLE_DEVICES=4,5,6,7 --gpu 4 --dist True`,
   `--imgNum 35454`, and a Scene Flow list path absent from this repo.
   启动脚本硬编码了 8 卡机器、Scene Flow 的样本数，以及本仓库根本没有的清单路径。
8. Model and dataloader returned empty lists, so the template could not be run or
   debugged at all. Both now ship a runnable tiny example marked `REPLACE`.
   模型和 dataloader 全部返回空 list，模板根本无法运行，也就无从调试。
9. `Scripts/generate_path.sh` called `Source/Tools/your_python_file_path.py`, a
   file that does not exist. Added a working `generate_train_list.py`.
   `generate_path.sh` 调用了不存在的脚本，已补上可用的 `generate_train_list.py`。
10. The CI workflow ran `Scripts/start_debug_stereo_net.sh`, which does not exist,
    so the build badge could never pass. It now runs the real smoke test.
    CI 调用了不存在的脚本，构建永远不可能通过；现在改为运行真正的冒烟测试。
11. `Scripts/clean.sh` used `rm -r` without `-f` (failing when a directory was
    absent) and did not clean `Checkpoint/`.
    `clean.sh` 的 `rm -r` 缺 `-f`，且漏清 `Checkpoint/`。
12. Malformed encoding cookies in `main.py` and `user_define.py`.
    `main.py` 与 `user_define.py` 的编码声明写残了。
13. README embedded the old snippets verbatim, including `postprocess` and
    `accuary` (sic) — copying from it produced immediately-broken code.
    README 内嵌的示例代码含 `postprocess` 和拼错的 `accuary`，照抄即坏。

**Verified / 验证** — JackFramework @ 019, Python 3.11 / torch 2.5.1, CPU:
`start_train_dataset_model.sh` trains 10 epochs and saves checkpoints;
`start_test_dataset_model.sh` loads `model_epoch_9.pth` via `checkpoint.list` and
completes. Both end with `The Application has finished successfully.`

### 2021-05-29
1. Add the depth for transformer;
2. Fork the JackFramework to a new project;
3. Remove the JackFramework from this project.

### 2021-04-08
1. Add the stereo;
2. Add transformer.

### 2021-01-13
1. Fork a new project (based on pytorch);
2. Use a new code style;
3. Build the frameworks for pytorch;
4. Write ReadMe
