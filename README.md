# Template-jf

![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg?style=plastic)
![PyTorch 2.x](https://img.shields.io/badge/PyTorch%202.x-%23EE4C2C.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=plastic)

> Project template for **[JackFramework](https://github.com/Archaic-Atom/JackFramework)**.
> JackFramework owns the training loop, DDP launcher, checkpoint I/O, logging,
> progress bar and resume; you supply a model interface, a dataloader and a launcher.
>
> JackFramework 项目模板。框架负责训练循环、DDP 启动、checkpoint 读写、日志、
> 进度条和续训；你只需提供模型接口、dataloader 和启动脚本。

**This template runs as-is.** Out of the box it trains a tiny MLP on synthetic
tensors, so you can verify your environment before writing a line of your own code.
Replace the parts marked `REPLACE`.

**本模板开箱即跑。** 出厂状态会用合成张量训练一个极小的 MLP —— 在动手写自己的
代码之前就能验证环境是否正常。把标了 `REPLACE` 的部分换掉即可。

---

## Quick start / 快速开始

```bash
# 1. Install JackFramework  安装框架
git clone https://github.com/Archaic-Atom/JackFramework && cd JackFramework && ./install.sh

# 2. Smoke-test this template (CPU is fine)  冒烟测试（CPU 即可）
cd Template-jf
bash Scripts/start_train_dataset_model.sh     # trains 10 epochs on synthetic data
bash Scripts/start_test_dataset_model.sh      # loads the last checkpoint and runs test
```

If both finish with `The Application has finished successfully.` your setup is good.

两条都以 `The Application has finished successfully.` 结束就说明环境没问题。

---

## What to edit / 要改哪些文件

| File | Role |
|---|---|
| `Source/UserModelImplementation/Models/your_model/inference.py` | your network + loss + metric |
| `Source/UserModelImplementation/Models/__init__.py` | register it under a `--modelName` key |
| `Source/UserModelImplementation/Dataloaders/your_dataloader.py` | your dataset + batch splitting |
| `Source/UserModelImplementation/Dataloaders/__init__.py` | register it under a `--dataset` key |
| `Source/UserModelImplementation/user_interface.py` | extra CLI flags |
| `Scripts/start_*_dataset_model.sh` | launch parameters |

---

## Contract rules that actually bite / 真正会坑到人的几条约定

1. **`inference()` must return a LIST**, even for a single model — `[output]`.
   `loss()` and `accuracy()` receive that list, not the tensor. Unwrap with
   `output_data[self.ID_PRED]` (index 0) first.
   **`inference()` 必须返回 list**，哪怕只有一个模型。`loss()` / `accuracy()`
   收到的是这个 list 而不是张量，先用 `output_data[self.ID_PRED]`（下标 0）解包。

2. **Hook names are validated at class-definition time.** `post_process` has an
   underscore; writing `postprocess` now raises `TypeError` immediately instead of
   being silently skipped. Same for `pretreatment`, `load_model`, `load_opt`,
   `save_model`.
   **hook 名字在类定义时就会被校验。** `post_process` 带下划线；写成 `postprocess`
   会立刻抛 `TypeError`，不再是静默跳过。

3. **Zoo keys must match the launcher.** `--modelName` must equal a key in
   `model_zoo`, `--dataset` a key in `dataloaders_zoo`. A mismatch fails with a
   bare `AssertionError` and no hint.
   **zoo 的 key 必须和启动脚本一致**，对不上只会抛一个无提示的 `AssertionError`。

4. **`user_parser()` must return the parser.** JackFramework only keeps the return
   value if it is an `argparse.ArgumentParser`; returning `None` silently discards
   every flag you added.
   **`user_parser()` 必须把 parser 返回**，返回 `None` 会让你加的参数被静默丢弃。

5. **`--gpu` is a COUNT, not a device list.** Select devices with
   `CUDA_VISIBLE_DEVICES`. `--port` must differ between concurrent runs.
   **`--gpu` 是数量不是设备列表**；设备用 `CUDA_VISIBLE_DEVICES` 选，
   并发任务的 `--port` 必须错开。

6. **Always debug single-process first** (`--dist False --gpu 0/1`). DDP swallows
   tracebacks — an empty `ProcessRaisedException` usually means a registered
   parameter got no gradient (frozen backbone), not a C-level crash.
   **永远先单进程调试**。DDP 会吞掉 traceback —— 空的 `ProcessRaisedException`
   通常是有注册参数没拿到梯度（冻结的 backbone），不是 C 层崩溃。

7. **`get_train_dataset(path, is_training)` also builds the TEST set** when called
   with `is_training=False`. The name is misleading; honour the flag.
   **`get_train_dataset` 在 `is_training=False` 时构建的是测试集**，名字有误导性。

8. **`--modelDir` accepts a directory or a `.pth` file.** A directory without
   `checkpoint.list` inside fails with `Checkpoint list file not found` and loads
   nothing. The FIRST line of that file is what loads next — not the
   highest-numbered file.
   **`--modelDir` 既可给目录也可给 `.pth`。** 给目录但里面没有 `checkpoint.list`
   会报 `Checkpoint list file not found` 且什么都不加载；该文件**第一行**才是
   下次加载的对象，不是编号最大的那个。

---

## File structure / 目录结构

```
Template-jf
├── Datasets/                 training/testing list csv
├── Scripts/
│   ├── clean.sh              clean build artifacts
│   ├── generate_path.sh      generate the training/testing list
│   ├── kill_process.sh
│   ├── start_train_dataset_model.sh
│   └── start_test_dataset_model.sh
├── Source/
│   ├── UserModelImplementation/
│   │   ├── Models/
│   │   │   ├── __init__.py           model_zoo  (--modelName)
│   │   │   └── your_model/
│   │   │       └── inference.py      model interface  <- edit
│   │   ├── Dataloaders/
│   │   │   ├── __init__.py           dataloaders_zoo  (--dataset)
│   │   │   └── your_dataloader.py    data interface   <- edit
│   │   ├── user_define.py            global constants
│   │   └── user_interface.py         wires the two zoos together
│   ├── Tools/
│   └── main.py
├── LICENSE
└── README.md
```

Note: directory names are lowercase (`your_model`, not `Your_Model`). macOS is
case-insensitive and will hide a mismatch that breaks the import on Linux.

注意：目录名用小写（`your_model` 而非 `Your_Model`）。macOS 不区分大小写会掩盖
这个问题，但在 Linux 上会直接 import 失败。

---

## Update log

### 2026-07-28
1. Fixed `postprocess` -> `post_process` — the old name is now rejected by
   JackFramework's hook-name validation at class-definition time.
2. Fixed zoo keys vs launcher flags (`YourMode`/`your_model`/`dataset_name` all
   disagreed); launchers now use `YourModel` / `YourDataloader`.
3. Fixed the never-matching `case('')` default branch, which left `model`
   unassigned and raised `UnboundLocalError` on a wrong `--modelName`.
4. Renamed `Models/Your_Model/` -> `Models/your_model/` to match what git tracked;
   the old mixed case broke the import on case-sensitive filesystems.
5. `user_parser()` now returns the parser instead of `None`.
6. Removed the duplicated `__str2bool` (the base class provides `_str2bool`).
7. Launchers no longer hardcode an 8-GPU box, a Scene Flow list path that is not in
   this repo, or `--imgNum 35454`; defaults now run on CPU against the bundled
   synthetic dataset.
8. Model and dataloader now ship a runnable tiny example instead of returning
   empty lists, so the template can be smoke-tested end to end.
9. README rewritten — the previous embedded snippets contained `postprocess` and
   `accuary` (sic), both of which fail immediately on current JackFramework.

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
