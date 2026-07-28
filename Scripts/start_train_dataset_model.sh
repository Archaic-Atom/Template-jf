#!/bin/bash
# ============================================================
# Train launcher for the JackFramework template.
# JackFramework 模板的训练启动脚本。
#
# Ships with values that RUN AS-IS on the bundled synthetic dataset
# (CPU or 1 GPU). Change them once you plug in your own data/model.
# 出厂参数在自带的合成数据集上**开箱即跑**（CPU 或单卡）。
# 接入自己的数据和模型后再改。
#
# NOTE: --modelName / --dataset must match the keys registered in
#   Source/UserModelImplementation/Models/__init__.py      (model_zoo)
#   Source/UserModelImplementation/Dataloaders/__init__.py (dataloaders_zoo)
# A mismatch fails with a bare AssertionError and no hint.
# 注意：--modelName / --dataset 必须与两个 zoo 里注册的 key 一致，
# 对不上只会抛一个无提示的 AssertionError。
# ============================================================
set -e

model_name='YourModel'          # must match model_zoo key
dataset_name='YourDataloader'   # must match dataloaders_zoo key
train_list_path='./Datasets/dataset_example_training_list.csv'

out_dir='./Result/'
model_dir='./Checkpoint/'
log_dir='./log/'
dist_port=8800

mkdir -p ${out_dir} ${model_dir} ${log_dir}

# Single-process debug run. ALWAYS get this passing before enabling DDP —
# DDP swallows tracebacks and turns real errors into empty exceptions.
# 单进程调试跑法。开 DDP 之前**务必**先让这条跑通 ——
# DDP 会吞掉 traceback，把真实错误变成空异常。
echo "Begin to train the model (single process)!"
python -u Source/main.py \
    --mode train \
    --dist False \
    --gpu 0 \
    --batchSize 4 \
    --lr 0.001 \
    --maxEpochs 10 \
    --imgNum 64 \
    --valImgNum 16 \
    --auto_save_num 1 \
    --dataloaderNum 0 \
    --modelName ${model_name} \
    --dataset ${dataset_name} \
    --trainListPath ${train_list_path} \
    --modelDir ${model_dir} \
    --outputDir ${out_dir} \
    --log ${log_dir}

# ------------------------------------------------------------
# Multi-GPU (DDP) variant — enable only after the run above works.
# 多卡 (DDP) 版本 —— 上面那条跑通之后再启用。
#
# --gpu is a COUNT, not a device list; pick devices via CUDA_VISIBLE_DEVICES.
# --port must differ between concurrent runs or they collide.
# --gpu 是**数量**不是设备列表；设备用 CUDA_VISIBLE_DEVICES 选。
# --port 并发多个任务时必须错开，否则会撞。
# ------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u Source/main.py \
#     --mode train --dist True --gpu 4 --port ${dist_port} \
#     --batchSize 8 --lr 0.001 --maxEpochs 200 \
#     --imgNum 35454 --valImgNum 4370 --auto_save_num 1 \
#     --dataloaderNum 8 \
#     --modelName ${model_name} --dataset ${dataset_name} \
#     --trainListPath ${train_list_path} \
#     --modelDir ${model_dir} --outputDir ${out_dir} --log ${log_dir} \
#     > TrainRun.log 2>&1 &
# echo "Use (tail -f TrainRun.log) to watch the training process."
