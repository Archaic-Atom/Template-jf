#!/bin/bash
# ============================================================
# Test launcher for the JackFramework template.
# JackFramework 模板的测试启动脚本。
#
# --modelDir accepts EITHER a directory (JF reads checkpoint.list inside)
# OR a .pth file directly. A directory without checkpoint.list fails with
# "Checkpoint list file not found" and nothing loads.
# --modelDir 既可以给**目录**（JF 读里面的 checkpoint.list），
# 也可以直接给 **.pth 文件**。给了目录却没有 checkpoint.list 会报
# "Checkpoint list file not found" 且什么都不加载。
# ============================================================
set -e

model_name='YourModel'          # must match model_zoo key
dataset_name='YourDataloader'   # must match dataloaders_zoo key
test_list_path='./Datasets/dataset_example_training_list.csv'

out_dir='./Result/'
model_dir='./Checkpoint/'
log_dir='./log/'

mkdir -p ${out_dir} ${log_dir}

echo "Begin to test the model!"
python -u Source/main.py \
    --mode test \
    --dist False \
    --gpu 0 \
    --batchSize 4 \
    --imgNum 16 \
    --dataloaderNum 0 \
    --modelName ${model_name} \
    --dataset ${dataset_name} \
    --trainListPath ${test_list_path} \
    --modelDir ${model_dir} \
    --outputDir ${out_dir} \
    --resultImgDir ${out_dir} \
    --log ${log_dir}

# If the run finishes but no final CSV appears, the per-rank temp files
# already hold the predictions: <outputDir>/.tmp_test_*_rank<N>.csv
# (leading dot — plain `ls` hides them).
# 如果跑完没有最终 CSV，预测其实已经在各 rank 的临时文件里：
# <outputDir>/.tmp_test_*_rank<N>.csv（前导点，普通 ls 看不见）。
