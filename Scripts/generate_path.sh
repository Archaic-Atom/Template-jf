#!/bin/bash
# Generate the training/testing list CSV.
# 生成训练/测试清单 CSV。
#
# Point --root_path at your dataset, then make sure the CSV columns match
# what get_train_dataset() expects.
# 把 --root_path 指向你的数据集，并确保 CSV 的列与
# get_train_dataset() 的读取方式一致。
set -e

root_path='/path/to/your/dataset'      # REPLACE
output='./Datasets/dataset_example_training_list.csv'

python Source/Tools/generate_train_list.py \
    --root_path "${root_path}" \
    --output "${output}"
