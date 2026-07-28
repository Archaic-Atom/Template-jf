#!/bin/bash
# Remove run artifacts. 清理运行产物。
# Safe to run at any time; nothing here is version-controlled.
# 任何时候都可以跑；这里清掉的东西都不在版本控制里。
echo "Start to clean the project"

rm -rf Result ResultImg Checkpoint log
rm -f ./*.log
find . -name '*.pyc' -delete
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null

echo "Finish"
