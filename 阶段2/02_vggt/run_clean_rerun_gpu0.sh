#!/bin/bash
# Clean rerun P0-2(plant_view 部分):只监控 GPU0,等空闲 >=39GB 后跑 6 个长序列。
# 用法: nohup bash run_clean_rerun_gpu0.sh > clean_rerun_pv.log 2>&1 &
set -u
export VGGT_RUN_ID="clean_rerun_20260826T065510Z"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
PY=/home/test/miniconda3/envs/vggt_lora/bin/python
BASE="/fj/VGGT+head+lora实验/阶段2"
cd "$BASE/02_vggt"

LONG=$(ls "$BASE/01_sequences/sequences/plant_view/"*.json)

while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    echo "[wait] GPU0 free ${FREE} MiB, need 39000 ($(date +%H:%M:%S))"
    [ "$FREE" -ge 39000 ] && break
    sleep 300
done

echo "=== plant_view sequences (6) on GPU0, starting $(date) ==="
$PY run_vggt_inference.py $LONG --out-base "$BASE/02_vggt/v2_clean_rerun" || exit 1
echo "=== ALL DONE $(date) ==="
