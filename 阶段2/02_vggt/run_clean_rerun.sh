#!/bin/bash
# Clean rerun P0-2:等 GPU 空闲后重跑全部 17 个 ready 序列(vggt_lora)。
# 用法: nohup bash run_clean_rerun.sh > clean_rerun.log 2>&1 &
set -u
export VGGT_RUN_ID="clean_rerun_20260826T065510Z"
export HF_ENDPOINT=https://hf-mirror.com
PY=/home/test/miniconda3/envs/vggt_lora/bin/python
BASE="/fj/VGGT+head+lora实验/阶段2"
SEQS="$BASE/01_sequences/sequences"

cd "$BASE/02_vggt"

# 返回第一块空闲显存 >= $1 MiB 的 GPU 编号,否则返回空
pick_gpu() {
    MIN=$1
    for i in 0 1; do
        FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $i)
        [ "$FREE" -ge "$MIN" ] && { echo $i; return 0; }
    done
    return 1
}

wait_gpu() {
    NEED=$1
    while true; do
        G=$(pick_gpu "$NEED")
        if [ -n "$G" ]; then echo "[wait_gpu] using GPU $G ($(date +%H:%M:%S))" >&2; echo "$G"; return 0; fi
        echo "[wait_gpu] need ${NEED}MiB, none free ($(date +%H:%M:%S))" >&2
        sleep 600
    done
}

SHORT=$(ls "$SEQS"/wheat3dgs/*.json "$SEQS"/mustc/*.json)
LONG=$(ls "$SEQS"/plant_view/*.json)

echo "=== short sequences (11, need >=12GB) ==="
G=$(wait_gpu 12000)
echo "CUDA_VISIBLE_DEVICES=$G (GPU0 has 30GB, should work)"
CUDA_VISIBLE_DEVICES=$G $PY run_vggt_inference.py $SHORT --out-base "$BASE/02_vggt/v2_clean_rerun" || exit 1

echo "=== plant_view sequences (6, full sequence, need >=39GB) ==="
while true; do
    G=$(wait_gpu 39000)
    CUDA_VISIBLE_DEVICES=$G $PY run_vggt_inference.py $LONG --out-base "$BASE/02_vggt/v2_clean_rerun" && break
    echo "[plant_view] OOM or partial failure, will retry missing after cooldown" >&2; sleep 300
done

echo "=== ALL DONE $(date) ==="
