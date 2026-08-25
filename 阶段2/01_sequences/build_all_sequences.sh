#!/bin/bash
# 一键生成全部 sequence.json。禁止覆盖:重复运行会因文件已存在而报错。
set -e
cd "$(dirname "$0")/adapters"
PY=${PY:-python3}

echo "=== plant_view ==="
$PY adapt_plant_view.py
echo "=== wheat3dgs ==="
$PY adapt_wheat3dgs.py
echo "=== mustc ==="
$PY adapt_mustc.py
echo "=== terraref ==="
$PY adapt_terraref.py --seasons season_6 season_4 --max-seqs 5

echo "ALL DONE"
