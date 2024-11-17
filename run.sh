#!/bin/bash

# 檢查是否有足夠的參數輸入
if [ "$#" -ne 4 ]; then
    echo "Usage: bash ./run.sh /path/to/model /path/to/adapter_checkpoint /path/to/input.json /path/to/output.json"
    exit 1
fi

# 設定變數
MODEL_PATH="${1}"
CHECKPOINT_PATH="${2}"
INPUT_PATH="${3}"
OUTPUT_PATH="${4}"

# 執行 predict.py 並傳遞參數
echo "Running inference with model: $MODEL_PATH, checkpoint: $CHECKPOINT_PATH, input: $INPUT_PATH, output: $OUTPUT_PATH"
python3 predict.py --model "$MODEL_PATH" --adapter_checkpoint "$CHECKPOINT_PATH" --input "$INPUT_PATH" --output "$OUTPUT_PATH"
