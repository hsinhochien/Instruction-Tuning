#!/bin/bash

FOLDER_ID="https://drive.google.com/drive/folders/1-ANKZoGTECfX3uf_xzqGRw87Zu4JgWDA?usp=sharing"

# 定義下載後存放的目錄
MODEL_DIR="./adapter_checkpoint"

# 檢查第一個模型目錄是否存在，不存在則創建
if [ ! -d "$MODEL_DIR" ]; then
    echo "Creating model directory: $MODEL_DIR"
    mkdir -p "$MODEL_DIR"
fi

# 下載Google雲端硬碟資料夾的所有內容
echo "Downloading adapter_checkpoint from Google Drive..."
gdown --folder "$FOLDER_ID" -O "$MODEL_DIR"

echo "Model parameters downloaded to $MODEL_DIR"