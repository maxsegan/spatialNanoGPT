#!/bin/bash
# Script to run the GPT2 sparsity evaluation workflow

set -e  # Exit on error

# Check if data directory exists
DATA_DIR="data/openwebtext"
if [ ! -d "$DATA_DIR" ]; then
  echo "Error: Data directory $DATA_DIR not found."
  echo "Please make sure the OpenWebText dataset is prepared correctly."
  exit 1
fi

# Check for val.bin
if [ ! -f "$DATA_DIR/val.bin" ]; then
  echo "Error: Validation data file $DATA_DIR/val.bin not found."
  exit 1
fi

# Download models if needed (uncomment if you want to pre-download)
# echo "Downloading models from HuggingFace (if not already downloaded)..."
# python -c "import torch; torch.hub.load_state_dict_from_url('https://huggingface.co/maxsegan/gpt2_l2_1e-1_100k/resolve/main/pytorch_model.bin')"
# python -c "import torch; torch.hub.load_state_dict_from_url('https://huggingface.co/maxsegan/gpt2-spatial_loss5_100k/resolve/main/pytorch_model.bin')"
# python -c "import torch; torch.hub.load_state_dict_from_url('https://huggingface.co/maxsegan/gpt2-spatial_loss1_100k/resolve/main/pytorch_model.bin')"

# Run evaluation
echo "Starting GPT2 sparsity evaluation..."
python simplified_eval_script.py

echo "Evaluation complete. Results saved in sparsity_results.csv."
