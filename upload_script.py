import os
import json
import torch
import argparse
from huggingface_hub import HfApi
from model import GPTConfig, GPT
import shutil

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Upload trained model to HuggingFace Hub')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint file')
parser.add_argument('--repo_id', type=str, required=True, help='HuggingFace repository ID (e.g., username/model-name)')
args = parser.parse_args()

api = HfApi()

# Load the checkpoint
checkpoint = torch.load(args.model_path, map_location="cpu")
state_dict = checkpoint["model"]
model_args = checkpoint["model_args"]

# Clean up state dict if needed
unwanted_prefix = '_orig_mod.'
for k in list(state_dict.keys()):
    if k.startswith(unwanted_prefix):
        new_key = k[len(unwanted_prefix):]
        state_dict[new_key] = state_dict.pop(k)

# Create model config
config = GPTConfig(
    block_size=model_args['block_size'],
    vocab_size=model_args['vocab_size'],
    n_layer=model_args['n_layer'],
    n_head=model_args['n_head'],
    n_embd=model_args['n_embd'],
    dropout=model_args.get('dropout', 0.0),
    bias=model_args.get('bias', True)
)

# Load state dict into model
model = GPT(config)
model.load_state_dict(state_dict)
model.eval()

# Create temporary directory for model files
save_dir = "temp_model_dir"
os.makedirs(save_dir, exist_ok=True)

# Save model state dict and config
torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
with open(os.path.join(save_dir, "config.json"), "w") as f:
    json.dump(config.__dict__, f)

# Create README.md with basic info
with open(os.path.join(save_dir, "README.md"), "w") as f:
    f.write(f"# {args.repo_id.split('/')[-1]}\n\n")
    f.write("## Model Details\n\n")
    f.write(f"- Block size: {model_args['block_size']}\n")
    f.write(f"- Vocabulary size: {model_args['vocab_size']}\n")
    f.write(f"- Layers: {model_args['n_layer']}\n")
    f.write(f"- Heads: {model_args['n_head']}\n")
    f.write(f"- Embedding size: {model_args['n_embd']}\n")

# Create or update the repository
api.create_repo(repo_id=args.repo_id, exist_ok=True)

# Upload the model files
api.upload_folder(
    folder_path=save_dir,
    path_in_repo=".",
    repo_id=args.repo_id,
    repo_type="model",
)

# Clean up
shutil.rmtree(save_dir)

print(f"Model successfully uploaded to {args.repo_id}")