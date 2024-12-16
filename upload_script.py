import os
import json
import torch
from huggingface_hub import HfApi
from model import GPTConfig, GPT
import shutil

api = HfApi()

variants = [
    ("out/gpt2-1e5ckpt.pt", "maxsegan/gpt2-spatial_loss1e-5", 1e-5),
    ("out/gpt2-1e4ckpt.pt", "maxsegan/gpt2-spatial_loss1e-4", 1e-4),
    ("out/gpt2-1e3ckpt.pt", "maxsegan/gpt2-spatial_loss1e-3", 1e-3),
    ("out/gpt2-1e2ckpt.pt", "maxsegan/gpt2-spatial_loss1e-2", 1e-2),
    ("out/gpt2-1e1ckpt.pt", "maxsegan/gpt2-spatial_loss1e-1", 1e-1),
    ("out/gpt2-1ckpt.pt",   "maxsegan/gpt2-spatial_loss1",    1.0),
    ("out/gpt2-10ckpt.pt",  "maxsegan/gpt2-spatial_loss10",   10.0),
    ("../nanoGPT/out/ckpt.pt", "maxsegan/gpt2-spatial_loss_baseline",  0),
]

for ckpt_file, repo_id, spatial_cost_scale in variants:
    checkpoint = torch.load(ckpt_file, map_location="cpu")
    state_dict = checkpoint["model"]
    model_args = checkpoint["model_args"]

    # only required for the baseline
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            new_key = k[len(unwanted_prefix):]
            state_dict[new_key] = state_dict.pop(k)

    config = GPTConfig(
        block_size=model_args['block_size'],
        vocab_size=model_args['vocab_size'],
        n_layer=model_args['n_layer'],
        n_head=model_args['n_head'],
        n_embd=model_args['n_embd'],
        dropout=model_args.get('dropout', 0.0),
        bias=model_args.get('bias', False)
    )

    model = GPT(config)
    model.load_state_dict(state_dict)
    model.eval()

    save_dir = "temp_model_dir"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f)
    #api.create_repo(repo_id=repo_id, exist_ok=True)

    api.upload_folder(
        folder_path=save_dir,
        path_in_repo=".",
        repo_id=repo_id,
        repo_type="model",
    )

    shutil.rmtree(save_dir)
