"""
Completely fixed version that ensures each model is loaded and evaluated separately.
"""

import os
import math
import numpy as np
import torch
from tqdm import tqdm
import random
import hashlib

from model import GPT, GPTConfig

# Configuration
data_dir = 'data/openwebtext'
eval_iters = 200
batch_size = 12
block_size = 1024
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
ptdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
ctx = torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=ptdtype)

# Models to evaluate (ensure they're clearly different files/models)
models = [
    "maxsegan/gpt2_l2_1e-1_100k",
    "maxsegan/gpt2-spatial_loss5_100k",
    "maxsegan/gpt2-spatial_loss1_100k"
]

# Short names for display
model_display_names = {
    "maxsegan/gpt2_l2_1e-1_100k": "L2 1e-1",
    "maxsegan/gpt2-spatial_loss5_100k": "Spatial 5",
    "maxsegan/gpt2-spatial_loss1_100k": "Spatial 1"
}

# Sparsity levels to evaluate
sparsity_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

# Create consistent validation batches that we can reuse for all evaluations
def generate_validation_batches():
    """Generate fixed validation batches that will be saved and reused."""
    print("Generating fixed validation batches...")
    
    # Set a seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Load the validation data
    data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # Generate batch indices
    batches = []
    for k in range(eval_iters):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        batches.append((x.to(device), y.to(device)))
    
    return batches

def apply_magnitude_pruning(model, sparsity):
    """Apply weight pruning by zeroing out the lowest magnitude weights."""
    if sparsity == 0.0:
        print(f"Target sparsity: {sparsity:.2f}, Actual sparsity: 0.0000")
        return 0.0  # Return 0.0 for no pruning
    
    # Collect all weights that should be pruned
    all_weights = []
    for name, param in model.named_parameters():
        # Focus on weight matrices, not biases, layernorms, or embeddings
        if ('weight' in name and param.dim() > 1 and 
            not any(x in name for x in ['ln', 'layernorm', 'embed', 'wpe'])):
            all_weights.append(param.abs().view(-1))
    
    # Find the magnitude threshold
    all_weights = torch.cat(all_weights)
    threshold_idx = int(sparsity * len(all_weights))
    if threshold_idx >= len(all_weights):
        threshold_idx = len(all_weights) - 1
    threshold = all_weights.sort()[0][threshold_idx]
    
    # Apply pruning to each parameter
    total_weights = 0
    total_pruned = 0
    for name, param in model.named_parameters():
        if ('weight' in name and param.dim() > 1 and 
            not any(x in name for x in ['ln', 'layernorm', 'embed', 'wpe'])):
            mask = param.abs() <= threshold
            param.data[mask] = 0.0
            
            total_weights += param.numel()
            total_pruned += mask.sum().item()
    
    actual_sparsity = total_pruned / total_weights
    print(f"Target sparsity: {sparsity:.2f}, Actual sparsity: {actual_sparsity:.4f}")
    return actual_sparsity

@torch.no_grad()
def estimate_loss(model, fixed_batches):
    """Estimate the loss on the fixed validation batches."""
    model.eval()
    losses = torch.zeros(eval_iters)
    
    for k in range(eval_iters):
        X, Y = fixed_batches[k]
        with ctx:
            _, loss = model(X, Y)
        losses[k] = loss.item()
    
    return losses.mean().item()

def calculate_model_hash(model):
    """Calculate a hash of model weights to verify they're different."""
    hasher = hashlib.md5()
    
    for name, param in sorted(model.named_parameters()):
        # Only hash a subset of parameters to avoid excessive computation
        if 'weight' in name and not name.endswith('.wpe.weight'):
            # Convert to CPU first to ensure consistent hash
            data = param.data.detach().cpu().numpy().tobytes()
            hasher.update(data[:1000])  # Only hash the first 1000 bytes
    
    return hasher.hexdigest()

def load_model(model_path):
    """Load a model from HuggingFace."""
    print(f"Loading model from {model_path}")
    
    # Create a standard GPT2 configuration
    config = GPTConfig(
        block_size=block_size,
        vocab_size=50304,  # Padded to nearest multiple of 64 for efficiency
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,  # No dropout for evaluation
        bias=False     # Models were trained with bias=False
    )
    
    # Initialize the model
    model = GPT(config)
    
    # Load the weights
    try:
        # Try to load directly from Hugging Face
        checkpoint = torch.hub.load_state_dict_from_url(
            f"https://huggingface.co/{model_path}/resolve/main/pytorch_model.bin",
            map_location=device,
            file_name=f"{model_path.split('/')[-1]}.bin"  # Force different cache files
        )
    except:
        # Fallback to local file if it exists
        local_path = f"{model_path.split('/')[-1]}_pytorch_model.bin"
        if os.path.exists(local_path):
            checkpoint = torch.load(local_path, map_location=device)
        else:
            raise FileNotFoundError(f"Could not load model from {model_path} or local file")
    
    # Clean state dict if needed
    unwanted_prefix = '_orig_mod.'
    for k, v in list(checkpoint.items()):
        if k.startswith(unwanted_prefix):
            checkpoint[k[len(unwanted_prefix):]] = checkpoint.pop(k)
    
    # Load state dict with strict=False to skip missing keys
    model.load_state_dict(checkpoint, strict=False)
    
    # Verify model loaded correctly
    print(f"Model loaded. Parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    model.to(device)
    model.eval()
    
    return model

def main():
    """Run the evaluation."""
    results = {}
    model_hashes = {}
    
    # Check if validation data exists
    if not os.path.exists(os.path.join(data_dir, 'val.bin')):
        raise FileNotFoundError(f"Validation data not found at {os.path.join(data_dir, 'val.bin')}")
    
    # Generate fixed validation batches once
    fixed_batches = generate_validation_batches()
    
    # Evaluate each model on these fixed batches
    for model_path in models:
        model_key = model_path.split('/')[-1]
        print(f"\n{'='*50}")
        print(f"Evaluating {model_key}")
        print(f"{'='*50}")
        
        # First, load the model without any pruning to calculate its hash
        initial_model = load_model(model_path)
        model_hash = calculate_model_hash(initial_model)
        model_hashes[model_key] = model_hash
        print(f"Model hash: {model_hash[:10]}...")
        
        # Verify this model is different from others we've loaded
        for k, h in model_hashes.items():
            if k != model_key and h == model_hash:
                print(f"WARNING: Model {model_key} has the same hash as {k}!")
        
        # Free memory
        del initial_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        losses = []
        for sparsity in tqdm(sparsity_levels, desc="Evaluating sparsity levels"):
            # Load a fresh model for each sparsity level
            model = load_model(model_path)
            
            # Verify loaded model matches the initial hash
            if calculate_model_hash(model) != model_hash:
                print("WARNING: Loaded model doesn't match initial hash!")
                
            # Apply weight pruning
            actual_sparsity = apply_magnitude_pruning(model, sparsity)
            
            # Measure loss with consistent batches across models
            loss = estimate_loss(model, fixed_batches)
            losses.append(loss)
            print(f"Sparsity {sparsity*100:.1f}% (actual {actual_sparsity*100:.1f}%): Loss = {loss:.6f}")
            
            # Free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        results[model_key] = losses
    
    # Print results in a table format
    print("\nEvaluation Results:")
    model_names = [model_display_names.get(m, m.split('/')[-1]) for m in models]
    print(f"{'Sparsity':<10} | " + " | ".join(f"{name:<12}" for name in model_names))
    print("-" * (10 + 3 + sum(15 for _ in models)))
    
    for i, sparsity in enumerate(sparsity_levels):
        values = [f"{results[model_path.split('/')[-1]][i]:.6f}" for model_path in models]
        print(f"{sparsity*100:<10.1f}% | " + " | ".join(f"{val:<12}" for val in values))
    
    # Save results to CSV
    with open('sparsity_results.csv', 'w') as f:
        # Header with model names
        f.write('Sparsity')
        for model_path in models:
            model_key = model_path.split('/')[-1]
            f.write(f',{model_key}')
        f.write('\n')
        
        # Data rows
        for i, sparsity in enumerate(sparsity_levels):
            f.write(f"{sparsity*100:.1f}%")
            for model_path in models:
                model_key = model_path.split('/')[-1]
                f.write(f",{results[model_key][i]}")
            f.write('\n')
    
    print("\nResults saved to sparsity_results.csv")

if __name__ == "__main__":
    main()
