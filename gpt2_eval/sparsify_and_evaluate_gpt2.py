"""
Sparsify and evaluate GPT2 models loaded from Hugging Face at different sparsity levels.
This script:
1. Loads trained GPT2 models from Hugging Face repository
2. Applies different levels of sparsity (0% to 95%)
3. Evaluates the performance (loss) at each sparsity level
4. Saves results to CSV files and creates visualization plots
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import sys
import random

# Hack city import to find the model module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig

random.seed(1776)

# Configure argument parser
parser = argparse.ArgumentParser(description="Sparsify and evaluate GPT2 models from Hugging Face")
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for evaluation')
parser.add_argument('--block_size', type=int, default=1024,
                    help='Block size for evaluation')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='Device to run evaluation on')
parser.add_argument('--results_dir', type=str, default='gpt2_sparsity_results',
                    help='Directory to save results')
parser.add_argument('--force_reevaluate', action='store_true',
                    help='Force re-evaluation of all models, even if results exist')
parser.add_argument('--eval_iters', type=int, default=1000,
                    help='Number of iterations for evaluation')
parser.add_argument('--data_dir', type=str, default='../data/openwebtext',
                    help='Directory with dataset')
args = parser.parse_args()

args.data_dir = os.path.normpath(args.data_dir)

# Create results directory
os.makedirs(args.results_dir, exist_ok=True)

# Define checkpoints to evaluate
CHECKPOINTS = [
    {
        'repo_id': 'maxsegan/gpt2_l1_8_100k',
        'filename': 'pytorch_model.bin',
        'name': 'maxsegan_gpt2_l1_8_100k',
        'group': 'L1'
    },
    {
        'repo_id': 'maxsegan/gpt2_l1_16_100k',
        'filename': 'pytorch_model.bin',
        'name': 'maxsegan_gpt2_l1_16_100k',
        'group': 'L1'
    },
    {
        'repo_id': 'maxsegan/gpt2_l1_32_100k',
        'filename': 'pytorch_model.bin',
        'name': 'maxsegan_gpt2_l1_32_100k',
        'group': 'L1'
    },
    {
        'repo_id': 'maxsegan/gpt2_l1_64_100k',
        'filename': 'pytorch_model.bin',
        'name': 'maxsegan_gpt2_l1_64_100k',
        'group': 'L1'
    },
    {
        'repo_id': 'maxsegan/gpt2_full_spatial_128_100k',
        'filename': 'pytorch_model.bin',
        'name': 'gpt2_full_spatial_128_100k',
        'group': 'Spatial'
    },
    {
        'repo_id': 'maxsegan/gpt2_full_spatial_64_100k',
        'filename': 'pytorch_model.bin',
        'name': 'gpt2_full_spatial_64_100k',
        'group': 'Spatial'
    },
    {
        'repo_id': 'maxsegan/gpt2_d_spatial_64_0.1_100k',
        'filename': 'pytorch_model.bin',
        'name': 'gpt2_d_spatial_64_0.1_100k',
        'group': 'Spatial'
    },
    {
        'repo_id': 'maxsegan/gpt2_full_spatial_16_100k',
        'filename': 'pytorch_model.bin',
        'name': 'gpt2_full_spatial_16_100k',
        'group': 'Spatial'
    },
    {
        'repo_id': 'maxsegan/gpt2_combo_l1_16_spatial_64_100k',
        'filename': 'pytorch_model.bin',
        'name': 'gpt2_l1_16_spatial_64_100k',
        'group': 'Combo'
    },
    {
        'repo_id': 'maxsegan/gpt2_l1_64_spatial_16_100k',
        'filename': 'pytorch_model.bin',
        'name': 'gpt2_l1_64_spatial_16_100k',
        'group': 'Combo'
    },
    {
        'repo_id': 'maxsegan/gpt2_l1_16_spatial_32_100k',
        'filename': 'pytorch_model.bin',
        'name': 'gpt2_l1_16_spatial_32_100k',
        'group': 'Combo'
    },
    {
        'repo_id': 'maxsegan/gpt2_l1_16_spatial_16_100k',
        'filename': 'pytorch_model.bin',
        'name': 'gpt2_l1_16_spatial_16_100k',
        'group': 'Combo'
    },
    {
        'repo_id': 'maxsegan/gpt2_l1_32_spatial_16_100k',
        'filename': 'pytorch_model.bin',
        'name': 'gpt2_l1_32_spatial_16_100k',
        'group': 'Combo'
    },
    {
        'repo_id': 'maxsegan/gpt2_l1_32_spatial_64_100k',
        'filename': 'pytorch_model.bin',
        'name': 'gpt2_l1_32_spatial_16_100k',
        'group': 'Combo'
    },
    {
        'repo_id': 'maxsegan/gpt2_l1_32_spatial_32_100k',
        'filename': 'pytorch_model.bin',
        'name': 'gpt2_l1_32_spatial_32_100k',
        'group': 'Combo'
    }
]

# Define sparsity levels
sparsity_levels = [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

# Apply magnitude pruning to a model
def apply_magnitude_pruning(model, sparsity):
    """
    Apply weight pruning by zeroing out the lowest magnitude weights.
    Returns the actual sparsity achieved.
    """
    if sparsity <= 0.0:
        return 0.0  # No pruning
    
    # Collect all weights that should be pruned
    all_weights = []
    weight_tensors = []
    
    for name, param in model.named_parameters():
        # Focus on weight matrices, not biases, layernorms, or embeddings
        if ('weight' in name and param.dim() > 1 and 
            not any(x in name for x in ['ln', 'layernorm', 'embed', 'wpe', 'wte'])):
            all_weights.append(param.abs().view(-1))
            weight_tensors.append(param)
    
    # Find the magnitude threshold
    all_weights_flat = torch.cat(all_weights)
    threshold_idx = int(sparsity * len(all_weights_flat))
    if threshold_idx >= len(all_weights_flat):
        threshold_idx = len(all_weights_flat) - 1
    threshold = all_weights_flat.sort()[0][threshold_idx]
    
    # Apply pruning to each parameter
    total_weights = 0
    total_pruned = 0
    for param in weight_tensors:
        mask = param.abs() <= threshold
        param.data[mask] = 0.0
        
        total_weights += param.numel()
        total_pruned += mask.sum().item()
    
    actual_sparsity = total_pruned / total_weights if total_weights > 0 else 0.0
    return actual_sparsity

# Set up data loading
def get_batch(split, data_dir=args.data_dir, block_size=args.block_size, batch_size=args.batch_size):
    """Get a random batch from the dataset."""
    try:
        data_file = os.path.join(data_dir, f'{split}.bin')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} not found")
        
        data = np.memmap(data_file, dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(args.device), y.to(args.device)
        return x, y
    except Exception as e:
        print(f"Error preparing real data: {str(e)}")

        return None, None

def create_fixed_batches(n_batches=args.eval_iters):
    """Create fixed batches for consistent evaluation."""
    fixed_batches = []
    for i in range(n_batches):
        try:
            x, y = get_batch('val')
            fixed_batches.append((x, y))
        except Exception as e:
            print(f"Error creating batch {i}: {str(e)}")
    return fixed_batches

# Function to compute loss using the model
@torch.no_grad()
def estimate_loss(model, fixed_batches):
    """Estimate model loss on validation data using fixed batches."""
    model.eval()
    batch_count = len(fixed_batches)
    losses = torch.zeros(batch_count, device=args.device)
    
    # Process each batch
    for i, (x, y) in enumerate(fixed_batches):
        print("batch #", i)
        # Forward pass with your custom GPT model
        logits, loss = model(x, y)
        losses[i] = loss
    
    return losses.mean().item()

# Find checkpoint files and download from Hugging Face
def find_checkpoints():
    """Find and download checkpoint files from the Hugging Face repositories."""
    checkpoints = []
    
    try:
        print(f"Loading {len(CHECKPOINTS)} checkpoints for evaluation...")
        
        for ckpt in CHECKPOINTS:
            try:
                local_file = hf_hub_download(repo_id=ckpt['repo_id'], filename=ckpt['filename'])
                checkpoints.append({
                    'name': ckpt['name'],
                    'checkpoint_path': local_file,
                    'group': ckpt.get('group', 'Other')
                })
            except Exception as e:
                print(f"Error downloading checkpoint from {ckpt['repo_id']}: {str(e)}")
                continue
        
        print(f"Found total of {len(checkpoints)} checkpoints for evaluation")
        for ckpt in checkpoints:
            print(f"  - {ckpt['name']} (Group: {ckpt['group']})")
        
        return checkpoints
    
    except Exception as e:
        print(f"Error accessing Hugging Face repositories: {str(e)}")
        return []

# Check for existing evaluation results
def check_existing_results(model_name, results_dir=args.results_dir):
    """Check if evaluation results already exist for this model and identify missing sparsity levels."""
    csv_path = os.path.join(results_dir, f"{model_name}_sparsity.csv")
    if os.path.exists(csv_path):
        try:
            # Load the CSV and check which sparsity levels need evaluation
            df = pd.read_csv(csv_path)
            existing_sparsity = df['target_sparsity'].values
            
            # Find missing sparsity levels with tolerance for floating point issues
            tolerance = 0.001  # Tolerance for floating point comparison
            missing_sparsity = []
            
            for target in sparsity_levels:
                # Check if any existing sparsity level is close to this target
                if not any(abs(existing - target) <= tolerance for existing in existing_sparsity):
                    missing_sparsity.append(target)
            
            if not missing_sparsity:
                print(f"Found complete results for {model_name}")
                return df, []
            else:
                # Format missing levels for cleaner output
                missing_formatted = [f"{s:.2f}" for s in missing_sparsity]
                print(f"Found partial results for {model_name}, missing sparsity levels: {', '.join(missing_formatted)}")
                return df, missing_sparsity
        except Exception as e:
            print(f"Error reading existing results for {model_name}: {str(e)}")
            return None, sparsity_levels
    return None, sparsity_levels

# Load combined results file
def load_combined_results(results_dir=args.results_dir):
    """Load the combined results file if it exists."""
    combined_csv = os.path.join(results_dir, "all_gpt2_models_sparsity.csv")
    if os.path.exists(combined_csv):
        try:
            return pd.read_csv(combined_csv)
        except Exception as e:
            print(f"Error reading combined results: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()

# Update combined results with new model results
def update_combined_results(new_results, existing_results=None, results_dir=args.results_dir):
    """Update the combined results with new model results."""
    if existing_results is None or existing_results.empty:
        # Just write the new results
        new_results.to_csv(os.path.join(results_dir, "all_gpt2_models_sparsity.csv"), index=False)
        return new_results
    
    # Check if the model already exists in the combined results
    model_name = new_results['model_name'].iloc[0]
    combined_df = existing_results.copy()
    
    # Remove existing entries for this model if they exist
    if model_name in existing_results['model_name'].values:
        combined_df = combined_df[combined_df['model_name'] != model_name]
    
    # Append the new results
    combined_df = pd.concat([combined_df, new_results], ignore_index=True)
    
    # Write the updated combined results
    combined_df.to_csv(os.path.join(results_dir, "all_gpt2_models_sparsity.csv"), index=False)
    return combined_df

def load_model_from_checkpoint(checkpoint_path, device=args.device):
    """Load a model from a checkpoint file with support for your custom GPT implementation."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None, None
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Determine if this is a HuggingFace standard PyTorch model
        is_huggingface_model = os.path.basename(checkpoint_path) == 'pytorch_model.bin'
        
        # Extract model configuration if available
        if isinstance(checkpoint, dict) and 'model_args' in checkpoint:
            # Handle custom checkpoint format
            model_args = checkpoint['model_args']
            
            # Create the model using your GPT implementation
            config = GPTConfig(**model_args)
            model = GPT(config)
            
            # Load the state dict
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                # If 'model' key doesn't exist, try to load directly
                print("No 'model' key found in checkpoint, attempting to load state_dict directly")
                model.load_state_dict(checkpoint)
                
            model.to(device)
            model.eval()
            
            # Extract regularization parameters if available
            print(f"Best validation loss for this model: {checkpoint.get('best_val_loss', 0.0):.4f}")
            regularization = checkpoint.get('regularization', {})
            l1_scale = regularization.get('l1_scale', 0.0)
            weight_decay = regularization.get('weight_decay', 0.0)
            spatial_cost_scale = regularization.get('spatial_cost_scale', 0.0)
            
            return model, {'l1_scale': l1_scale, 'weight_decay': weight_decay, 'spatial_cost_scale': spatial_cost_scale}
        
        # If model_args isn't available, try to infer configuration from the checkpoint structure
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            # Extract model hyperparameters from state dict if possible
            state_dict = checkpoint['model']
            
            # Try to determine model configuration from state dict
            # Example: look at embedding dimension from wte weight
            if 'transformer.wte.weight' in state_dict:
                vocab_size, n_embd = state_dict['transformer.wte.weight'].shape
                
                # Look at number of layers by finding the highest layer index
                n_layer = 0
                for key in state_dict:
                    if 'transformer.h.' in key:
                        layer_idx = int(key.split('.')[2])
                        n_layer = max(n_layer, layer_idx + 1)
                
                # Estimate n_head from layer 0 attention matrix
                if 'transformer.h.0.attn.c_attn.weight' in state_dict:
                    _, attn_dim = state_dict['transformer.h.0.attn.c_attn.weight'].shape
                    n_head = n_embd // 64  # Assuming head size of 64
                else:
                    n_head = 12  # Default for GPT2
                
                print(f"Inferring model config: n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}, vocab_size={vocab_size}")
                
                # Create model with inferred config
                config = GPTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd, 
                                  vocab_size=vocab_size, block_size=args.block_size)
                model = GPT(config)
                
                # Attempt to load the state dict
                try:
                    model.load_state_dict(state_dict)
                except Exception as e:
                    print(f"Error loading state dict: {str(e)}")
                    print("Attempting to load with strict=False")
                    model.load_state_dict(state_dict, strict=False)
                
                model.to(device)
                model.eval()
                
                # Extract regularization parameters if available
                regularization = checkpoint.get('regularization', {})
                l1_scale = regularization.get('l1_scale', 0.0)
                weight_decay = regularization.get('weight_decay', 0.0)
                spatial_cost_scale = regularization.get('spatial_cost_scale', 0.0)
                
                return model, {'l1_scale': l1_scale, 'weight_decay': weight_decay, 'spatial_cost_scale': spatial_cost_scale}
        
        # Handle HuggingFace standard PyTorch model format - direct state dict
        elif is_huggingface_model:
            print("Loading HuggingFace standard PyTorch model")
            
            # For HuggingFace models, try to load config.json from the same repo
            repo_id = None
            for ckpt in CHECKPOINTS:
                if os.path.basename(checkpoint_path) == os.path.basename(ckpt['filename']) and 'maxsegan' in ckpt['repo_id']:
                    repo_id = ckpt['repo_id']
                    break
            
            model_config = None
            if repo_id:
                try:
                    # Try to download and load config.json
                    config_file = hf_hub_download(repo_id=repo_id, filename='config.json')
                    with open(config_file, 'r') as f:
                        import json
                        model_config = json.load(f)
                    print(f"Loaded model config from {repo_id}/config.json")
                except Exception as e:
                    print(f"Could not load config.json: {str(e)}")
            
            # If we have a config file, use it to create the model
            if model_config:
                # Use GPT-2 default configuration, with appropriate adjustments
                n_layer = model_config.get('n_layer', 12)
                n_head = model_config.get('n_head', 12)
                n_embd = model_config.get('n_embd', 768)
                vocab_size = model_config.get('vocab_size', 50257)
                
                print(f"Using config from config.json: n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}, vocab_size={vocab_size}")
            else:
                # Try to infer from state dict
                if isinstance(checkpoint, dict):
                    # Try to determine config from state dict keys and shapes
                    wte_key = next((k for k in checkpoint.keys() if 'wte.weight' in k), None)
                    if wte_key:
                        vocab_size, n_embd = checkpoint[wte_key].shape
                    else:
                        # Default GPT-2 small config
                        vocab_size, n_embd = 50257, 768
                    
                    # Count layers by scanning for layer indices
                    n_layer = 0
                    for key in checkpoint.keys():
                        if '.h.' in key or '.transformer.h.' in key:
                            parts = key.split('.')
                            for i, part in enumerate(parts):
                                if part == 'h' and i + 1 < len(parts) and parts[i+1].isdigit():
                                    layer_idx = int(parts[i+1])
                                    n_layer = max(n_layer, layer_idx + 1)
                    
                    if n_layer == 0:  # If no layers found, use default
                        n_layer = 12
                    
                    # Default GPT-2 head config
                    n_head = 12
                    
                    print(f"Inferring model config from state dict: n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}, vocab_size={vocab_size}")
                else:
                    # Complete fallback to default GPT-2 small config
                    n_layer, n_head, n_embd, vocab_size = 12, 12, 768, 50257
                    print(f"Using default GPT-2 small config: n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}, vocab_size={vocab_size}")
            
            # Create model with inferred or default config
            config = GPTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd, 
                              vocab_size=vocab_size, block_size=args.block_size)
            model = GPT(config)
            
            # Try to adapt HuggingFace state dict to our model format
            try:
                # If checkpoint is a state dict with HF-style keys, try mapping the keys
                hf_state_dict = checkpoint
                our_state_dict = {}
                
                # Map HuggingFace keys to our model keys
                key_mapping = {
                    # This is a rough mapping, may need adjustment
                    'transformer.wte.weight': 'transformer.wte.weight',
                    'transformer.wpe.weight': 'transformer.wpe.weight',
                    'transformer.ln_f.weight': 'transformer.ln_f.weight',
                    'transformer.ln_f.bias': 'transformer.ln_f.bias',
                    'lm_head.weight': 'lm_head.weight',
                }
                
                # Layer-specific mappings
                for i in range(n_layer):
                    layer_map = {
                        f'transformer.h.{i}.ln_1.weight': f'transformer.h.{i}.ln_1.weight',
                        f'transformer.h.{i}.ln_1.bias': f'transformer.h.{i}.ln_1.bias',
                        f'transformer.h.{i}.ln_2.weight': f'transformer.h.{i}.ln_2.weight',
                        f'transformer.h.{i}.ln_2.bias': f'transformer.h.{i}.ln_2.bias',
                        f'transformer.h.{i}.attn.c_attn.weight': f'transformer.h.{i}.attn.c_attn.weight',
                        f'transformer.h.{i}.attn.c_attn.bias': f'transformer.h.{i}.attn.c_attn.bias',
                        f'transformer.h.{i}.attn.c_proj.weight': f'transformer.h.{i}.attn.c_proj.weight',
                        f'transformer.h.{i}.attn.c_proj.bias': f'transformer.h.{i}.attn.c_proj.bias',
                        f'transformer.h.{i}.mlp.c_fc.weight': f'transformer.h.{i}.mlp.c_fc.weight',
                        f'transformer.h.{i}.mlp.c_fc.bias': f'transformer.h.{i}.mlp.c_fc.bias',
                        f'transformer.h.{i}.mlp.c_proj.weight': f'transformer.h.{i}.mlp.c_proj.weight',
                        f'transformer.h.{i}.mlp.c_proj.bias': f'transformer.h.{i}.mlp.c_proj.bias',
                    }
                    key_mapping.update(layer_map)
                
                # Create a state dict for our model format
                for hf_key, our_key in key_mapping.items():
                    if hf_key in hf_state_dict:
                        our_state_dict[our_key] = hf_state_dict[hf_key]
                    elif our_key in hf_state_dict:  # Already in our format
                        our_state_dict[our_key] = hf_state_dict[our_key]
                
            # For HF models, other keys might need different prefix mapping
                for key in hf_state_dict:
                    if key not in our_state_dict.keys() and key not in [v for k, v in key_mapping.items()]:
                        # Try to map additional keys
                        if 'transformer.' in key:
                            our_state_dict[key] = hf_state_dict[key]
                
                # Initialize missing bias terms with zeros if needed
                model_state_dict = model.state_dict()
                for key in model_state_dict:
                    if key not in our_state_dict and '.bias' in key:
                        print(f"Initializing missing bias term: {key}")
                        our_state_dict[key] = torch.zeros_like(model_state_dict[key])
                
                # Try to load the remapped state dict
                try:
                    model.load_state_dict(our_state_dict)
                except Exception as e:
                    print(f"Error loading remapped state dict: {str(e)}")
                    print("Attempting to load with strict=False")
                    model.load_state_dict(our_state_dict, strict=False)
                
                model.to(device)
                model.eval()
                
                # For HF models, we don't have regularization info, use defaults
                l1_scale = 0.0
                spatial_cost_scale = 0.0
                
                # Extract regularization info from the repo name if possible
                for ckpt in CHECKPOINTS:
                    if ckpt['filename'] == os.path.basename(checkpoint_path):
                        repo_parts = ckpt['repo_id'].split('/')[-1].split('_')
                        
                        # Check for l1 scale
                        for i, part in enumerate(repo_parts):
                            if part == 'l1' and i + 1 < len(repo_parts) and repo_parts[i+1].isdigit():
                                l1_scale = int(repo_parts[i+1]) / 100.0
                                print(f"Extracted l1_scale={l1_scale} from repo name")
                        
                        # Check for spatial cost scale
                        for i, part in enumerate(repo_parts):
                            if part == 'spatial' and i + 1 < len(repo_parts) and repo_parts[i+1].isdigit():
                                spatial_cost_scale = int(repo_parts[i+1]) / 100.0
                                print(f"Extracted spatial_cost_scale={spatial_cost_scale} from repo name")
                        
                        break
                
                return model, {'l1_scale': l1_scale, 'weight_decay': 0.0, 'spatial_cost_scale': spatial_cost_scale}
                
            except Exception as e:
                print(f"Error adapting HuggingFace model: {str(e)}")
                import traceback
                traceback.print_exc()
                return None, None
            
        else:
            print("Unsupported checkpoint format")
            return None, None
                
    except Exception as e:
        print(f"Error loading model from {checkpoint_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None
    
def inspect_checkpoint(checkpoint_path):
    """Analyze the structure of a checkpoint file to help debug loading issues."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"\nCheckpoint inspection for {os.path.basename(checkpoint_path)}:")
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint is a dictionary with {len(checkpoint)} keys")
            print(f"Top-level keys: {list(checkpoint.keys())}")
            
            # Check for model weights
            if 'model' in checkpoint:
                model_dict = checkpoint['model']
                print(f"Found 'model' key with {len(model_dict)} parameters")
                
                # Sample some keys and shapes
                sample_keys = list(model_dict.keys())[:5]
                print("Sample keys and shapes:")
                for k in sample_keys:
                    if isinstance(model_dict[k], torch.Tensor):
                        print(f"  {k}: {model_dict[k].shape}")
            
            # Check for other common keys
            for key in ['state_dict', 'config', 'model_args', 'regularization']:
                if key in checkpoint:
                    if key in ['config', 'model_args', 'regularization']:
                        print(f"Found '{key}': {checkpoint[key]}")
                    else:
                        print(f"Found '{key}' with {len(checkpoint[key])} items")
        else:
            print("Checkpoint is not a dictionary, may be a direct state dict")
            
            # If it's a state dict
            if hasattr(checkpoint, 'keys'):
                print(f"Checkpoint has {len(checkpoint)} keys")
                sample_keys = list(checkpoint.keys())[:5]
                print("Sample keys:")
                for k in sample_keys:
                    if isinstance(checkpoint[k], torch.Tensor):
                        print(f"  {k}: {checkpoint[k].shape}")
        
        return True
    except Exception as e:
        print(f"Error inspecting checkpoint: {str(e)}")
        return False

def main():
    # Prepare evaluation data - create fixed batches for consistent evaluation
    print("Creating fixed validation batches...")
    try:
        fixed_batches = create_fixed_batches()
    except Exception as e:
        print(f"Error creating fixed batches: {str(e)}")
        return
    
    # Load existing combined results
    combined_results = load_combined_results()
    if not combined_results.empty:
        print(f"Loaded existing results for {combined_results['model_name'].nunique()} models")
    
    # Find all checkpoints from the predefined list
    print("Finding checkpoints...")
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("No checkpoints found. Exiting.")
        return
    
    # Filtered checkpoints to only process new ones or missing sparsity levels
    models_to_process = []
    
    for ckpt_info in checkpoints:
        model_name = ckpt_info['name']
        
        if args.force_reevaluate:
            ckpt_info['missing_sparsity'] = sparsity_levels
            models_to_process.append(ckpt_info)
            continue
        
        # Check if we already have results for this model and what sparsity levels are missing
        existing_results, missing_sparsity = check_existing_results(model_name)
        
        if existing_results is not None:
            # Update combined results with existing data
            combined_results = update_combined_results(existing_results, combined_results)
            
            # If there are missing sparsity levels to evaluate, add to processing list
            if missing_sparsity:
                ckpt_info['existing_results'] = existing_results
                ckpt_info['missing_sparsity'] = missing_sparsity
                models_to_process.append(ckpt_info)
        else:
            # No existing results, process all sparsity levels
            ckpt_info['missing_sparsity'] = sparsity_levels
            models_to_process.append(ckpt_info)

    print(f"Will process {len(models_to_process)} models")
    
    # Process each model
    for ckpt_info in tqdm(models_to_process, desc="Processing models"):
        checkpoint_path = ckpt_info['checkpoint_path']
        model_name = ckpt_info['name']
        missing_sparsity = ckpt_info['missing_sparsity']

        print(f"\nInspecting and evaluating {model_name}...")
    
        # First inspect the checkpoint
        inspect_checkpoint(checkpoint_path)
        
        print(f"\nEvaluating {model_name} at {len(missing_sparsity)} missing sparsity levels...")
        
        # Get existing results or create empty results list
        if 'existing_results' in ckpt_info:
            model_results = ckpt_info['existing_results'].to_dict('records')
        else:
            model_results = []
        
        # Evaluate at each missing sparsity level
        for target_sparsity in tqdm(missing_sparsity, desc=f"Sparsity levels for {model_name}"):
            try:
                # Load a fresh model for this sparsity level
                model, regularization = load_model_from_checkpoint(checkpoint_path)
                
                if model is None:
                    print(f"Skipping sparsity {target_sparsity} for {model_name} due to loading error")
                    continue
                
                # Get regularization parameters
                l1_scale = regularization.get('l1_scale', 0.0)
                weight_decay = regularization.get('weight_decay', 0.0)
                spatial_cost_scale = regularization.get('spatial_cost_scale', 0.0)
                
                # Apply pruning
                actual_sparsity = apply_magnitude_pruning(model, target_sparsity)
                
                # Evaluate the model using our fixed batches
                loss = estimate_loss(model, fixed_batches)
                
                # Store results
                result = {
                    'model_name': model_name,
                    'target_sparsity': target_sparsity,
                    'actual_sparsity': actual_sparsity,
                    'val_loss': loss,
                    'l1_scale': l1_scale,
                    'weight_decay': weight_decay,
                    'spatial_cost_scale': spatial_cost_scale,
                    'group': ckpt_info['group']  # Use the explicit group from CHECKPOINTS
                }
                model_results.append(result)
                
                # Output progress
                print(f"Sparsity {target_sparsity*100:.1f}% → {actual_sparsity*100:.1f}%: Loss = {loss:.6f}")
                
                # Free up memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error evaluating {model_name} at sparsity {target_sparsity}: {str(e)}")
                continue
    
        # Save individual model results
        if model_results:
            model_df = pd.DataFrame(model_results)
            model_csv_path = os.path.join(args.results_dir, f"{model_name}_sparsity.csv")
            model_df.to_csv(model_csv_path, index=False)
            print(f"Results for {model_name} saved to {model_csv_path}")
            
            # Update combined results
            combined_results = update_combined_results(model_df, combined_results)
            
            # Plot sparsity vs. loss for this model
            try:
                plt.figure(figsize=(10, 6))
                
                # Sort by sparsity for cleaner plots
                model_df = model_df.sort_values('actual_sparsity')
                
                # Plot loss
                plt.plot(model_df['actual_sparsity'], model_df['val_loss'], 'o-', color='blue')
                plt.xlabel('Sparsity')
                plt.ylabel('Validation Loss')
                plt.yscale("log")
                plt.title(f'Sparsity vs. Loss for {model_name}')
                
                # Add grid and title
                plt.grid(True, alpha=0.3)
                
                # Save plot
                plt_path = os.path.join(args.results_dir, f"{model_name}_sparsity.png")
                plt.savefig(plt_path)
                plt.close()
            except Exception as e:
                print(f"Error creating plot for {model_name}: {str(e)}")
    
    # Create comparison plots of all models (if we have combined results)
    if not combined_results.empty:
        try:
            # Plot for loss - comparison by individual model
            plt.figure(figsize=(12, 8))
            
            # Plot each model as a separate line
            for model_name in combined_results['model_name'].unique():
                model_data = combined_results[combined_results['model_name'] == model_name]
                model_data = model_data.sort_values('actual_sparsity')  # Sort for cleaner lines
                plt.plot(model_data['actual_sparsity'], model_data['val_loss'], 'o-', label=model_name)
            
            plt.xlabel('Sparsity')
            plt.ylabel('Validation Loss')
            plt.title('Sparsity vs. Loss for All Models')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize='small')
            plt.yscale("log")
            plt_path = os.path.join(args.results_dir, "all_models_loss_comparison.png")
            plt.savefig(plt_path)
            plt.close()
            
            # Plot for loss - comparison by group (for Pareto front visualization)
            plt.figure(figsize=(12, 8))
            
            # Get unique groups
            if 'group' in combined_results.columns:
                groups = combined_results['group'].unique()
                
                # Color map for consistent colors per group
                import matplotlib.cm as cm
                colors = cm.tab10(np.linspace(0, 1, len(groups)))
                
                # Plot each group with distinct color and marker
                for i, group in enumerate(groups):
                    group_data = combined_results[combined_results['group'] == group]
                    
                    # For each model in the group
                    for model_name in group_data['model_name'].unique():
                        model_data = group_data[group_data['model_name'] == model_name]
                        model_data = model_data.sort_values('actual_sparsity')
                        
                        # Use same color for all models in group but different markers or line styles
                        plt.plot(model_data['actual_sparsity'], model_data['val_loss'], 'o-', 
                                 color=colors[i], label=f"{model_name} ({group})" if i == 0 else model_name)
                
                plt.xlabel('Sparsity')
                plt.ylabel('Validation Loss')
                plt.title('Sparsity vs. Loss by Model Groups')
                plt.yscale("log")
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best', fontsize='small')
                plt_path = os.path.join(args.results_dir, "model_groups_pareto_front.png")
                plt.savefig(plt_path)
                plt.close()
            
        except Exception as e:
            print(f"Error creating comparison plots: {str(e)}")
    
    print("Evaluation complete!")

if __name__ == '__main__':
    main()