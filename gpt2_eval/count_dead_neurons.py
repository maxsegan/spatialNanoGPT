"""
Count dead neurons in sparsified GPT2 models at different sparsity levels.
This script:
1. Loads trained GPT2 models from the same checkpoints as in the original scripts
2. Applies different levels of sparsity (0% to 95%) using the same pruning method
3. Counts the number of dead neurons in MLP layers and attention heads
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

# Import the model module (assumes same path as original scripts)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig

# Configure argument parser
parser = argparse.ArgumentParser(description="Count dead neurons in sparsified GPT2 models")
parser.add_argument('--results_dir', type=str, default='gpt2_sparsity_results',
                    help='Directory with sparsification results')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory for neuron counts (defaults to results_dir)')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='Device to run evaluation on')
parser.add_argument('--force_recount', action='store_true',
                    help='Force recounting of dead neurons, even if results exist')
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.results_dir

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Define the same checkpoints as in the original script
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
        'repo_id': 'maxsegan/gpt2_d_spatial_64_0.1_100k',
        'filename': 'pytorch_model.bin',
        'name': 'gpt2_d_spatial_64_0.1_100k',
        'group': 'Spatial'
    },
    {
        'repo_id': 'maxsegan/gpt2_full_spatial_64_100k',
        'filename': 'pytorch_model.bin',
        'name': 'gpt2_full_spatial_64_100k',
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

sparsity_levels = [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

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

def count_dead_parameters(model):
    """
    Compute a single metric for dead weights in the model.
    A weight matrix is analyzed to find dead rows and columns.
    
    Returns:
        dict: A dictionary with total counts and percentages
    """
    # Track total parameters and dead parameters
    total_params = 0
    total_dead_params = 0
    total_dead_rows = 0
    total_dead_cols = 0
    total_rows = 0
    total_cols = 0
    
    # Examine each weight matrix
    for name, param in model.named_parameters():
        # Only consider matrices in attention and MLP layers
        if ('weight' in name and param.dim() > 1 and 
            not any(x in name for x in ['ln', 'layernorm', 'embed', 'wpe', 'wte'])):
            
            # Check for dead rows (all zeros in a row)
            dead_rows = (param.abs().sum(dim=1) == 0).sum().item()
            
            # Check for dead columns (all zeros in a column)
            dead_cols = (param.abs().sum(dim=0) == 0).sum().item()
            
            # Count parameters represented by dead rows/columns
            shape = param.shape
            n_rows = shape[0]
            n_cols = shape[1]
            
            # Count total parameters in this layer
            params_in_layer = param.numel()
            
            # Count number of parameters that are explicitly zero
            zero_params = (param == 0).sum().item()
            
            # Update totals
            total_params += params_in_layer
            total_dead_params += zero_params
            total_dead_rows += dead_rows
            total_dead_cols += dead_cols
            total_rows += n_rows
            total_cols += n_cols
    
    # Calculate percentages
    dead_param_percent = (total_dead_params / total_params * 100) if total_params > 0 else 0
    dead_row_percent = (total_dead_rows / total_rows * 100) if total_rows > 0 else 0
    dead_col_percent = (total_dead_cols / total_cols * 100) if total_cols > 0 else 0
    
    return {
        "total_params": total_params,
        "dead_params": total_dead_params,
        "dead_param_percent": dead_param_percent,
        "total_rows": total_rows,
        "dead_rows": total_dead_rows,
        "dead_row_percent": dead_row_percent,
        "total_cols": total_cols,
        "dead_cols": total_dead_cols,
        "dead_col_percent": dead_col_percent
    }

def load_model_from_checkpoint(checkpoint_path, device=args.device):
    """Load a model from a checkpoint file."""
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Determine if this is a HuggingFace standard PyTorch model
        is_huggingface_model = os.path.basename(checkpoint_path) == 'pytorch_model.bin'
        
        # Extract model configuration if available
        if isinstance(checkpoint, dict) and 'model_args' in checkpoint:
            # Handle custom checkpoint format
            model_args = checkpoint['model_args']
            
            # Create the model using GPT implementation
            config = GPTConfig(**model_args)
            model = GPT(config)
            
            # Load the state dict
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                # If 'model' key doesn't exist, try to load directly
                model.load_state_dict(checkpoint)
                
            model.to(device)
            model.eval()
            return model
            
        # Handle HuggingFace model
        elif is_huggingface_model:
            print("Loading HuggingFace standard PyTorch model")
            
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
                
                print(f"Inferring model config: n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}, vocab_size={vocab_size}")
                
                # Create model with inferred config
                config = GPTConfig(n_layer=n_layer, n_head=n_head, n_embd=n_embd, 
                                  vocab_size=vocab_size, block_size=1024)
                model = GPT(config)
                
                # Try to adapt HuggingFace state dict to our model format
                try:
                    # Map HuggingFace keys to our model keys
                    state_dict = {}
                    for key, value in checkpoint.items():
                        if '.attn.c_attn.' in key or '.attn.c_proj.' in key or '.mlp.c_fc.' in key or '.mlp.c_proj.' in key:
                            state_dict[key] = value
                        elif '.wte.' in key or '.wpe.' in key or '.ln_' in key:
                            state_dict[key] = value
                    
                    # Load with strict=False to skip missing keys
                    model.load_state_dict(state_dict, strict=False)
                    model.to(device)
                    model.eval()
                    return model
                    
                except Exception as e:
                    print(f"Error adapting HuggingFace model: {str(e)}")
                    return None
        
        print(f"Unsupported checkpoint format for {checkpoint_path}")
        return None
                
    except Exception as e:
        print(f"Error loading model from {checkpoint_path}: {str(e)}")
        return None

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
        
        print(f"Found total of {len(checkpoints)} checkpoints")
        return checkpoints
    
    except Exception as e:
        print(f"Error accessing Hugging Face repositories: {str(e)}")
        return []

def check_existing_results(model_name, output_dir=args.output_dir):
    """Check if dead parameter count results already exist for this model."""
    count_path = os.path.join(output_dir, f"{model_name}_dead_params.csv")
    if os.path.exists(count_path) and not args.force_recount:
        try:
            df = pd.read_csv(count_path)
            existing_sparsity = df['target_sparsity'].values
            
            # Find missing sparsity levels
            tolerance = 0.001
            missing_sparsity = []
            
            for target in sparsity_levels:
                if not any(abs(existing - target) <= tolerance for existing in existing_sparsity):
                    missing_sparsity.append(target)
            
            if not missing_sparsity:
                print(f"Found complete dead parameter results for {model_name}")
                return df, []
            else:
                print(f"Found partial results for {model_name}, missing sparsity levels: {missing_sparsity}")
                return df, missing_sparsity
        except Exception as e:
            print(f"Error reading existing results for {model_name}: {str(e)}")
            return None, sparsity_levels
    return None, sparsity_levels

def main():
    # Find all checkpoints
    print("Finding checkpoints...")
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("No checkpoints found. Exiting.")
        return
    
    # Process each model
    for ckpt_info in tqdm(checkpoints, desc="Processing models"):
        checkpoint_path = ckpt_info['checkpoint_path']
        model_name = ckpt_info['name']
        
        # Check if we already have results for this model
        existing_counts, missing_sparsity = check_existing_results(model_name)
        
        if existing_counts is not None and not missing_sparsity:
            print(f"Skipping {model_name}, dead parameter counts already exist")
            continue
        
        print(f"\nCounting dead parameters for {model_name} at {len(missing_sparsity)} sparsity levels...")
        
        # Get existing results or create empty results list
        if existing_counts is not None:
            param_counts = existing_counts.to_dict('records')
        else:
            param_counts = []
        
        # Process each sparsity level
        for target_sparsity in tqdm(missing_sparsity, desc=f"Sparsity levels for {model_name}"):
            try:
                # Load a fresh model for this sparsity level
                model = load_model_from_checkpoint(checkpoint_path)
                
                if model is None:
                    print(f"Skipping sparsity {target_sparsity} for {model_name} due to loading error")
                    continue
                
                # Apply pruning
                actual_sparsity = apply_magnitude_pruning(model, target_sparsity)
                
                # Count dead parameters
                count_results = count_dead_parameters(model)
                
                # Store results
                result = {
                    'model_name': model_name,
                    'target_sparsity': target_sparsity,
                    'actual_sparsity': actual_sparsity,
                    'dead_params': count_results['dead_params'],
                    'total_params': count_results['total_params'],
                    'dead_param_percent': count_results['dead_param_percent'],
                    'dead_rows': count_results['dead_rows'],
                    'total_rows': count_results['total_rows'],
                    'dead_row_percent': count_results['dead_row_percent'],
                    'dead_cols': count_results['dead_cols'],
                    'total_cols': count_results['total_cols'],
                    'dead_col_percent': count_results['dead_col_percent'],
                    'group': ckpt_info['group']
                }
                
                param_counts.append(result)
                
                # Output simple progress
                print(f"Sparsity {target_sparsity*100:.1f}% → {actual_sparsity*100:.1f}%: "
                      f"{count_results['dead_params']:,}/{count_results['total_params']:,} "
                      f"({count_results['dead_param_percent']:.1f}%) parameters are zero")
                print(f"Dead rows: {count_results['dead_rows']:,}/{count_results['total_rows']:,} "
                      f"({count_results['dead_row_percent']:.1f}%)")
                print(f"Dead columns: {count_results['dead_cols']:,}/{count_results['total_cols']:,} "
                      f"({count_results['dead_col_percent']:.1f}%)")
                
                # Free up memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing {model_name} at sparsity {target_sparsity}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save individual model results
        if param_counts:
            df = pd.DataFrame(param_counts)
            df = df.sort_values('target_sparsity')  # Sort for cleaner visualization
            csv_path = os.path.join(args.output_dir, f"{model_name}_dead_params.csv")
            df.to_csv(csv_path, index=False)
            print(f"Dead parameter results for {model_name} saved to {csv_path}")
            
            # Create plots
            try:
                plt.figure(figsize=(12, 8))
                
                # Plot parameter sparsity
                plt.plot(df['actual_sparsity'], df['dead_param_percent'], 'o-', color='blue', 
                         label='Dead Parameters (%)')
                
                # Plot row death rate
                plt.plot(df['actual_sparsity'], df['dead_row_percent'], 's-', color='red',
                        label='Dead Rows (%)')
                
                # Plot column death rate
                plt.plot(df['actual_sparsity'], df['dead_col_percent'], '^-', color='green',
                        label='Dead Columns (%)')
                
                plt.xlabel('Sparsity')
                plt.ylabel('Percentage (%)')
                plt.title(f'Dead Units in {model_name}')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                # Save plot
                plt_path = os.path.join(args.output_dir, f"{model_name}_dead_params.png")
                plt.savefig(plt_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Plot saved to {plt_path}")
            except Exception as e:
                print(f"Error creating plot for {model_name}: {str(e)}")
    
    # Create combined results file
    try:
        combined_counts = []
        for file in os.listdir(args.output_dir):
            if file.endswith("_dead_params.csv"):
                file_path = os.path.join(args.output_dir, file)
                df = pd.read_csv(file_path)
                combined_counts.append(df)
        
        if combined_counts:
            combined_df = pd.concat(combined_counts, ignore_index=True)
            combined_csv = os.path.join(args.output_dir, "all_models_dead_params.csv")
            combined_df.to_csv(combined_csv, index=False)
            print(f"Combined dead parameter results saved to {combined_csv}")
            
            # Create comparison plot by regularization group
            if 'group' in combined_df.columns:
                plt.figure(figsize=(16, 10))
                
                # Get unique groups
                groups = combined_df['group'].unique()
                
                # Define markers and colors for each group
                markers = {'L1': 's', 'Spatial': 'v', 'Combo': 'D', 'L1Only': '^', 'Other': 'o'}
                colors = {'L1': 'red', 'Spatial': 'green', 'Combo': 'blue', 'L1Only': 'purple', 'Other': 'gray'}
                
                # Plot parameter death rate by group
                for group in groups:
                    group_data = combined_df[combined_df['group'] == group]
                    # Group by sparsity and average
                    sparsity_data = group_data.groupby('target_sparsity')[['dead_param_percent']].mean().reset_index()
                    plt.plot(sparsity_data['target_sparsity'], sparsity_data['dead_param_percent'], 
                             marker=markers.get(group, 'o'), color=colors.get(group, 'black'),
                             linewidth=2, markersize=10, label=f"{group}")
                
                plt.xlabel('Target Sparsity', fontsize=14)
                plt.ylabel('Average Dead Parameters (%)', fontsize=14)
                plt.title('Dead Parameters by Regularization Group', fontsize=18)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=12)
                
                # Save plot
                plt_path = os.path.join(args.output_dir, "dead_params_by_group.png")
                plt.savefig(plt_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Group comparison plot saved to {args.output_dir}")
    except Exception as e:
        print(f"Error creating combined results: {str(e)}")
    
    print("Dead parameter counting complete!")

if __name__ == '__main__':
    main()