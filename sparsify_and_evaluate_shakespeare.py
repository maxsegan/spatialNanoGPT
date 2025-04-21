"""
Sparsify and evaluate trained models at different sparsity levels.
This script checks for existing evaluation results and only processes new models.

This script:
1. Finds all trained models from regularization experiments
2. Checks if evaluation results already exist for each model
3. For new models only, applies different levels of sparsity (0% to 80%)
4. Evaluates the performance at each sparsity level
5. Saves results to CSV files and updates the combined results
"""

import os
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import csv
import matplotlib.pyplot as plt
from model import GPT, GPTConfig

# Configure argument parser
parser = argparse.ArgumentParser(description="Sparsify and evaluate models")
parser.add_argument('--experiments_dir', type=str, default='regularization_experiments',
                    help='Directory with all experiment folders')
parser.add_argument('--data_dir', type=str, default='data/shakespeare_char',
                    help='Directory with dataset')
parser.add_argument('--block_size', type=int, default=256,
                    help='Block size for evaluation')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for evaluation')
parser.add_argument('--eval_iters', type=int, default=200,
                    help='Number of iterations for evaluation')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='Device to run evaluation on')
parser.add_argument('--results_dir', type=str, default='sparsity_results',
                    help='Directory to save results')
parser.add_argument('--force_reevaluate', action='store_true',
                    help='Force re-evaluation of all models, even if results exist')
args = parser.parse_args()

# Create results directory
results_path = os.path.join(args.experiments_dir, args.results_dir)
os.makedirs(results_path, exist_ok=True)

# Define sparsity levels
sparsity_levels = [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]

# Set up data loading
def get_batch(split, data_dir=args.data_dir, block_size=args.block_size, batch_size=args.batch_size):
    """Get a random batch from the dataset."""
    data_file = os.path.join(data_dir, f'{split}.bin')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} not found")
    
    data = np.memmap(data_file, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(args.device), y.to(args.device)
    return x, y

def create_fixed_batches(n_batches=args.eval_iters):
    """Create fixed batches for consistent evaluation."""
    fixed_batches = []
    for i in range(n_batches):
        x, y = get_batch('val')
        fixed_batches.append((x, y))
    return fixed_batches

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

@torch.no_grad()
def estimate_loss(model, fixed_batches):
    """Estimate model loss on validation data using fixed batches in parallel."""
    model.eval()
    batch_count = len(fixed_batches)
    losses = torch.zeros(batch_count, device=args.device)
    
    # Process batches in chunks of 20 (or less for the final chunk)
    chunk_size = 20
    for i in range(0, batch_count, chunk_size):
        end_idx = min(i + chunk_size, batch_count)
        current_chunk = fixed_batches[i:end_idx]
        
        batch_X = []
        batch_Y = []
        for X, Y in current_chunk:
            batch_X.append(X)
            batch_Y.append(Y)
        
        batched_X = torch.cat(batch_X, dim=0)
        batched_Y = torch.cat(batch_Y, dim=0)
        
        logits, loss = model(batched_X, batched_Y)
        
        # Still need to test this
        # If the model returns a single loss, we need to reshape the outputs
        # to get individual losses for each original batch
        if isinstance(loss, torch.Tensor) and loss.numel() == 1:
            # The model would need to be modified to return per-sample losses
            # For now, we'll just duplicate the loss for each batch in the chunk
            chunk_losses = torch.full((end_idx - i,), loss.item(), device=args.device)
        else:
            # Assuming the model now returns a tensor of losses, one per original batch
            chunk_losses = loss
            
        losses[i:end_idx] = chunk_losses
    
    return losses.mean().item()

def load_model_from_checkpoint(checkpoint_path, device=args.device):
    """Load a model from a checkpoint file with support for spatial modes."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None, None
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_args = checkpoint['model_args']
        
        # Create the model
        config = GPTConfig(**model_args)
        model = GPT(config)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        
        # Extract regularization parameters
        regularization = checkpoint.get('regularization', {
            'l1_scale': 0.0,
            'spatial_cost_scale': 0.0,
            'weight_decay': 0.0,
            'spatial_mode': 'fixed',  # Default mode
        })
        
        # Handle spatial matrices for different modes
        spatial_mode = regularization.get('spatial_mode', 'fixed')
        
        # If the model was using a spatial regularization mode, we need to properly initialize it
        if regularization.get('spatial_cost_scale', 0.0) > 0:
            from regularized_gpt import RegularizedGPT
            
            # Recreate the regularized model with the same parameters
            regularized_model = RegularizedGPT(
                model=model,
                l1_scale=regularization.get('l1_scale', 0.0),
                spatial_cost_scale=regularization.get('spatial_cost_scale', 0.0),
                A=1.0,  # Default values - should match what was used in training
                B=1.0,
                D=1.0,
                device=device,
                spatial_mode=spatial_mode
            )
            
            # For swappable mode, load the optimized distance matrices if available
            if spatial_mode == "swappable" and 'linear_distance_matrices' in checkpoint and 'value_distance_matrices' in checkpoint:
                spatial_net = regularized_model.spatial_net
                
                # Load linear distance matrices
                for i, matrix in enumerate(checkpoint['linear_distance_matrices']):
                    if i < len(spatial_net.linear_distance_matrices):
                        spatial_net.linear_distance_matrices[i] = matrix.to(device)
                
                # Load value distance matrices
                for i, matrix in enumerate(checkpoint['value_distance_matrices']):
                    if i < len(spatial_net.value_distance_matrices):
                        spatial_net.value_distance_matrices[i] = matrix.to(device)
                        
                print(f"Loaded optimized distance matrices from checkpoint for {spatial_mode} mode")
            
            # For learnable mode, load the spatial state dict
            elif spatial_mode == "learnable" and 'spatial_state' in checkpoint:
                regularized_model.spatial_net.load_state_dict(checkpoint['spatial_state'])
                print(f"Loaded learnable spatial parameters from checkpoint for {spatial_mode} mode")
            
            # Use the underlying model for sparsification
            model = regularized_model.model
        
        return model, regularization
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def find_checkpoints(experiments_dir=args.experiments_dir):
    """Find all checkpoint files in experiment directories."""
    checkpoints = []
    summary_csv = os.path.join(experiments_dir, 'experiment_summary.csv')
    
    if os.path.exists(summary_csv):
        # Read experiment summary if it exists
        try:
            # First, check the headers to deal with possible variations
            with open(summary_csv, 'r') as f:
                # Get the header line
                header_line = f.readline().strip()
                header = header_line.split(',')
                
            # Map the possible column names
            name_key = next((col for col in header if col.lower() == 'name'), None)
            l1_key = next((col for col in header if 'l1' in col.lower() and 'scale' in col.lower()), None)
            l2_key = next((col for col in header if ('l2' in col.lower() or 'weight' in col.lower() or 'decay' in col.lower())), None)
            spatial_key = next((col for col in header if 'spatial' in col.lower()), None)
            out_dir_key = next((col for col in header if 'output' in col.lower() or 'directory' in col.lower() or 'dir' in col.lower()), None)
            
            if not all([name_key, out_dir_key]):
                print(f"Warning: Could not find essential columns in {summary_csv}")
                print(f"Header: {header}")
                print("Falling back to directory scan")
                raise ValueError("Missing essential columns")
            
            # Now read the actual data
            df = pd.read_csv(summary_csv)
            
            for _, row in df.iterrows():
                exp_dir = row[out_dir_key]
                # Look for checkpoint in this directory
                ckpt_files = glob.glob(os.path.join(exp_dir, '*ckpt.pt'))
                if ckpt_files:
                    # Use the first checkpoint found
                    checkpoint_info = {
                        'name': row[name_key],
                        'checkpoint_path': ckpt_files[0]
                    }
                    
                    # Add regularization parameters if available
                    if l1_key: 
                        checkpoint_info['l1_scale'] = float(row[l1_key])
                    if l2_key:
                        checkpoint_info['weight_decay'] = float(row[l2_key])
                    if spatial_key:
                        checkpoint_info['spatial_cost_scale'] = float(row[spatial_key])
                    
                    checkpoints.append(checkpoint_info)
        except Exception as e:
            print(f"Error reading summary CSV: {str(e)}")
            print("Falling back to directory scan")
            
    # If CSV reading failed or file doesn't exist, scan directories
    if not checkpoints:
        print(f"Scanning experiment directories in {experiments_dir}...")
        exp_dirs = [d for d in os.listdir(experiments_dir) 
                    if os.path.isdir(os.path.join(experiments_dir, d))]
        
        for exp_dir in exp_dirs:
            if exp_dir == args.results_dir:  # Skip the results directory
                continue
                
            full_path = os.path.join(experiments_dir, exp_dir)
            ckpt_files = glob.glob(os.path.join(full_path, '*ckpt.pt'))
            if ckpt_files:
                # Parse name to extract regularization parameters (assuming naming convention)
                checkpoints.append({
                    'name': exp_dir,
                    'checkpoint_path': ckpt_files[0]
                })
    
    print(f"Found {len(checkpoints)} checkpoints")
    return checkpoints

def check_existing_results(model_name, results_dir=os.path.join(args.experiments_dir, args.results_dir)):
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

def load_combined_results(results_dir=os.path.join(args.experiments_dir, args.results_dir)):
    """Load the combined results file if it exists."""
    combined_csv = os.path.join(results_dir, "all_models_sparsity.csv")
    if os.path.exists(combined_csv):
        try:
            return pd.read_csv(combined_csv)
        except Exception as e:
            print(f"Error reading combined results: {str(e)}")
            return pd.DataFrame()
    return pd.DataFrame()

def update_combined_results(new_results, existing_results=None, 
                            results_dir=os.path.join(args.experiments_dir, args.results_dir)):
    """Update the combined results with new model results."""
    if existing_results is None or existing_results.empty:
        # Just write the new results
        new_results.to_csv(os.path.join(results_dir, "all_models_sparsity.csv"), index=False)
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
    combined_df.to_csv(os.path.join(results_dir, "all_models_sparsity.csv"), index=False)
    return combined_df

def main():
    # Load existing combined results
    combined_results = load_combined_results()
    if not combined_results.empty:
        print(f"Loaded existing results for {combined_results['model_name'].nunique()} models")
    
    # Create fixed validation batches for consistent evaluation
    print("Creating fixed validation batches...")
    try:
        fixed_batches = create_fixed_batches()
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print(f"Make sure the dataset is available at {args.data_dir}")
        return
    
    # Find all checkpoints
    print(f"Scanning for checkpoints in {args.experiments_dir}...")
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("No checkpoints found. Exiting.")
        return
    
    # Filter checkpoints to only process new ones or missing sparsity levels
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
    
    print(f"Will process {len(models_to_process)} models with missing sparsity levels")
    
    # Process each model
    for ckpt_info in tqdm(models_to_process, desc="Processing models"):
        checkpoint_path = ckpt_info['checkpoint_path']
        model_name = ckpt_info['name']
        missing_sparsity = ckpt_info['missing_sparsity']
        
        print(f"\nEvaluating {model_name} at {len(missing_sparsity)} missing sparsity levels...")
        
        # Get existing results or create empty results list
        if 'existing_results' in ckpt_info:
            model_results = ckpt_info['existing_results'].to_dict('records')
        else:
            model_results = []
        
        # Extract regularization parameters
        l1_scale = ckpt_info.get('l1_scale', 0.0)
        weight_decay = ckpt_info.get('weight_decay', 0.0)
        spatial_cost_scale = ckpt_info.get('spatial_cost_scale', 0.0)
        
        # Evaluate at each missing sparsity level
        for target_sparsity in tqdm(missing_sparsity, desc=f"Sparsity levels for {model_name}"):
            try:
                # Create a copy of the model for this sparsity level
                sparsified_model, regularization = load_model_from_checkpoint(checkpoint_path)
                
                if sparsified_model is None:
                    print(f"Skipping sparsity {target_sparsity} for {model_name} due to loading error")
                    continue
                
                # Update regularization parameters if available from checkpoint
                if regularization:
                    if l1_scale == 0.0 and 'l1_scale' in regularization:
                        l1_scale = regularization['l1_scale']
                    if weight_decay == 0.0 and 'weight_decay' in regularization:
                        weight_decay = regularization['weight_decay']
                    if spatial_cost_scale == 0.0 and 'spatial_cost_scale' in regularization:
                        spatial_cost_scale = regularization['spatial_cost_scale']
                    
                    # Also save spatial mode if available
                    spatial_mode = regularization.get('spatial_mode', 'fixed')
                
                # Apply pruning
                actual_sparsity = apply_magnitude_pruning(sparsified_model, target_sparsity)
                
                # Evaluate the model
                loss = estimate_loss(sparsified_model, fixed_batches)
                
                # Store results
                result = {
                    'model_name': model_name,
                    'target_sparsity': target_sparsity,
                    'actual_sparsity': actual_sparsity,
                    'val_loss': loss,
                    'l1_scale': l1_scale,
                    'weight_decay': weight_decay,
                    'spatial_cost_scale': spatial_cost_scale,
                    'spatial_mode': spatial_mode if 'spatial_mode' in locals() else 'fixed'
                }
                model_results.append(result)
                
                # Output progress
                print(f"Sparsity {target_sparsity*100:.1f}% → {actual_sparsity*100:.1f}%: Loss = {loss:.6f}")
                
                # Free up memory
                del sparsified_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error evaluating {model_name} at sparsity {target_sparsity}: {str(e)}")
                continue
        
        # Save individual model results
        if model_results:
            model_df = pd.DataFrame(model_results)
            model_csv_path = os.path.join(args.experiments_dir, args.results_dir, f"{model_name}_sparsity.csv")
            model_df.to_csv(model_csv_path, index=False)
            print(f"Results for {model_name} saved to {model_csv_path}")
            
            # Update combined results
            combined_results = update_combined_results(model_df, combined_results)
            
            # Plot sparsity vs. loss for this model
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(model_df['actual_sparsity'], model_df['val_loss'], 'o-')
                plt.xlabel('Sparsity')
                plt.ylabel('Validation Loss')
                plt.title(f'Sparsity vs. Loss for {model_name}')
                plt.grid(True, alpha=0.3)
                plt_path = os.path.join(args.experiments_dir, args.results_dir, f"{model_name}_sparsity.png")
                plt.savefig(plt_path)
                plt.close()
            except Exception as e:
                print(f"Error creating plot for {model_name}: {str(e)}")
    
    # Create comparison plot of all models (if we have combined results)
    if not combined_results.empty:
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot each model as a separate line
            for model_name in combined_results['model_name'].unique():
                model_data = combined_results[combined_results['model_name'] == model_name]
                plt.plot(model_data['actual_sparsity'], model_data['val_loss'], label=model_name)
            
            plt.xlabel('Sparsity')
            plt.ylabel('Validation Loss')
            plt.title('Sparsity vs. Loss for All Models')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize='small')
            plt_path = os.path.join(args.experiments_dir, args.results_dir, "all_models_comparison.png")
            plt.savefig(plt_path)
            plt.close()
        except Exception as e:
            print(f"Error creating comparison plot: {str(e)}")
    
    print("Evaluation complete!")

if __name__ == '__main__':
    main()