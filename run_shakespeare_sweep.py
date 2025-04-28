#!/usr/bin/env python
"""
Pareto Front Exploration for Model Regularization
Trains and evaluates models with specified hyperparameter combinations
to generate data for Pareto front analysis.
"""

import os
import sys
import subprocess
import argparse
import logging
import glob
import shutil
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "regularization_pareto")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Sparsity levels to evaluate
SPARSITY_LEVELS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]

# Hyperparameter combinations to evaluate
HYPERPARAMS = {
    "Baseline": [{"l1_scale": 0.0, "weight_decay": 0.0, "spatial_cost_scale": 0.0}],
    
    "L1_Only": [{"l1_scale": val, "weight_decay": 0.0, "spatial_cost_scale": 0.0} 
                for val in [0.5, 1, 2, 4, 8, 16, 32, 64, 128]],
    
    "L2_Only": [{"l1_scale": 0.0, "weight_decay": val, "spatial_cost_scale": 0.0} 
                for val in [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2]],
    
    "Spatial_Only": [{"l1_scale": 0.0, "weight_decay": 0.0, "spatial_cost_scale": val} 
                     for val in [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]],
    
    "L1_Spatial": [{"l1_scale": l1, "weight_decay": 0.0, "spatial_cost_scale": spatial} 
                   for l1 in [1, 4, 16, 32, 128] 
                   for spatial in [5, 10, 40, 100, 250]],
    
    "L2_Spatial": [{"l1_scale": 0.0, "weight_decay": l2, "spatial_cost_scale": spatial} 
                   for l2 in [0.1, 0.4, 1, 2.5, 5] 
                   for spatial in [5, 10, 40, 100, 250]],
    
    "L1_L2": [{"l1_scale": l1, "weight_decay": l2, "spatial_cost_scale": 0.0} 
              for l1 in [1, 4, 16, 32, 128] 
              for l2 in [0.1, 0.4, 1, 2.5, 5]],
    
    "All": [{"l1_scale": l1, "weight_decay": l2, "spatial_cost_scale": spatial} 
            for l1 in [1, 4, 16, 32, 128] 
            for l2 in [0.1, 0.4, 1, 2.5, 5] 
            for spatial in [5, 10, 40, 100]],

    "2DSpatial": [{"l1_scale": 0.0, "weight_decay": 0.0, "spatial_cost_scale": spatial, "spatial_d_value": d} 
                 for spatial in [5, 10, 40, 100, 250, 400]
                 for d in [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 4.0, 10.0, 20.0, 40.0, 80.0]]
}

def create_directories():
    """Create all necessary directories for the experiment."""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Create both potential results directories
    results_dir = os.path.join(OUTPUT_ROOT, "results")
    sweep_results_dir = os.path.join(OUTPUT_ROOT, "sweep_results")
    pareto_dir = os.path.join(OUTPUT_ROOT, "sweep_pareto_fronts")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(sweep_results_dir, exist_ok=True)
    os.makedirs(pareto_dir, exist_ok=True)
    
    logger.info(f"Created output directories in {OUTPUT_ROOT}")
    return results_dir, pareto_dir

def get_model_name(category, params):
    """Generate a model name from hyperparameters."""
    name_parts = [category]
    for key, value in params.items():
        # Use p for decimal point in filenames
        name_parts.append(f"{key}_{str(value).replace('.', 'p')}")
    
    return "_".join(name_parts)

def get_model_category(model_name):
    """Extract the category from a model name - handles existing results"""
    # First, try a direct match with the first component
    first_part = model_name.split('_')[0]
    
    if first_part in HYPERPARAMS:
        return first_part
        
    # If that fails, try each of the hyperparameter categories as a prefix match
    for category in HYPERPARAMS.keys():
        if model_name.startswith(category + "_"):
            return category
    
    # If still no match, return Unknown
    return "Unknown"

def run_training(name, l1_scale, weight_decay, spatial_cost_scale, spatial_d_value, max_iters=5000, gpu_id=0):
    """
    Run a training configuration with the specified parameters on the given GPU.
    """
    out_dir = os.path.join(OUTPUT_ROOT, name)
    if os.path.exists(out_dir):
        # Check if training already completed successfully
        checkpoint_files = glob.glob(os.path.join(out_dir, "*ckpt.pt"))
        if checkpoint_files:
            logger.info(f"Training for {name} already completed, skipping")
            return out_dir
        else:
            # Clear directory to restart training
            shutil.rmtree(out_dir)
    
    os.makedirs(out_dir, exist_ok=True)
    
    logger.info(f"Training {name} with L1={l1_scale}, L2={weight_decay}, Spatial={spatial_cost_scale}, D={spatial_d_value} on GPU {gpu_id}")
    
    # Find base config
    base_config_path = os.path.join(SCRIPT_DIR, "config", "shakespeare", "train_shakespeare_char.py")
    if not os.path.exists(base_config_path):
        base_config_path = os.path.join(SCRIPT_DIR, "config", "train_shakespeare_char.py")
        if not os.path.exists(base_config_path):
            logger.error(f"Base config not found")
            return None
    
    # Create config file - use forward slashes for Linux paths
    config_content = f"""# Configuration for {name}
# Import base configuration
exec(open("{base_config_path}").read())

# Override parameters
out_dir = "{out_dir}"
l1_scale = {l1_scale}
weight_decay = {weight_decay}
spatial_cost_scale = {spatial_cost_scale}
spatial_d_value = {spatial_d_value}
max_iters = {max_iters}
wandb_log = False
wandb_run_name = '{name}'
"""
    
    config_file = os.path.join(out_dir, "config.py")
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    # Run the training script on the specified GPU
    train_script = os.path.join(SCRIPT_DIR, "train.py")
    logger.info(f"Starting training for {name} on GPU {gpu_id}")
    
    # Set environment variable to specify GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Linux doesn't need shell=True for basic commands
    use_shell = False
    cmd = ["python", train_script, config_file]
    
    # Create output log files
    stdout_log = os.path.join(out_dir, "train_log.txt")
    stderr_log = os.path.join(out_dir, "train_error.txt")
    
    # Open log files for writing
    with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
        # Create the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=use_shell,
            env=env,
            cwd=SCRIPT_DIR,
            bufsize=1  # Line buffered
        )
        
        # Create threads to handle stdout and stderr
        def read_output(pipe, prefix, file):
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        log_line = f"[{name}] {line.rstrip()}"
                        logger.info(log_line)
                        file.write(line)
                        file.flush()  # Ensure output is written immediately
            except Exception as e:
                logger.error(f"Error reading {prefix} stream: {e}")
            finally:
                pipe.close()
        
        # Start the output threads
        stdout_thread = threading.Thread(
            target=read_output, 
            args=(process.stdout, "OUT", stdout_file)
        )
        stderr_thread = threading.Thread(
            target=read_output, 
            args=(process.stderr, "ERR", stderr_file)
        )
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete
        try:
            # Set a timeout (will continue after this if not finished)
            timeout = 24 * 3600  # 24 hours max
            exit_code = process.wait(timeout=timeout)
            
            # Make sure all output is processed
            stdout_thread.join(2)
            stderr_thread.join(2)
            
            # Check if training was successful
            if exit_code != 0:
                logger.error(f"Training failed for {name} with exit code {exit_code}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Training timed out for {name}")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            return None
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected, terminating process...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
    
    # Check if training produced a checkpoint
    checkpoint_files = glob.glob(os.path.join(out_dir, "*ckpt.pt"))
    if not checkpoint_files:
        logger.error(f"No checkpoint found for {name}")
        return None
    
    logger.info(f"Training completed successfully for {name}")
    return out_dir

def evaluate_model(model_name, model_dir, gpu_id=0):
    """
    Evaluate a trained model at different sparsity levels.
    """
    # Check both possible result directories
    results_dir = os.path.join(OUTPUT_ROOT, "results")
    sweep_results_dir = os.path.join(OUTPUT_ROOT, "sweep_results")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Look for results in both possible locations
    results_file = os.path.join(results_dir, f"{model_name}_sparsity.csv")
    sweep_results_file = os.path.join(sweep_results_dir, f"{model_name}_sparsity.csv")
    
    if os.path.exists(results_file):
        logger.info(f"Results already exist for {model_name}, skipping evaluation")
        return results_file
    
    logger.info(f"Evaluating model {model_name} on GPU {gpu_id}")
    
    # Run evaluation script
    eval_script = os.path.join(SCRIPT_DIR, "sparsify_and_evaluate_shakespeare.py")
    data_dir = os.path.join(SCRIPT_DIR, "data", "shakespeare_char")
    
    # Set environment variable to specify GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Use array of arguments for Linux
    use_shell = False
    cmd = [
        "python", 
        eval_script, 
        f"--experiments_dir={OUTPUT_ROOT}", 
        f"--data_dir={data_dir}", 
        "--block_size=256", 
        "--batch_size=64", 
        "--eval_iters=200", 
        "--results_dir=results"
    ]
    
    # Create output log files
    stdout_log = os.path.join(results_dir, f"{model_name}_eval_log.txt")
    stderr_log = os.path.join(results_dir, f"{model_name}_eval_error.txt")
    
    # Open log files for writing
    with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
        # Create the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=use_shell,
            env=env,
            cwd=SCRIPT_DIR,
            bufsize=1  # Line buffered
        )
        
        # Create threads to handle stdout and stderr
        def read_output(pipe, prefix, file):
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        log_line = f"[EVAL:{model_name}] {line.rstrip()}"
                        logger.info(log_line)
                        file.write(line)
                        file.flush()  # Ensure output is written immediately
            except Exception as e:
                logger.error(f"Error reading {prefix} stream: {e}")
            finally:
                pipe.close()
        
        # Start the output threads
        stdout_thread = threading.Thread(
            target=read_output, 
            args=(process.stdout, "OUT", stdout_file)
        )
        stderr_thread = threading.Thread(
            target=read_output, 
            args=(process.stderr, "ERR", stderr_file)
        )
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete
        try:
            # Set a timeout (will continue after this if not finished)
            timeout = 12 * 3600  # 12 hours max
            exit_code = process.wait(timeout=timeout)
            
            # Make sure all output is processed
            stdout_thread.join(2)
            stderr_thread.join(2)
            
            # Check if evaluation was successful
            if exit_code != 0:
                logger.error(f"Evaluation failed for {model_name} with exit code {exit_code}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Evaluation timed out for {model_name}")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            return None
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected, terminating process...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
    
    # Check if evaluation produced results
    if os.path.exists(results_file):
        logger.info(f"Evaluation completed successfully for {model_name}")
        return results_file
    elif os.path.exists(sweep_results_file):
        logger.info(f"Evaluation completed successfully for {model_name} (found in sweep_results)")
        return sweep_results_file
    else:
        logger.error(f"Results file not found for {model_name} in either results or sweep_results")
        return None

def process_model(model_info, gpu_id=0):
    """Process a single model (train and evaluate)."""
    category = model_info["category"]
    params = model_info["params"]
    model_name = get_model_name(category, params)
    
    # Train model
    model_dir = run_training(
        name=model_name,
        l1_scale=params["l1_scale"],
        weight_decay=params["weight_decay"],
        spatial_cost_scale=params["spatial_cost_scale"],
        spatial_d_value=params.get("spatial_d_value", 0),  # Added D value parameter
        max_iters=5000,  # Default to 5000 iterations
        gpu_id=gpu_id
    )
    
    if not model_dir:
        logger.error(f"Training failed for {model_name}")
        return None
    
    # Evaluate model
    results_file = evaluate_model(model_name, model_dir, gpu_id=gpu_id)
    
    if not results_file:
        logger.error(f"Evaluation failed for {model_name}")
        return None
    
    return results_file

def generate_pareto_fronts(results_dir, pareto_dir):
    """Generate Pareto fronts for each category and sparsity level."""
    logger.info("Generating Pareto fronts")
    
    # Load all results from both possible directories
    all_results = []
    
    # Try both the main results directory and sweep_results directory
    result_files = glob.glob(os.path.join(results_dir, "*.csv"))
    sweep_results_dir = os.path.join(OUTPUT_ROOT, "sweep_results")
    sweep_result_files = glob.glob(os.path.join(sweep_results_dir, "*.csv"))
    
    # Combine files from both directories
    result_files.extend(sweep_result_files)
    logger.info(f"Found {len(result_files)} result files across all directories")
    
    # Load all results from both possible directories
    all_results = []
    
    # Try both the main results directory and sweep_results directory
    result_files = glob.glob(os.path.join(results_dir, "*.csv"))
    sweep_results_dir = os.path.join(OUTPUT_ROOT, "sweep_results")
    sweep_result_files = glob.glob(os.path.join(sweep_results_dir, "*.csv"))
    
    # Combine files from both directories
    result_files.extend(sweep_result_files)
    logger.info(f"Found {len(result_files)} result files across all directories")
    
    for file in result_files:
        try:
            df = pd.read_csv(file)
            
            # Apply the category function to each model in the dataframe
            categories = []
            for model_name in df["model_name"]:
                category = get_model_category(model_name)
                categories.append(category)
            
            # Add category column based on our function
            df["category"] = categories
            
            # Extract spatial_d_value if available
            if "spatial_d_value" not in df.columns:
                param_pattern = "spatial_d_value_"
                for idx, model_name in enumerate(df["model_name"]):
                    param_index = model_name.find(param_pattern)
                    
                    if param_index >= 0:
                        # Extract value after parameter name until next underscore
                        param_str = model_name[param_index + len(param_pattern):]
                        param_end = param_str.find("_") if "_" in param_str else len(param_str)
                        param_value = param_str[:param_end].replace("p", ".")
                        
                        try:
                            param_value = float(param_value)
                            # If this is the first one, create the column
                            if "spatial_d_value" not in df.columns:
                                df["spatial_d_value"] = 1.0  # Default
                            # Set this value
                            df.at[idx, "spatial_d_value"] = param_value
                        except ValueError:
                            pass
                
                # If column wasn't created, create it with defaults
                if "spatial_d_value" not in df.columns:
                    df["spatial_d_value"] = 1.0
            
            all_results.append(df)
        except Exception as e:
            logger.error(f"Error processing file {file}: {str(e)}")
    
    # Combine all results
    if not all_results:
        logger.error("No results found")
        return
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Log summary of data
    logger.info(f"Combined dataframe contains {len(combined_df)} rows")
    logger.info(f"Categories present: {combined_df['category'].unique()}")
    
    # Log count of models per category
    category_counts = combined_df.groupby('category').size()
    for category, count in category_counts.items():
        logger.info(f"Category {category}: {count} models")
    
    # Log count of models per sparsity level
    sparsity_counts = combined_df.groupby('target_sparsity').size()
    for sparsity, count in sparsity_counts.items():
        logger.info(f"Sparsity {sparsity}: {count} models")
    
    combined_df.to_csv(os.path.join(results_dir, "all_models_results.csv"), index=False)
    
    # Generate Pareto fronts for each category and sparsity level
    # Only generate these for categories in HYPERPARAMS
    for category in HYPERPARAMS.keys():
        if category in combined_df["category"].unique():
            # Get only the models explicitly from this category
            category_df = combined_df[combined_df["category"] == category]
            
            # Debug information
            logger.info(f"Processing category {category} with {len(category_df)} models")
            
            # Create a figure for this category
            plt.figure(figsize=(12, 8))
            
            for sparsity in SPARSITY_LEVELS:
                sparsity_df = category_df[category_df["target_sparsity"] == sparsity]
                
                if len(sparsity_df) == 0:
                    continue
                
                # Sort by validation loss
                sparsity_df = sparsity_df.sort_values("val_loss")
                
                # Debug: Log the models in this category/sparsity
                model_names = sparsity_df["model_name"].tolist()
                logger.info(f"  {category} at sparsity {sparsity:.2f} has {len(model_names)} models: {model_names[:2]}...")
                
                # Verify each model actually belongs to this category
                verified_models = []
                for idx, row in sparsity_df.iterrows():
                    model_name = row["model_name"]
                    model_category = get_model_category(model_name)
                    if model_category == category:
                        verified_models.append(row)
                    else:
                        logger.warning(f"  Removing model {model_name} (category {model_category}) from {category} results")
                
                # If we removed some models, recreate the dataframe
                if len(verified_models) < len(sparsity_df):
                    sparsity_df = pd.DataFrame(verified_models)
                    
                    # If no models left, skip
                    if len(sparsity_df) == 0:
                        logger.warning(f"  No valid models for {category} at sparsity {sparsity:.2f}")
                        continue
                
                # Save to CSV
                pareto_file = os.path.join(pareto_dir, f"{category}_sparsity_{sparsity:.2f}.csv")
                sparsity_df.to_csv(pareto_file, index=False)
                
                # Plot sparsity vs loss
                plt.plot(sparsity_df["actual_sparsity"].to_numpy(), sparsity_df["val_loss"].to_numpy(), 
                        'o-', label=f"Sparsity {sparsity:.2f}")
            
            plt.xlabel("Actual Sparsity")
            plt.ylabel("Validation Loss")
            plt.title(f"Sparsity vs Loss for {category}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot
            plt.savefig(os.path.join(pareto_dir, f"{category}_pareto_front.png"))
            plt.close()
    
    # Create comparison plots across all hyperparameter categories
    # Define plot types based on HYPERPARAMS keys
    for plot_type, categories_to_plot, filename in [
        ("All Categories", list(HYPERPARAMS.keys()), "best_models_comparison_all.png"),
        ("Top Performers", ["L1_Only", "L1_Spatial", "L1_L2", "All", "2DSpatial"], "best_models_comparison_top.png")
    ]:
        plt.figure(figsize=(15, 10))
        
        # Filter categories from our HYPERPARAMS
        valid_categories = [cat for cat in categories_to_plot if cat in combined_df["category"].unique()]
        
        # For each sparsity level
        for sparsity in SPARSITY_LEVELS:
            # For each category, find the best model at this sparsity
            best_models = []
            
            for category in valid_categories:
                category_df = combined_df[(combined_df["category"] == category) & 
                                        (combined_df["target_sparsity"] == sparsity)]
                
                if len(category_df) > 0:
                    # Get the model with the lowest validation loss
                    best_model = category_df.loc[category_df["val_loss"].idxmin()]
                    best_models.append(best_model)
            
            if best_models:
                # Create a dataframe with the best models
                best_df = pd.DataFrame(best_models)
                
                # Save to CSV
                best_file = os.path.join(pareto_dir, f"best_models_{plot_type.lower().replace(' ', '_')}_sparsity_{sparsity:.2f}.csv")
                best_df.to_csv(best_file, index=False)
                
                # Plot category vs loss
                plt.scatter(best_df["category"].to_numpy(), best_df["val_loss"].to_numpy(), 
                          label=f"Sparsity {sparsity:.2f}", s=100)
        
        plt.xlabel("Category")
        plt.ylabel("Best Validation Loss")
        plt.title(f"Best Models by {plot_type} and Sparsity")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        
        # Save plot
        plt.savefig(os.path.join(pareto_dir, filename))
        plt.close()
    
    # For 2DSpatial category, create a special comparison plot by D value
    if "2DSpatial" in combined_df["category"].values:
        plt.figure(figsize=(12, 8))
        
        # Filter for 2DSpatial models
        spatial_df = combined_df[combined_df["category"] == "2DSpatial"]
        
        # Group by D value and sparsity
        d_values = sorted(spatial_df["spatial_d_value"].unique())
        
        # Create a colormap for D values
        d_cmap = plt.cm.get_cmap("viridis", len(d_values))
        
        # For each D value
        for i, d in enumerate(d_values):
            d_data = spatial_df[spatial_df["spatial_d_value"] == d]
            
            # For each sparsity level
            sparsity_results = []
            
            for sparsity in SPARSITY_LEVELS:
                sparsity_data = d_data[d_data["target_sparsity"] == sparsity]
                
                if len(sparsity_data) > 0:
                    # Get the model with the lowest validation loss
                    best_model = sparsity_data.loc[sparsity_data["val_loss"].idxmin()]
                    sparsity_results.append((sparsity, best_model["val_loss"]))
            
            if sparsity_results:
                sparsities, losses = zip(*sparsity_results)
                plt.plot(np.array(sparsities), np.array(losses), 'o-', 
                       label=f"D={d}", 
                       color=d_cmap(i))
        
        plt.xlabel("Target Sparsity")
        plt.ylabel("Best Validation Loss")
        plt.title("Effect of D Value on 2D Spatial Regularization")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(pareto_dir, "2dspatial_d_value_comparison.png"))
        plt.close()
    
    # Create a separate plot with just the top performing groups
    plt.figure(figsize=(15, 10))
    
    # Only include specific top performing groups, excluding 2DSpatial due to scaling issues
    top_groups = ["L1_Only", "L1_Spatial", "L1_L2", "All"]
    valid_top_groups = [group for group in top_groups 
                       if group in combined_df["category"].unique()]
    
    # Track if we successfully plotted anything
    plotted_any = False
    
    # Create a colormap for the groups
    top_group_cmap = plt.cm.get_cmap("tab10", len(valid_top_groups))
    
    # For each top hyperparameter group
    for i, group in enumerate(valid_top_groups):
        # Find all models in this group
        group_models = combined_df[combined_df["category"] == group]
        
        if len(group_models) == 0:
            logger.warning(f"No models found for top group {group}")
            continue
        
        # Get best model at each sparsity level
        sparsity_results = []
        
        for sparsity in SPARSITY_LEVELS:
            sparsity_models = group_models[group_models["target_sparsity"] == sparsity]
            
            if len(sparsity_models) > 0:
                # Get the model with the lowest validation loss
                best_model = sparsity_models.loc[sparsity_models["val_loss"].idxmin()]
                sparsity_results.append((sparsity, best_model["val_loss"]))
        
        if sparsity_results:
            sparsities, losses = zip(*sparsity_results)
            plt.plot(np.array(sparsities), np.array(losses), 'o-', 
                   label=group,
                   color=top_group_cmap(i),
                   linewidth=2,
                   markersize=8)
            plotted_any = True
    
    # Only add these elements if we actually plotted data
    if plotted_any:
        plt.xlabel("Target Sparsity")
        plt.ylabel("Best Validation Loss")
        plt.title("Comparison of Top Performing Hyperparameter Groups")
        plt.grid(True, alpha=0.3)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No valid data to plot", 
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes)
    
    # Save plot
    plt.savefig(os.path.join(pareto_dir, "hyperparameter_group_comparison_top.png"))
    plt.close()
    
    # Create another special plot with just 2DSpatial by itself
    plt.figure(figsize=(15, 10))
    
    # Find 2DSpatial models
    spatial_models = combined_df[combined_df["category"] == "2DSpatial"]
    
    if len(spatial_models) > 0:
        # Get best model at each sparsity level
        sparsity_results = []
        
        for sparsity in SPARSITY_LEVELS:
            sparsity_models = spatial_models[spatial_models["target_sparsity"] == sparsity]
            
            if len(sparsity_models) > 0:
                # Get the model with the lowest validation loss
                best_model = sparsity_models.loc[sparsity_models["val_loss"].idxmin()]
                sparsity_results.append((sparsity, best_model["val_loss"]))
        
        if sparsity_results:
            sparsities, losses = zip(*sparsity_results)
            plt.plot(np.array(sparsities), np.array(losses), 'o-', 
                   label="2DSpatial",
                   color="cyan",
                   linewidth=2,
                   markersize=8)
            
            plt.xlabel("Target Sparsity")
            plt.ylabel("Best Validation Loss")
            plt.title("Performance of 2DSpatial Models")
            plt.grid(True, alpha=0.3)
            plt.legend()
        else:
            plt.text(0.5, 0.5, "No valid 2DSpatial data to plot", 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=plt.gca().transAxes)
    else:
        plt.text(0.5, 0.5, "No 2DSpatial models found", 
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes)
    
    # Save plot
    plt.savefig(os.path.join(pareto_dir, "2dspatial_performance.png"))
    plt.close()
    
    plt.xlabel("Target Sparsity")
    plt.ylabel("Best Validation Loss")
    plt.title("Comparison of Best Models by Hyperparameter Group")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plt.savefig(os.path.join(pareto_dir, "hyperparameter_group_comparison.png"))
    plt.close()
    
    # Call our special direct comparison plot function
    generate_simple_comparison_plot(combined_df, os.path.join(pareto_dir, "hyperparameter_group_comparison.png"))
    
    logger.info("Pareto front generation completed")

def worker(model_queue, results_queue, gpu_id):
    """Worker function for parallel processing."""
    while not model_queue.empty():
        try:
            model_info = model_queue.get()
            results = process_model(model_info, gpu_id)
            results_queue.put({"model_info": model_info, "results": results})
        except Exception as e:
            logger.error(f"Error in worker on GPU {gpu_id}: {str(e)}")
        finally:
            model_queue.task_done()

def generate_simple_comparison_plot(combined_df, output_path):
    """Generate a very simple comparison plot directly from the data."""
    plt.figure(figsize=(15, 10))
    
    # Define the categories we want to plot
    categories = ["Baseline", "L1_Only", "L2_Only", "Spatial_Only", 
                 "L1_Spatial", "L2_Spatial", "L1_L2", "All", "2DSpatial"]
    
    # Define nice colors for the categories
    colors = ['blue', 'orange', 'green', 'red', 'purple', 
              'brown', 'pink', 'gray', 'cyan']
    
    # Create a dictionary to store the best model for each category and sparsity
    best_models = {}
    
    # Initialize the dictionary
    for category in categories:
        best_models[category] = {}
        for sparsity in SPARSITY_LEVELS:
            best_models[category][sparsity] = None
    
    # Find the best model for each category and sparsity
    for _, row in combined_df.iterrows():
        model_name = row["model_name"]
        category = None
        
        # Determine the category from the model name
        for cat in categories:
            if model_name.startswith(cat + "_"):
                category = cat
                break
        
        if category is None:
            continue
            
        sparsity = row["target_sparsity"]
        val_loss = row["val_loss"]
        
        # Check if this is the best model for this category and sparsity
        if sparsity in best_models[category]:
            current_best = best_models[category][sparsity]
            if current_best is None or val_loss < current_best[1]:
                best_models[category][sparsity] = (model_name, val_loss)
    
    # Plot the best models for each category
    for i, category in enumerate(categories):
        # Get the sparsity and loss values
        points = []
        for sparsity in SPARSITY_LEVELS:
            if sparsity in best_models[category] and best_models[category][sparsity] is not None:
                points.append((sparsity, best_models[category][sparsity][1]))
        
        if points:
            # Sort by sparsity
            points.sort(key=lambda x: x[0])
            
            # Extract x and y values
            x_values = [p[0] for p in points]
            y_values = [p[1] for p in points]
            
            # Plot
            plt.plot(x_values, y_values, 'o-', 
                   label=category, 
                   color=colors[i % len(colors)],
                   linewidth=2,
                   markersize=8)
    
    # Add labels and legend
    plt.xlabel("Target Sparsity")
    plt.ylabel("Best Validation Loss")
    plt.title("Comparison of Best Models by Hyperparameter Group")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()

def main():
    """Main function to run the parameter search."""
    parser = argparse.ArgumentParser(description='Parameter Search for Regularization Experiments')
    parser.add_argument('--gpu-split', type=int, choices=[0, 1], default=None,
                      help='Split jobs across 2 GPUs. Use 0 for first half, 1 for second half')
    parser.add_argument('--categories', nargs='+', choices=list(HYPERPARAMS.keys()) + ['all'],
                      default=['all'], help='Categories to run')
    parser.add_argument('--max-iters', type=int, default=5000,
                      help='Maximum training iterations')
    parser.add_argument('--eval-only', action='store_true',
                      help='Only evaluate existing models and generate Pareto fronts')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                      help='GPU IDs to use for training')
    args = parser.parse_args()
    
    # Create directories
    results_dir, pareto_dir = create_directories()
    
    # If eval-only, just generate Pareto fronts and exit
    if args.eval_only:
        logger.info("Generating Pareto fronts from existing results")
        generate_pareto_fronts(results_dir, pareto_dir)
        return
    
    # Build the list of models to train
    models_to_process = []
    
    categories = list(HYPERPARAMS.keys()) if 'all' in args.categories else args.categories
    
    for category in categories:
        for params in HYPERPARAMS[category]:
            models_to_process.append({
                "category": category,
                "params": params
            })
    
    # If gpu-split is specified, split the models
    if args.gpu_split is not None:
        n_models = len(models_to_process)
        half_point = n_models // 2
        
        if args.gpu_split == 0:
            models_to_process = models_to_process[:half_point]
            logger.info(f"Processing first half: {len(models_to_process)} models")
        else:
            models_to_process = models_to_process[half_point:]
            logger.info(f"Processing second half: {len(models_to_process)} models")
    
    # Process models
    if len(args.gpus) == 1:
        # Single GPU processing
        gpu_id = args.gpus[0]
        logger.info(f"Processing {len(models_to_process)} models on GPU {gpu_id}")
        
        for model_info in models_to_process:
            process_model(model_info, gpu_id=gpu_id)
            
    else:
        # Multi-GPU processing
        logger.info(f"Processing {len(models_to_process)} models on GPUs {args.gpus}")
        
        # Create queues for models and results
        with Manager() as manager:
            model_queue = manager.JoinableQueue()
            results_queue = manager.Queue()
            
            # Add models to queue
            for model_info in models_to_process:
                model_queue.put(model_info)
            
            # Create workers for each GPU
            processes = []
            for gpu_id in args.gpus:
                p = Process(target=worker, args=(model_queue, results_queue, gpu_id))
                p.start()
                processes.append(p)
            
            # Wait for all models to be processed
            model_queue.join()
            
            # Terminate processes
            for p in processes:
                p.terminate()
    
    # Generate Pareto fronts
    generate_pareto_fronts(results_dir, pareto_dir)
    
    logger.info("Parameter search completed")

if __name__ == "__main__":
    main()
