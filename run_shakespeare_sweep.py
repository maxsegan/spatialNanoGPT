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
                   for spatial in [5, 10, 40, 100, 250, 400, 750, 1000, 1500, 2000]],
    
    "L2_Spatial": [{"l1_scale": 0.0, "weight_decay": l2, "spatial_cost_scale": spatial} 
                   for l2 in [0.1, 0.4, 1, 2.5, 5] 
                   for spatial in [5, 10, 40, 100, 250, 400, 750, 1000, 1500, 2000]],
    
    "L1_L2": [{"l1_scale": l1, "weight_decay": l2, "spatial_cost_scale": 0.0} 
              for l1 in [1, 4, 16, 32, 128] 
              for l2 in [0.1, 0.4, 1, 2.5, 5]],
    
    "All": [{"l1_scale": l1, "weight_decay": l2, "spatial_cost_scale": spatial} 
            for l1 in [1, 4, 16, 32, 128] 
            for l2 in [0.1, 0.4, 1, 2.5, 5] 
            for spatial in [5, 10, 40, 100, 250, 400, 750, 1000, 1500, 2000]],

    "2DSpatial": [{"l1_scale": 0.0, "weight_decay": 0.0, "spatial_cost_scale": spatial, "spatial_d_value": d} 
                 for spatial in [5, 10, 40, 100, 250, 400, 750, 1000, 1500, 2000]
                 for d in [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]]
}

def create_directories():
    """Create all necessary directories for the experiment."""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "sweep_results"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "sweep_pareto_fronts"), exist_ok=True)
    logger.info(f"Created output directories in {OUTPUT_ROOT}")
    return os.path.join(OUTPUT_ROOT, "sweep_results"), os.path.join(OUTPUT_ROOT, "sweep_pareto_fronts")

def get_model_name(category, params):
    """Generate a model name from hyperparameters."""
    name_parts = [category]
    for key, value in params.items():
        # Use p for decimal point in filenames
        name_parts.append(f"{key}_{str(value).replace('.', 'p')}")
    
    return "_".join(name_parts)

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
    
    # Create config file
    config_content = f"""# Configuration for {name}
# Import base configuration
exec(open(r"{base_config_path.replace(os.sep, '/')}").read())

# Override parameters
out_dir = r"{out_dir.replace(os.sep, '/')}"
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
    
    # Yehuda, you probably need to change this since I assume you're on Linux
    use_shell = sys.platform == 'win32'
    cmd = f"python {train_script} {config_file}"
    
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
    results_dir = os.path.join(OUTPUT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"{model_name}_sparsity.csv")
    
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
    
    # Use platform-specific command setup
    use_shell = sys.platform == 'win32'
    cmd = f"python {eval_script} --experiments_dir={OUTPUT_ROOT} --data_dir={data_dir} --block_size=256 --batch_size=64 --eval_iters=200 --results_dir=results"
    
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
    if not os.path.exists(results_file):
        logger.error(f"Results file not found for {model_name}")
        return None
    
    logger.info(f"Evaluation completed successfully for {model_name}")
    return results_file

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
    
    # Load all results
    all_results = []
    result_files = glob.glob(os.path.join(results_dir, "*.csv"))
    
    for file in result_files:
        try:
            df = pd.read_csv(file)
            model_name = df["model_name"].iloc[0]
            
            # Extract category from model name
            category = model_name.split("_")[0]
            
            # Add category column
            df["category"] = category
            
            # Extract spatial_d_value if available
            if "spatial_d_value" not in df.columns:
                param_pattern = "spatial_d_value_"
                param_index = model_name.find(param_pattern)
                
                if param_index >= 0:
                    # Extract value after parameter name until next underscore
                    param_str = model_name[param_index + len(param_pattern):]
                    param_end = param_str.find("_") if "_" in param_str else len(param_str)
                    param_value = param_str[:param_end].replace("p", ".")
                    
                    try:
                        param_value = float(param_value)
                        df["spatial_d_value"] = param_value
                    except ValueError:
                        df["spatial_d_value"] = 1.0  # Default value if parsing fails
                else:
                    df["spatial_d_value"] = 1.0  # Default value
            
            all_results.append(df)
        except Exception as e:
            logger.error(f"Error processing file {file}: {str(e)}")
    
    # Combine all results
    if not all_results:
        logger.error("No results found")
        return
    
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(os.path.join(results_dir, "all_models_results.csv"), index=False)
    
    # Generate Pareto fronts for each category and sparsity level
    for category in combined_df["category"].unique():
        category_df = combined_df[combined_df["category"] == category]
        
        # Create a figure for this category
        plt.figure(figsize=(12, 8))
        
        for sparsity in SPARSITY_LEVELS:
            sparsity_df = category_df[category_df["target_sparsity"] == sparsity]
            
            if len(sparsity_df) == 0:
                continue
            
            # Sort by validation loss
            sparsity_df = sparsity_df.sort_values("val_loss")
            
            # Save to CSV
            pareto_file = os.path.join(pareto_dir, f"{category}_sparsity_{sparsity:.2f}.csv")
            sparsity_df.to_csv(pareto_file, index=False)
            
            # Plot sparsity vs loss
            plt.plot(sparsity_df["actual_sparsity"], sparsity_df["val_loss"], 
                    'o-', label=f"Sparsity {sparsity:.2f}")
        
        plt.xlabel("Actual Sparsity")
        plt.ylabel("Validation Loss")
        plt.title(f"Sparsity vs Loss for {category}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(pareto_dir, f"{category}_pareto_front.png"))
        plt.close()
    
    # Create two comparison plots across categories (all and top performers)
    for plot_type, categories_to_plot, filename in [
        ("All Categories", None, "best_models_comparison_all.png"),
        ("Top Performers", ["L1_Only", "L1_Spatial", "L1_L2", "All", "2DSpatial"], "best_models_comparison_top.png")
    ]:
        plt.figure(figsize=(15, 10))
        
        # Filter categories if needed
        plot_categories = combined_df["category"].unique() if categories_to_plot is None else categories_to_plot
        
        # For each sparsity level
        for sparsity in SPARSITY_LEVELS:
            # For each category, find the best model at this sparsity
            best_models = []
            
            for category in plot_categories:
                if category not in combined_df["category"].values:
                    continue
                    
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
                plt.scatter(best_df["category"], best_df["val_loss"], 
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
                plt.plot(sparsities, losses, 'o-', 
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