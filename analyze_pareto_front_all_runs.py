#!/usr/bin/env python
"""
Analyze and visualize Pareto fronts from regularization experiments.
This script processes the results of the parameter search and creates
visualizations of the Pareto fronts for different hyperparameter combinations.
"""

import os
import argparse
import logging
import glob
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "regularization_pareto")
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")
PARETO_DIR = os.path.join(OUTPUT_ROOT, "pareto_fronts")
ANALYSIS_DIR = os.path.join(OUTPUT_ROOT, "analysis")

# Ensure directories exist
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Sparsity levels to eval - Judah, you can change these if you want
SPARSITY_LEVELS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]

# Categories to group
CATEGORIES = [
    "Baseline", 
    "L1_Only", 
    "L2_Only", 
    "Spatial_Only", 
    "L1_Spatial", 
    "L2_Spatial", 
    "L1_L2", 
    "All",
    "2DSpatial"
]

# Impossible to differentiate these when graphed with the others
TOP_CATEGORIES = [
    "L1_Only", 
    "L1_Spatial", 
    "L1_L2", 
    "All",
]

def load_all_results():
    """Load all result files into a single DataFrame."""
    logger.info("Loading all results...")
    
    result_files = glob.glob(os.path.join(RESULTS_DIR, "*_sparsity.csv"))
    
    if not result_files:
        logger.error("No result files found!")
        return None
    
    all_dfs = []
    
    for file in result_files:
        try:
            df = pd.read_csv(file)
            
            # Extract model name and category
            model_name = os.path.basename(file).replace("_sparsity.csv", "")
            category = model_name.split("_")[0]
            
            if category not in CATEGORIES:
                # Try to infer the category from the hyperparameters, def a bit janky
                if "l1_scale_0p0" in model_name and "weight_decay_0p0" in model_name and "spatial_cost_scale_0p0" in model_name:
                    category = "Baseline"
                elif "l1_scale_0p0" not in model_name and "weight_decay_0p0" in model_name and "spatial_cost_scale_0p0" in model_name:
                    category = "L1_Only"
                elif "l1_scale_0p0" in model_name and "weight_decay_0p0" not in model_name and "spatial_cost_scale_0p0" in model_name:
                    category = "L2_Only"
                elif "l1_scale_0p0" in model_name and "weight_decay_0p0" in model_name and "spatial_cost_scale_0p0" not in model_name:
                    if "spatial_d_value" in model_name and "spatial_d_value_1p0" not in model_name:
                        category = "2DSpatial"
                    else:
                        category = "Spatial_Only"
                elif "l1_scale_0p0" not in model_name and "weight_decay_0p0" not in model_name and "spatial_cost_scale_0p0" in model_name:
                    category = "L1_L2"
                elif "l1_scale_0p0" not in model_name and "weight_decay_0p0" in model_name and "spatial_cost_scale_0p0" not in model_name:
                    category = "L1_Spatial"
                elif "l1_scale_0p0" in model_name and "weight_decay_0p0" not in model_name and "spatial_cost_scale_0p0" not in model_name:
                    category = "L2_Spatial"
                elif "l1_scale_0p0" not in model_name and "weight_decay_0p0" not in model_name and "spatial_cost_scale_0p0" not in model_name:
                    category = "All"
                else:
                    logger.warning(f"Could not determine category for {model_name}, skipping")
                    continue
            
            if "category" not in df.columns:
                df["category"] = category
            
            # Extract all parameters
            for param_name in ["l1_scale", "weight_decay", "spatial_cost_scale", "spatial_d_value"]:
                param_pattern = f"{param_name}_"
                
                if param_name not in df.columns:
                    # Find parameter value in model name
                    param_index = model_name.find(param_pattern)
                    if param_index >= 0:
                        # Extract value after parameter name until next underscore
                        param_str = model_name[param_index + len(param_pattern):]
                        param_end = param_str.find("_") if "_" in param_str else len(param_str)
                        param_value = param_str[:param_end].replace("p", ".")
                        
                        try:
                            param_value = float(param_value)
                        except ValueError:
                            param_value = 0.0  # Default value if parsing fails
                            
                        df[param_name] = param_value
                    else:
                        # Default values
                        if param_name == "spatial_d_value":
                            df[param_name] = 1.0
                        else:
                            df[param_name] = 0.0
            
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Error processing {file}: {str(e)}")
    
    if not all_dfs:
        logger.error("No valid result dataframes found!")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} rows from {len(all_dfs)} files")
    
    # Save the combined results
    combined_df.to_csv(os.path.join(ANALYSIS_DIR, "all_models_combined.csv"), index=False)
    
    return combined_df

def find_optimal_hyperparameters(df):
    """Find the optimal hyperparameters for each sparsity level."""
    logger.info("Finding optimal hyperparameters...")
    
    optimal_params = []
    
    # For each sparsity level
    for sparsity in SPARSITY_LEVELS:
        # Filter data for this sparsity
        sparsity_df = df[df["target_sparsity"] == sparsity]
        
        if len(sparsity_df) == 0:
            continue
        
        # Find the best model (lowest validation loss)
        best_model = sparsity_df.loc[sparsity_df["val_loss"].idxmin()]
        
        optimal_params.append({
            "sparsity": sparsity,
            "model_name": best_model["model_name"],
            "category": best_model["category"],
            "l1_scale": best_model["l1_scale"],
            "weight_decay": best_model["weight_decay"],
            "spatial_cost_scale": best_model["spatial_cost_scale"],
            "spatial_d_value": best_model["spatial_d_value"] if "spatial_d_value" in best_model else 1.0,
            "val_loss": best_model["val_loss"],
            "actual_sparsity": best_model["actual_sparsity"]
        })
    
    # Create DataFrame and save
    optimal_df = pd.DataFrame(optimal_params)
    optimal_df.to_csv(os.path.join(ANALYSIS_DIR, "optimal_hyperparameters.csv"), index=False)
    
    # Create summary table
    summary_table = optimal_df[["sparsity", "category", "l1_scale", "weight_decay", 
                               "spatial_cost_scale", "spatial_d_value", "val_loss", "actual_sparsity"]]
    
    # Save as CSV
    summary_table.to_csv(os.path.join(ANALYSIS_DIR, "optimal_hyperparameters_summary.csv"), index=False)
    
    logger.info("Optimal hyperparameters found and saved")
    
    return optimal_df

def plot_category_comparison(df):
    """Plot a comparison of all categories at each sparsity level."""
    logger.info("Generating category comparison plots...")
    
    # Color map for categories with specific colors for better distinction
    category_colors = {
        "Baseline": "#000000",      # Black
        "L1_Only": "#E41A1C",       # Red
        "L2_Only": "#377EB8",       # Blue
        "Spatial_Only": "#4DAF4A",  # Green
        "L1_L2": "#984EA3",         # Purple
        "L1_Spatial": "#FF7F00",    # Orange
        "L2_Spatial": "#FFFF33",    # Yellow
        "All": "#A65628",           # Brown
        "2DSpatial": "#F781BF"      # Pink
    }
    
    # Store best validation loss for each category at each sparsity
    best_losses = {}
    
    # For each sparsity level
    for sparsity in SPARSITY_LEVELS:
        best_losses[sparsity] = {}
        
        # For each category
        for category in CATEGORIES:
            # Get data for this category and sparsity
            cat_data = df[(df["category"] == category) & (df["target_sparsity"] == sparsity)]
            
            if len(cat_data) > 0:
                # Find best model (lowest validation loss)
                best_model = cat_data.loc[cat_data["val_loss"].idxmin()]
                best_losses[sparsity][category] = best_model["val_loss"]
    
    # Convert to DataFrame for easier plotting
    sparsity_values = []
    category_values = []
    loss_values = []
    
    for sparsity in SPARSITY_LEVELS:
        for category in best_losses[sparsity]:
            sparsity_values.append(sparsity)
            category_values.append(category)
            loss_values.append(best_losses[sparsity][category])
    
    comparison_df = pd.DataFrame({
        "Sparsity": sparsity_values,
        "Category": category_values,
        "Best Loss": loss_values
    })
    
    # Save to CSV
    comparison_df.to_csv(os.path.join(ANALYSIS_DIR, "category_comparison.csv"), index=False)
    
    # Create comparison plots - one with all categories, one with top performers only
    for plot_type, categories_to_plot, filename in [
        ("All Categories", CATEGORIES, "regularization_performance_comparison_all.png"),
        ("Top Performers", TOP_CATEGORIES, "regularization_performance_comparison_top.png")
    ]:
        plt.figure(figsize=(15, 10))
        
        # Define marker styles for better distinction
        markers = {
            "Baseline": "o",       # Circle
            "L1_Only": "s",        # Square
            "L2_Only": "^",        # Triangle up
            "Spatial_Only": "d",   # Diamond
            "L1_L2": "p",          # Pentagon
            "L1_Spatial": "*",     # Star
            "L2_Spatial": "h",     # Hexagon
            "All": "X",            # X
            "2DSpatial": "P"       # Plus
        }
        
        for category in categories_to_plot:
            if category not in comparison_df["Category"].values:
                continue
                
            cat_data = comparison_df[comparison_df["Category"] == category]
            
            if len(cat_data) > 1:
                plt.plot(cat_data["Sparsity"], cat_data["Best Loss"], 
                         marker=markers[category], 
                         linestyle='-', 
                         linewidth=2.5,
                         markersize=10,
                         label=category.replace("_", "+"),
                         color=category_colors[category])
            elif len(cat_data) == 1:
                plt.plot(cat_data["Sparsity"], cat_data["Best Loss"], 
                         marker=markers[category], 
                         markersize=10,
                         label=category.replace("_", "+"),
                         color=category_colors[category])
        
        plt.xlabel("Sparsity", fontsize=14)
        plt.ylabel("Validation Loss", fontsize=14)
        plt.title(f"Regularization Performance by {plot_type} and Sparsity", fontsize=16)
        plt.grid(True, alpha=0.3)
        
        # Create a legend with larger font
        plt.legend(fontsize=12, loc='best', framealpha=0.7)
        
        # Add annotations for key points if space allows
        for category in categories_to_plot:
            if category not in comparison_df["Category"].values:
                continue
                
            cat_data = comparison_df[comparison_df["Category"] == category]
            
            # Annotate the highest sparsity point for each category
            if len(cat_data) > 0:
                max_sparsity_idx = cat_data["Sparsity"].idxmax()
                max_sparsity_point = cat_data.loc[max_sparsity_idx]
                
                if max_sparsity_point["Sparsity"] >= 0.8:  # Only annotate high sparsity points
                    plt.annotate(
                        f"{category.replace('_', '+')}",
                        xy=(max_sparsity_point["Sparsity"], max_sparsity_point["Best Loss"]),
                        xytext=(5, 0),
                        textcoords="offset points",
                        fontsize=10,
                        alpha=0.8
                    )
        
        # x-axis  ticks and labels
        plt.xticks(SPARSITY_LEVELS, [f"{s:.0%}" for s in SPARSITY_LEVELS], fontsize=12)
        
        # light gray background grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save high-resolution figure
        plt.savefig(os.path.join(ANALYSIS_DIR, filename), 
                    dpi=300, bbox_inches="tight")
        
        plt.close()
    
    # Plot 2DSpatial results by D value
    if "2DSpatial" in df["category"].values:
        plt.figure(figsize=(15, 10))
        
        # Filter for 2DSpatial models
        spatial_df = df[df["category"] == "2DSpatial"]
        
        # Get unique D values
        d_values = sorted(spatial_df["spatial_d_value"].unique())
        
        # Create a colormap for D values
        d_cmap = plt.cm.get_cmap("viridis", len(d_values))
        d_colors = {d: d_cmap(i) for i, d in enumerate(d_values)}
        
        # For each D value
        for d in d_values:
            # Store best results for each sparsity
            d_sparsity = []
            d_loss = []
            
            # For each sparsity level
            for sparsity in SPARSITY_LEVELS:
                d_data = spatial_df[(spatial_df["spatial_d_value"] == d) & 
                                    (spatial_df["target_sparsity"] == sparsity)]
                
                if len(d_data) > 0:
                    # Find best model (lowest validation loss)
                    best_model = d_data.loc[d_data["val_loss"].idxmin()]
                    d_sparsity.append(sparsity)
                    d_loss.append(best_model["val_loss"])
            
            if len(d_sparsity) > 1:  # Only plot if we have multiple points
                plt.plot(d_sparsity, d_loss, 
                         marker='o', 
                         linestyle='-', 
                         linewidth=2.5,
                         markersize=10,
                         label=f"D={d}",
                         color=d_colors[d])
            elif len(d_sparsity) == 1:  # Just a single point
                plt.plot(d_sparsity, d_loss, 
                         marker='o', 
                         markersize=10,
                         label=f"D={d}",
                         color=d_colors[d])
        
        plt.xlabel("Sparsity", fontsize=14)
        plt.ylabel("Validation Loss", fontsize=14)
        plt.title("Effect of D Value on 2D Spatial Regularization", fontsize=16)
        plt.grid(True, alpha=0.3)
        
        # Create a legend with larger font
        plt.legend(fontsize=12, loc='best', framealpha=0.7)
        
        # Improve x-axis with more ticks and labels
        plt.xticks(SPARSITY_LEVELS, [f"{s:.0%}" for s in SPARSITY_LEVELS], fontsize=12)
        
        # Add a light gray background grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save high-resolution figure
        plt.savefig(os.path.join(ANALYSIS_DIR, "2d_spatial_d_values.png"), 
                    dpi=300, bbox_inches="tight")
        
        # Save a vector version for publications
        plt.savefig(os.path.join(ANALYSIS_DIR, "2d_spatial_d_values.pdf"), 
                    bbox_inches="tight")
        
        plt.close()
    
    logger.info("Category comparison plots generated")

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description='Analyze regularization experiment results')
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR,
                       help='Directory containing result files')
    parser.add_argument('--analysis-dir', type=str, default=ANALYSIS_DIR,
                       help='Directory to save analysis results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.analysis_dir, exist_ok=True)
    
    # Load all results
    df = load_all_results()
    
    if df is None:
        logger.error("Failed to load results, exiting")
        return
    
    # Generate plots and analysis
    plot_category_comparison(df)  # This creates the main validation loss vs sparsity plot
    optimal_df = find_optimal_hyperparameters(df)
    
    logger.info("Analysis completed successfully")

if __name__ == "__main__":
    main()