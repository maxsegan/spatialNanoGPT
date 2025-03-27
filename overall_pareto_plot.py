import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Pareto front analysis for sparsity results")
    parser.add_argument('--experiments_dir', type=str, default='regularization_experiments',
                        help='Directory containing all experiment results')
    parser.add_argument('--results_dir', type=str, default='sparsity_results',
                        help='Subdirectory containing sparsity results')
    parser.add_argument('--output_dir', type=str, default='pareto_analysis',
                        help='Directory to save output files')
    return parser.parse_args()

def identify_regularization_type(row):
    """Identify which regularization type was used based on hyperparameters."""
    # Extract regularization parameters
    l1_scale = row.get('l1_scale', 0.0)
    weight_decay = row.get('weight_decay', 0.0)
    spatial_cost_scale = row.get('spatial_cost_scale', 0.0)
    spatial_mode = row.get('spatial_mode', 'fixed')
    
    # Determine type
    if spatial_cost_scale > 0:
        if spatial_mode == 'learnable':
            return "Spatial Learn"
        elif spatial_mode == 'swappable':
            return "Spatial Swappable"
        else:
            return "Spatial Fixed"
    elif l1_scale > 0 and weight_decay > 0:
        return "L1+L2"
    elif l1_scale > 0:
        return "L1 only"
    elif weight_decay > 0:
        return "L2"
    else:
        return "No Regularization"

def get_pareto_front(df):
    """
    Find the Pareto optimal points (minimizing loss while maximizing sparsity).
    Returns DataFrame with Pareto optimal points.
    """
    # Sort by sparsity (ascending) and loss (ascending)
    df_sorted = df.sort_values(['actual_sparsity', 'val_loss'], ascending=[True, True])
    
    # Start with the first point (lowest sparsity)
    pareto_indices = [0]
    current_best_loss = df_sorted.iloc[0]['val_loss']
    
    # Find points that have better loss than previous points
    for i in range(1, len(df_sorted)):
        if df_sorted.iloc[i]['val_loss'] < current_best_loss:
            pareto_indices.append(i)
            current_best_loss = df_sorted.iloc[i]['val_loss']
    
    return df_sorted.iloc[pareto_indices].copy()

def main():
    args = parse_arguments()
    
    # Create output directory
    full_output_dir = os.path.join(args.experiments_dir, args.output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Path to the sparsity results
    results_path = os.path.join(args.experiments_dir, args.results_dir)
    
    # Find all CSV files (excluding the combined one)
    all_csv_files = glob.glob(os.path.join(results_path, "*_sparsity.csv"))
    
    if not all_csv_files:
        print(f"No sparsity CSV files found in {results_path}")
        return
    
    print(f"Found {len(all_csv_files)} model result files")
    
    # Read and combine all CSV files
    all_data = []
    
    for csv_file in all_csv_files:
        try:
            model_df = pd.read_csv(csv_file)
            model_name = os.path.basename(csv_file).replace("_sparsity.csv", "")
            
            # Ensure model name is included in the dataframe
            if 'model_name' not in model_df.columns:
                model_df['model_name'] = model_name
                
            all_data.append(model_df)
        except Exception as e:
            print(f"Error reading {csv_file}: {str(e)}")
    
    # Combine all data
    if not all_data:
        print("No valid data found in CSV files")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Add regularization type
    combined_df['reg_type'] = combined_df.apply(identify_regularization_type, axis=1)
    
    # Group by sparsity level and regularization type
    sparsity_levels = sorted(combined_df['target_sparsity'].unique())
    reg_types = combined_df['reg_type'].unique()
    
    print(f"Found {len(reg_types)} regularization types: {', '.join(reg_types)}")
    print(f"Analyzing {len(sparsity_levels)} sparsity levels from {min(sparsity_levels):.2f} to {max(sparsity_levels):.2f}")
    
    # Create a dataframe to store the best model for each reg type at each sparsity level
    best_by_group = []
    
    for reg_type in reg_types:
        type_df = combined_df[combined_df['reg_type'] == reg_type]
        
        # Group by model and find the best loss at each sparsity level
        models = type_df['model_name'].unique()
        
        for model in models:
            model_df = type_df[type_df['model_name'] == model]
            
            for sparsity in sparsity_levels:
                sparsity_df = model_df[model_df['target_sparsity'] == sparsity]
                
                if not sparsity_df.empty:
                    # Find the best (lowest loss) for this model at this sparsity
                    best_row = sparsity_df.loc[sparsity_df['val_loss'].idxmin()].to_dict()
                    best_by_group.append(best_row)
    
    # Convert to DataFrame
    best_df = pd.DataFrame(best_by_group)
    
    # Compute Pareto fronts for each regularization type
    pareto_fronts = {}
    for reg_type in reg_types:
        type_df = best_df[best_df['reg_type'] == reg_type]
        if len(type_df) > 0:
            pareto_fronts[reg_type] = get_pareto_front(type_df)
    
    # Create a combined Pareto front dataframe
    all_pareto_df = pd.concat([df for df in pareto_fronts.values()], ignore_index=True)
    
    # Save each Pareto front to CSV
    for reg_type, front_df in pareto_fronts.items():
        # Remove spaces from reg_type for filename
        filename = f"pareto_front_{reg_type.replace(' ', '_').lower()}.csv"
        filepath = os.path.join(full_output_dir, filename)
        front_df.to_csv(filepath, index=False)
        print(f"Saved {len(front_df)} Pareto optimal points for {reg_type} to {filepath}")
    
    # Save combined Pareto fronts
    all_pareto_filepath = os.path.join(full_output_dir, "all_pareto_fronts.csv")
    all_pareto_df.to_csv(all_pareto_filepath, index=False)
    
    # Generate individual Pareto front plots
    plt.figure(figsize=(12, 8))
    
    for reg_type, front_df in pareto_fronts.items():
        if not front_df.empty:
            plt.plot(front_df['actual_sparsity'], front_df['val_loss'], 'o-', label=reg_type)
    
    plt.xlabel('Sparsity (Fraction of Zeros)', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Pareto Fronts by Regularization Type', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Save the plot
    plt.savefig(os.path.join(full_output_dir, 'pareto_fronts_combined.png'), dpi=300, bbox_inches='tight')
    
    # Generate individual plots for each regularization type
    for reg_type, front_df in pareto_fronts.items():
        if not front_df.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(front_df['actual_sparsity'], front_df['val_loss'], 'o-', 
                     label=f"{reg_type} Pareto Front", linewidth=2)
            
            # Add model names as annotations
            for i, row in front_df.iterrows():
                plt.annotate(row['model_name'], 
                             (row['actual_sparsity'], row['val_loss']),
                             textcoords="offset points",
                             xytext=(0,10),
                             ha='center')
            
            plt.xlabel('Sparsity (Fraction of Zeros)', fontsize=12)
            plt.ylabel('Validation Loss', fontsize=12)
            plt.title(f'Pareto Front for {reg_type} Regularization', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            
            # Save the plot
            plt_filename = f"pareto_front_{reg_type.replace(' ', '_').lower()}.png"
            plt.savefig(os.path.join(full_output_dir, plt_filename), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create a more detailed hyperparameter information table
    hyperparam_info = []
    
    for reg_type, front_df in pareto_fronts.items():
        for _, row in front_df.iterrows():
            # Extract hyperparameters
            info = {
                'reg_type': reg_type,
                'model_name': row['model_name'],
                'sparsity': row['actual_sparsity'],
                'val_loss': row['val_loss'],
                'l1_scale': row.get('l1_scale', 0),
                'weight_decay': row.get('weight_decay', 0),
                'spatial_cost_scale': row.get('spatial_cost_scale', 0),
                'spatial_mode': row.get('spatial_mode', 'fixed') if row.get('spatial_cost_scale', 0) > 0 else 'N/A'
            }
            hyperparam_info.append(info)
    
    # Create and save the hyperparameter information
    hyperparam_df = pd.DataFrame(hyperparam_info)
    hyperparam_filepath = os.path.join(full_output_dir, "pareto_hyperparameters.csv")
    hyperparam_df.to_csv(hyperparam_filepath, index=False)
    
    print(f"\nAnalysis complete! Results saved to {full_output_dir}")
    print(f"Generated {len(pareto_fronts) + 1} Pareto front plots and {len(pareto_fronts) + 2} CSV files")

if __name__ == "__main__":
    main()