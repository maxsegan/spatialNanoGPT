"""
Create a Pareto front visualization for different groups of models.
This script:
1. Loads model results from the combined CSV file
2. Groups models by their regularization technique 
3. Plots performance at different sparsity levels
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Configure argument parser
parser = argparse.ArgumentParser(description="Create Pareto front visualization for model groups")
parser.add_argument('--results_dir', type=str, default='gpt2_sparsity_results',
                    help='Directory with sparsification results')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory for visualization (defaults to results_dir)')
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.results_dir

def load_results(results_dir):
    """Load the combined results from CSV."""
    combined_file = os.path.join(results_dir, 'all_gpt2_models_sparsity.csv')
    if not os.path.exists(combined_file):
        print(f"Error: Combined results file not found at {combined_file}")
        return None
    
    return pd.read_csv(combined_file)

def create_pareto_front_visualization(df, output_dir):
    """Create the Pareto front visualization by group."""
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Check for group column
    if 'group' not in df.columns:
        print("No 'group' column found in results. Cannot create visualization without explicit group assignments.")
        return
    
    # Print count of models in each group
    print("\nModels by group:")
    for group, count in df.groupby('group')['model_name'].nunique().items():
        print(f"  {group}: {count} models")
    
    # Create a figure
    plt.figure(figsize=(16, 10))
    
    # Define fixed sparsity levels we want to show
    sparsity_levels = [0.0, 0.25, 0.5, 0.6, 0.7, 0.8]
    
    # Get unique groups
    groups = df['group'].unique()
    
    # Define markers and colors for each group
    markers = {'L1': 's', 'Spatial': 'v', 'Combo': 'D', 'L1Only': '^', 'Other': 'o'}
    colors = {'L1': 'red', 'Spatial': 'green', 'Combo': 'blue', 'L1Only': 'purple', 'Other': 'gray'}
    
    # For each group, find the best model at each sparsity level
    for group in groups:
        group_df = df[df['group'] == group]
        
        # Debug output
        unique_models = group_df['model_name'].unique()
        print(f"\nGroup '{group}' contains {len(unique_models)} models:")
        for model in unique_models:
            print(f"  - {model}")
        
        # Get best performance at each sparsity level
        x_values = []
        y_values = []
        
        for sparsity in sparsity_levels:
            # Find models within a range of this sparsity level
            tolerance = 0.025
            models_at_sparsity = group_df[
                (group_df['actual_sparsity'] >= sparsity - tolerance) & 
                (group_df['actual_sparsity'] <= sparsity + tolerance)
            ]
            
            if not models_at_sparsity.empty:
                # Get the model with lowest validation loss
                best_model = models_at_sparsity.loc[models_at_sparsity['val_loss'].idxmin()]
                x_values.append(sparsity)
                y_values.append(best_model['val_loss'])
                print(f"  Sparsity {sparsity*100:.1f}%: Best model is {best_model['model_name']} with loss {best_model['val_loss']:.4f}")
        
        # Skip if we don't have enough data points (need at least 2 for a line)
        if len(x_values) < 2:
            print(f"Not enough data points for group: {group} (only {len(x_values)} points found)")
            continue

        # Plot this group's Pareto front
        plt.plot(
            x_values, 
            y_values, 
            marker=markers.get(group, 'o'),
            color=colors.get(group, 'black'),
            linewidth=2,
            markersize=10,
            label=group
        )
    
    # Customize the plot
    plt.xlabel('Sparsity', fontsize=14)
    plt.ylabel('Validation Loss', fontsize=14)
    plt.title('Regularization Performance by Group and Sparsity', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale("log")
    
    # Set the x-axis ticks to show percentages
    plt.xticks(sparsity_levels, [f"{s*100:.0f}%" for s in sparsity_levels])
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Save the visualization
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'regularization_pareto_front.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPareto front visualization saved to {output_file}")

def main():
    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)
    
    if results is None or results.empty:
        print("No results found to visualize.")
        return
    
    print(f"Found data for {results['model_name'].nunique()} models.")
    
    # Create visualization
    print("Creating Pareto front visualization...")
    create_pareto_front_visualization(results, args.output_dir)

if __name__ == '__main__':
    main()