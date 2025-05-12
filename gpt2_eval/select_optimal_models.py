"""
Select optimal models based on sparsification results and desired criteria.

This script:
1. Analyzes the sparsification results to identify models with optimal trade-offs
2. Allows for filtering based on sparsity requirements, performance thresholds, and more
3. Creates group-based Pareto front visualization for different regularization techniques
"""

import os
import pandas as pd
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

# Configure argument parser
parser = argparse.ArgumentParser(description="Select optimal sparse models based on evaluation results")
parser.add_argument('--results_dir', type=str, default='gpt2_sparsity_results',
                    help='Directory with sparsification results')
parser.add_argument('--output_file', type=str, default='optimal_models.json',
                    help='Output file for model recommendations')
parser.add_argument('--min_sparsity', type=float, default=0.0,
                    help='Minimum sparsity level to consider (0.0-1.0)')
parser.add_argument('--max_performance_drop', type=float, default=0.1,
                    help='Maximum acceptable performance drop ratio compared to baseline (0.0-1.0)')
args = parser.parse_args()

def load_results(results_dir):
    """Load sparsification results from the directory."""
    combined_results = os.path.join(results_dir, 'all_gpt2_models_sparsity.csv')
    
    if os.path.exists(combined_results):
        return pd.read_csv(combined_results)
    
    # If no combined file, try to load from individual files
    all_data = []
    for file in os.listdir(results_dir):
        if file.endswith('_sparsity.csv') and not file.startswith('all_'):
            file_path = os.path.join(results_dir, file)
            df = pd.read_csv(file_path)
            all_data.append(df)
    
    if not all_data:
        print("No results found. Please run the sparsification script first.")
        return None
    
    return pd.concat(all_data, ignore_index=True)

def normalize_model_performance(df):
    """Normalize performance metrics relative to the baseline (0% sparsity) for each model."""
    normalized_df = df.copy()
    
    # Get baseline performance for each model
    baselines = {}
    for model in df['model_name'].unique():
        model_data = df[df['model_name'] == model]
        baseline = model_data[model_data['actual_sparsity'] < 0.05]
        
        if not baseline.empty:
            baselines[model] = {
                'loss': baseline['val_loss'].min()
            }
    
    # Create normalized columns
    normalized_df['norm_loss'] = 0.0
    normalized_df['relative_drop'] = 0.0
    
    # Apply normalization
    for idx, row in normalized_df.iterrows():
        model = row['model_name']
        if model in baselines:
            normalized_df.at[idx, 'norm_loss'] = row['val_loss'] / baselines[model]['loss']
            normalized_df.at[idx, 'relative_drop'] = normalized_df.at[idx, 'norm_loss'] - 1.0
    
    return normalized_df

def find_pareto_frontier(df, x_col, y_col, maximize_x=True, minimize_y=True):
    """
    Find the Pareto frontier points in the dataset.
    For model selection, typically:
    - x_col = 'actual_sparsity' (maximize_x=True)
    - y_col = 'val_loss' (minimize_y=True)
    """
    # Make a copy of the data
    df_copy = df.copy()
    
    # Convert to numpy arrays
    x = df_copy[x_col].values
    y = df_copy[y_col].values
    
    # Adjust sign based on maximize/minimize preferences
    x_sign = 1 if maximize_x else -1
    y_sign = -1 if minimize_y else 1
    
    # Find Pareto frontier
    pareto_indices = []
    for i in range(len(x)):
        dominated = False
        for j in range(len(x)):
            if i != j:
                if (x_sign * x[j] >= x_sign * x[i] and 
                    y_sign * y[j] >= y_sign * y[i] and
                    (x_sign * x[j] > x_sign * x[i] or 
                     y_sign * y[j] > y_sign * y[i])):
                    dominated = True
                    break
        if not dominated:
            pareto_indices.append(i)
    
    # Return the Pareto optimal points
    return df_copy.iloc[pareto_indices].sort_values(by=x_col, ascending=not maximize_x)

def create_group_pareto_visualization(df, output_dir):
    """
    Create a visualization of performance by group and sparsity level.
    This helps visualize the Pareto front for different regularization techniques.
    """
    if 'group' not in df.columns:
        print("No 'group' column found in the data. Skipping group visualization.")
        return
    
    # Create a plot to show performance by group
    plt.figure(figsize=(16, 10))
    
    # Convert sparsity to percentage points for x-axis
    sparsity_ticks = [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    sparsity_labels = [f"{s*100:.0f}%" for s in sparsity_ticks]
    
    # Get the unique groups
    groups = df['group'].unique()
    
    # Define consistent markers and colors for groups
    markers = {'L1': 's', 'Spatial': 'v', 'Combo': 'D', 'L1Only': '^', 'Other': 'o'}
    colors = {'L1': 'red', 'Spatial': 'green', 'Combo': 'blue', 'L1Only': 'purple', 'Other': 'gray'}
    
    # For each group, find the best model at each sparsity level
    for group in groups:
        group_df = df[df['group'] == group]
        
        # Skip if fewer than 3 data points for this group
        if len(group_df) < 3:
            continue
            
        # Get best performance at each sparsity level
        best_at_sparsity = {}
        for sparsity in sparsity_ticks:
            # Find models within a small range of this sparsity level
            tolerance = 0.025
            sparsity_models = group_df[
                (group_df['actual_sparsity'] >= sparsity - tolerance) & 
                (group_df['actual_sparsity'] <= sparsity + tolerance)
            ]
            
            if not sparsity_models.empty:
                # Get the model with minimum loss at this sparsity
                best = sparsity_models.loc[sparsity_models['val_loss'].idxmin()]
                best_at_sparsity[sparsity] = best
        
        # Plot the best models for this group across sparsity levels
        if best_at_sparsity:
            x_values = []
            y_values = []
            for sparsity, row in sorted(best_at_sparsity.items()):
                x_values.append(sparsity)
                y_values.append(row['val_loss'])
            
            plt.plot(x_values, y_values, '-', 
                     marker=markers.get(group, 'o'),
                     color=colors.get(group, 'black'), 
                     linewidth=2,
                     markersize=10,
                     label=group)
    
    plt.xlabel('Sparsity', fontsize=14)
    plt.ylabel('Validation Loss', fontsize=14)
    plt.title('Regularization Performance by Group and Sparsity', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(sparsity_ticks, sparsity_labels)
    plt.legend(fontsize=12)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'group_pareto_front.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Group Pareto front visualization saved to {output_dir}/group_pareto_front.png")

def select_optimal_models(df, min_sparsity=args.min_sparsity, max_drop=args.max_performance_drop):
    """
    Select optimal models based on sparsity and performance criteria.
    Returns recommendations for different use cases.
    """
    if df is None or df.empty:
        return None
    
    # Normalize performance relative to baseline
    normalized_df = normalize_model_performance(df)
    
    # Filter by minimum sparsity
    filtered_df = normalized_df[normalized_df['actual_sparsity'] >= min_sparsity]
    
    if filtered_df.empty:
        print(f"No models meet the minimum sparsity requirement of {min_sparsity*100:.0f}%")
        return None
    
    # Find models within acceptable performance drop
    acceptable_models = filtered_df[filtered_df['relative_drop'] <= max_drop]
    
    if acceptable_models.empty:
        print(f"No models meet both sparsity ({min_sparsity*100:.0f}%) and " 
              f"performance (max {max_drop*100:.0f}% drop) requirements")
        return None
    
    # Find Pareto frontier (optimal trade-offs)
    pareto_models = find_pareto_frontier(acceptable_models, 
                                         'actual_sparsity', 
                                         'val_loss', 
                                         maximize_x=True, 
                                         minimize_y=True)
    
    # Select specific recommendations
    recommendations = {}
    
    # Most balanced model (middle of Pareto frontier)
    if len(pareto_models) > 0:
        middle_idx = len(pareto_models) // 2
        balanced = pareto_models.iloc[middle_idx]
        recommendations['balanced'] = {
            'model_name': balanced['model_name'],
            'sparsity': balanced['actual_sparsity'],
            'val_loss': balanced['val_loss'],
            'relative_drop': balanced['relative_drop'],
            'group': balanced.get('group', 'Unknown') if 'group' in balanced else 'Unknown'
        }
    
    # Best performance model (within criteria)
    best_perf = acceptable_models.loc[acceptable_models['val_loss'].idxmin()]
    recommendations['best_performance'] = {
        'model_name': best_perf['model_name'],
        'sparsity': best_perf['actual_sparsity'],
        'val_loss': best_perf['val_loss'],
        'relative_drop': best_perf['relative_drop'],
        'group': best_perf.get('group', 'Unknown') if 'group' in best_perf else 'Unknown'
    }
    
    # Highest sparsity model (within criteria)
    highest_sparsity = acceptable_models.loc[acceptable_models['actual_sparsity'].idxmax()]
    recommendations['highest_sparsity'] = {
        'model_name': highest_sparsity['model_name'],
        'sparsity': highest_sparsity['actual_sparsity'],
        'val_loss': highest_sparsity['val_loss'],
        'relative_drop': highest_sparsity['relative_drop'],
        'group': highest_sparsity.get('group', 'Unknown') if 'group' in highest_sparsity else 'Unknown'
    }
    
    # Best models by group at specific sparsity levels
    if 'group' in acceptable_models.columns:
        recommendations['by_group'] = {}
        
        sparsity_targets = [0.25, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
        groups = acceptable_models['group'].unique()
        
        for group in groups:
            group_models = acceptable_models[acceptable_models['group'] == group]
            if not group_models.empty:
                recommendations['by_group'][group] = {}
                
                for target in sparsity_targets:
                    # Find models close to this sparsity level
                    tolerance = 0.05
                    candidates = group_models[
                        (group_models['actual_sparsity'] >= target - tolerance) & 
                        (group_models['actual_sparsity'] <= target + tolerance)
                    ]
                    
                    if not candidates.empty:
                        best = candidates.loc[candidates['val_loss'].idxmin()]
                        recommendations['by_group'][group][f"{target*100:.0f}%"] = {
                            'model_name': best['model_name'],
                            'sparsity': best['actual_sparsity'],
                            'val_loss': best['val_loss'],
                            'relative_drop': best['relative_drop']
                        }
    
    # Best models at specific sparsity levels (regardless of group)
    recommendations['sparsity_targets'] = {}
    
    sparsity_targets = [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    for target in sparsity_targets:
        # Find models close to this sparsity level
        tolerance = 0.05
        candidates = acceptable_models[
            (acceptable_models['actual_sparsity'] >= target - tolerance) & 
            (acceptable_models['actual_sparsity'] <= target + tolerance)
        ]
        
        if not candidates.empty:
            best = candidates.loc[candidates['val_loss'].idxmin()]
            recommendations['sparsity_targets'][f"{target*100:.0f}%"] = {
                'model_name': best['model_name'],
                'sparsity': best['actual_sparsity'],
                'val_loss': best['val_loss'],
                'relative_drop': best['relative_drop'],
                'group': best.get('group', 'Unknown') if 'group' in best else 'Unknown'
            }
    
    return recommendations, pareto_models