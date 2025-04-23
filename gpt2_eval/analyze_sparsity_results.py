"""
Analyze GPT2 model sparsification results and generate comparative visualizations
across different regularization techniques and sparsity levels.

This script:
1. Loads results from the gpt2_sparsity_results directory
2. Creates detailed visualizations comparing different regularization approaches
3. Identifies the best performing models at each sparsity level
4. Generates a summary report with key findings
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re

# Configure argument parser
parser = argparse.ArgumentParser(description="Analyze GPT2 model sparsification results")
parser.add_argument('--results_dir', type=str, default='gpt2_sparsity_results',
                    help='Directory with sparsification results')
parser.add_argument('--output_dir', type=str, default='sparsity_analysis',
                    help='Directory to save analysis results')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Function to extract model parameters from names
def parse_model_params(model_name):
    """Extract regularization parameters and model configuration from the name."""
    params = {
        'l1_scale': 0.0,
        'spatial_scale': 0.0,
        'weight_decay': 0.0,
        'spatial_mode': 'fixed'
    }
    
    # Extract L1 scale
    l1_match = re.search(r'l1_(\d+\.\d+e-\d+)', model_name)
    if l1_match:
        params['l1_scale'] = float(l1_match.group(1))
    
    # Extract spatial scale
    spatial_match = re.search(r'spatial_(\d+\.\d+e-\d+)', model_name)
    if spatial_match:
        params['spatial_scale'] = float(spatial_match.group(1))
    
    # Extract weight decay
    wd_match = re.search(r'wd_(\d+\.\d+e-\d+)', model_name)
    if wd_match:
        params['weight_decay'] = float(wd_match.group(1))
    
    # Determine regularization type and create a more readable name
    reg_type = 'Baseline'
    if params['l1_scale'] > 0 and params['spatial_scale'] > 0:
        reg_type = f"L1 ({params['l1_scale']:.0e}) + Spatial ({params['spatial_scale']:.0e})"
    elif params['l1_scale'] > 0:
        reg_type = f"L1 ({params['l1_scale']:.0e})"
    elif params['spatial_scale'] > 0:
        mode = 'learnable' if 'learnable' in model_name else 'swappable' if 'swappable' in model_name else 'fixed'
        reg_type = f"Spatial-{mode} ({params['spatial_scale']:.0e})"
    elif params['weight_decay'] > 0:
        reg_type = f"Weight Decay ({params['weight_decay']:.0e})"
    
    params['reg_type'] = reg_type
    
    return params

# Load all results
def load_all_results(results_dir=args.results_dir):
    """Load all individual model results and the combined results file."""
    all_models = []
    combined_file = os.path.join(results_dir, 'all_gpt2_models_sparsity.csv')
    
    if os.path.exists(combined_file):
        print(f"Loading combined results from {combined_file}")
        combined_df = pd.read_csv(combined_file)
        return combined_df
    
    # If combined file doesn't exist, load individual CSVs
    print("No combined results file found. Loading individual model results...")
    for file in os.listdir(results_dir):
        if file.endswith('_sparsity.csv') and not file.startswith('all_'):
            try:
                file_path = os.path.join(results_dir, file)
                df = pd.read_csv(file_path)
                all_models.append(df)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue
    
    if not all_models:
        print("No results found. Please run the sparsification script first.")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_models, ignore_index=True)
    return combined_df

# Add regularization type and other metadata to results
def enrich_results(df):
    """Add metadata like regularization type to the results dataframe."""
    if df is None or df.empty:
        return None
    
    # Add regularization info based on model name
    enriched = []
    for _, row in df.iterrows():
        model_name = row['model_name']
        params = parse_model_params(model_name)
        
        # Add parameters to row
        new_row = row.to_dict()
        new_row['reg_type'] = params['reg_type']
        
        # Create a more readable model name for plots
        l1_str = f"L1={params['l1_scale']:.0e}" if params['l1_scale'] > 0 else ""
        spatial_str = f"Sp={params['spatial_scale']:.0e}" if params['spatial_scale'] > 0 else ""
        wd_str = f"WD={params['weight_decay']:.0e}" if params['weight_decay'] > 0 else ""
        
        components = [s for s in [l1_str, spatial_str, wd_str] if s]
        if components:
            readable_name = " ".join(components)
        else:
            readable_name = "Baseline"
        
        if 'learnable' in model_name:
            readable_name += " (Learn)"
        elif 'swappable' in model_name:
            readable_name += " (Swap)"
            
        new_row['readable_name'] = readable_name
        enriched.append(new_row)
    
    return pd.DataFrame(enriched)


def generate_best_per_group_comparison(enriched_results, output_dir=args.output_dir):
    """
    Generate a comparison plot showing the best performing model from each group
    at each sparsity level.
    """
    if enriched_results is None or enriched_results.empty:
        print("No results available to generate comparison")
        return
    
    # Define the correct groups for your current project
    groups = {
        'Baseline': ['gpt2'],
        'L1': ['l1_'],
        'L1Only': ['l1only'],
        'Spatial': ['spatial']
    }
    
    # Define the standard sparsity levels to plot
    plot_sparsity_levels = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    
    # Prepare data for plotting
    plot_data = {}
    
    # For each group, find the best model at each sparsity level
    for group_name, prefixes in groups.items():
        plot_data[group_name] = {'sparsity': [], 'loss': []}
        
        # For each sparsity level
        for target_sparsity in plot_sparsity_levels:
            # Find models that match this group and have sparsity close to target
            tolerance = 0.025  # Adjust as needed
            
            group_models = pd.DataFrame()
            for prefix in prefixes:
                # Check if any model name starts with this prefix
                models_with_prefix = enriched_results[enriched_results['model_name'].str.contains(prefix, regex=False)]
                group_models = pd.concat([group_models, models_with_prefix])
            
            if group_models.empty:
                continue
                
            # Find models at this sparsity level
            sparsity_models = group_models[
                (group_models['actual_sparsity'] >= target_sparsity - tolerance) & 
                (group_models['actual_sparsity'] <= target_sparsity + tolerance)
            ]
            
            if sparsity_models.empty:
                continue
            
            # Find the best model (lowest loss)
            best_model = sparsity_models.loc[sparsity_models['val_loss'].idxmin()]
            
            # Add to plot data
            plot_data[group_name]['sparsity'].append(best_model['actual_sparsity'])
            plot_data[group_name]['loss'].append(best_model['val_loss'])
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Plot each group
    markers = ['o', 's', '^', 'v']
    colors = ['black', 'red', 'blue', 'green']
    
    for i, (group_name, data) in enumerate(plot_data.items()):
        if len(data['sparsity']) > 0:
            # Sort by sparsity for clean lines
            sparsity_values = np.array(data['sparsity'])
            loss_values = np.array(data['loss'])
            sorted_indices = np.argsort(sparsity_values)
            
            x = sparsity_values[sorted_indices]
            y = loss_values[sorted_indices]
            
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            plt.plot(x, y, '-' + marker, linewidth=2, markersize=8, 
                     label=group_name, color=color)
    
    # Customize the plot
    plt.xlabel('Sparsity', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Regularization Performance by Group and Sparsity', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Format x-axis as percentages
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    # Add specific sparsity ticks
    plt.xticks([0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9])
    
    # Save the plot
    output_path = os.path.join(output_dir, 'regularization_comparison_by_group.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated group comparison plot at {output_path}")


# Find best models at each sparsity level
def identify_best_models(df):
    """Identify the best performing models at each sparsity level."""
    if df is None or df.empty:
        return None
    
    # Define standard sparsity levels
    sparsity_levels = [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    results = []
    
    for target in sparsity_levels:
        # Find models with actual sparsity close to target
        tolerance = 0.05
        models = df[(df['actual_sparsity'] >= target - tolerance) & 
                    (df['actual_sparsity'] <= target + tolerance)]
        
        if models.empty:
            continue
        
        # Find best model (lowest loss)
        best_model = models.loc[models['val_loss'].idxmin()]
        
        # Check if required columns exist in the dataframe
        required_columns = ['val_loss', 'l1_scale', 'spatial_cost_scale', 'weight_decay']
        missing_columns = [col for col in required_columns if col not in best_model.index]
        
        # Create result dict with available columns
        result = {
            'sparsity_level': target,
            'actual_sparsity': best_model['actual_sparsity'],
            'best_model': best_model['readable_name'],
            'val_loss': best_model['val_loss']
        }
        
        # Add regularization parameters if they exist
        if 'l1_scale' in best_model:
            result['l1_scale'] = best_model['l1_scale']
        if 'spatial_cost_scale' in best_model:
            result['spatial_cost_scale'] = best_model['spatial_cost_scale'] 
        if 'weight_decay' in best_model:
            result['weight_decay'] = best_model['weight_decay']
        
        results.append(result)
    
    return pd.DataFrame(results)


def main():
    print(f"Loading results from {args.results_dir}...")
    
    # Load and process results
    results = load_all_results()
    if results is None or results.empty:
        print("No results found to analyze.")
        return
    
    # Enrich with metadata
    enriched_results = enrich_results(results)
    if enriched_results is None:
        print("Error processing results.")
        return
    
    print(f"Analyzing {enriched_results['model_name'].nunique()} models...")

    print("Generating comparison plot of best models by group...")
    generate_best_per_group_comparison(enriched_results)
    
    # Generate and save best models list
    best_models = identify_best_models(enriched_results)
    if best_models is not None:
        best_models_path = os.path.join(args.output_dir, 'best_models_by_sparsity.csv')
        best_models.to_csv(best_models_path, index=False)
        print(f"Saved best models list to {best_models_path}")
    
    print(f"Analysis complete! Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()