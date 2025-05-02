"""
Fix group assignments in the combined results file.
This script:
1. Loads the combined results CSV
2. Fixes missing or incorrect group assignments 
3. Saves the updated results back to the CSV
"""

import os
import pandas as pd
import argparse

# Configure argument parser
parser = argparse.ArgumentParser(description="Fix group assignments in model results")
parser.add_argument('--results_dir', type=str, default='gpt2_sparsity_results',
                    help='Directory with sparsification results')
args = parser.parse_args()

def main():
    # Load the combined results file
    combined_file = os.path.join(args.results_dir, 'all_gpt2_models_sparsity.csv')
    if not os.path.exists(combined_file):
        print(f"Error: Combined results file not found at {combined_file}")
        return
    
    print(f"Loading results from {combined_file}...")
    df = pd.read_csv(combined_file)
    
    # Backup the original file
    backup_file = os.path.join(args.results_dir, 'all_gpt2_models_sparsity.backup.csv')
    df.to_csv(backup_file, index=False)
    print(f"Original file backed up to {backup_file}")
    
    # Check current group assignments
    print("\nCurrent group assignments:")
    if 'group' in df.columns:
        for group, count in df.groupby('group')['model_name'].nunique().items():
            group_str = str(group) if not pd.isna(group) else "NaN"
            print(f"  {group_str}: {count} models")
        
        # Count NaN values
        nan_count = df['group'].isna().sum()
        if nan_count > 0:
            print(f"  NaN values: {nan_count} entries")
    else:
        print("  No 'group' column found in the data")
        # Add group column
        df['group'] = None
    
    # Make a copy to track changes
    df_original = df.copy()
    
    # Fix group assignments based on model name patterns
    print("\nAssigning groups based on model name patterns...")
    
    # Assign Spatial_D group to models with 'd_spatial' in the name
    spatial_d_mask = df['model_name'].str.contains('d_spatial', case=False)
    df.loc[spatial_d_mask, 'group'] = 'Spatial_D'
    
    # Assign Spatial group to models with 'spatial' in the name, but not already assigned to Spatial_D
    spatial_mask = df['model_name'].str.contains('spatial', case=False) & ~spatial_d_mask
    df.loc[spatial_mask, 'group'] = 'Spatial'
    
    # Assign L1 group to models with 'l1' in the name
    l1_mask = df['model_name'].str.contains('l1', case=False)
    df.loc[l1_mask, 'group'] = 'L1'
    
    # For all other models, set group to 'Other' if not already assigned
    other_mask = ~(spatial_mask | l1_mask | spatial_d_mask)
    df.loc[other_mask & df['group'].isna(), 'group'] = 'Other'
    
    # Count changes
    changes = (df['group'] != df_original['group']).sum()
    print(f"Updated {changes} group assignments")
    
    # Print updated group assignments
    print("\nUpdated group assignments:")
    for group, count in df.groupby('group')['model_name'].nunique().items():
        print(f"  {group}: {count} models")
    
    # List models in each group
    print("\nModels by group:")
    for group in df['group'].unique():
        models = df[df['group'] == group]['model_name'].unique()
        print(f"\n  {group} ({len(models)} models):")
        for model in models:
            print(f"    - {model}")
    
    # Save the updated file
    df.to_csv(combined_file, index=False)
    print(f"\nUpdated results saved to {combined_file}")

if __name__ == '__main__':
    main()