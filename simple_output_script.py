"""
Simple script to output sparsity evaluation results in an easy-to-copy format.
Just run this on the CSV results file.
"""

import pandas as pd
import sys
import argparse

def print_copyable_results(csv_file='sparsity_results.csv'):
    """
    Read the CSV results file and print values in a simple copyable format.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Clean up column names to be more readable
    columns = df.columns.tolist()
    simplified_columns = []
    for col in columns:
        if col == 'Sparsity':
            simplified_columns.append('Sparsity %')
        else:
            # Extract the model name from path if needed
            model_name = col.split('/')[-1] if '/' in col else col
            # Further simplify model names
            model_name = (model_name
                         .replace('gpt2_l2_1e-1_100k', 'L2 1e-1')
                         .replace('gpt2-spatial_loss5_100k', 'Spatial 5')
                         .replace('gpt2-spatial_loss1_100k', 'Spatial 1'))
            simplified_columns.append(model_name)
    
    # Rename columns
    output_df = df.copy()
    output_df.columns = simplified_columns
    
    # Clean up sparsity column if it has % signs
    if 'Sparsity %' in output_df.columns:
        if '%' in str(output_df['Sparsity %'].iloc[0]):
            output_df['Sparsity %'] = output_df['Sparsity %'].str.replace('%', '')
    
    # Print as formatted table
    print("\n=== COPYABLE RESULTS TABLE ===\n")
    
    # Print header
    header = " ".join(f"{col:<12}" for col in output_df.columns)
    print(header)
    print("-" * len(header))
    
    # Print data rows
    for _, row in output_df.iterrows():
        formatted_row = " ".join(f"{str(val):<12}" for val in row)
        print(formatted_row)
    
    # Print in TSV format for easy spreadsheet pasting
    print("\n=== TSV FORMAT (FOR SPREADSHEETS) ===\n")
    print("\t".join(output_df.columns))
    for _, row in output_df.iterrows():
        print("\t".join(str(val) for val in row))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print sparsity results in copyable format')
    parser.add_argument('--csv', default='sparsity_results.csv', help='Path to CSV results file')
    
    args = parser.parse_args()
    print_copyable_results(args.csv)
