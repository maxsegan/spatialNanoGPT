#!/bin/bash

# Run the GPT2 model sparsification evaluation pipeline

# Set default parameters
BATCH_SIZE=4
BLOCK_SIZE=1024
MIN_SPARSITY=0.0
MAX_PERF_DROP=0.5
FORCE_EVAL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --block_size)
            BLOCK_SIZE="$2"
            shift 2
            ;;
        --eval_subset)
            EVAL_SUBSET="$2"
            shift 2
            ;;
        --min_sparsity)
            MIN_SPARSITY="$2"
            shift 2
            ;;
        --max_perf_drop)
            MAX_PERF_DROP="$2"
            shift 2
            ;;
        --force_reevaluate)
            FORCE_EVAL=true
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Create directories
if [ ! -d "gpt2_sparsity_results" ]; then
    mkdir -p "gpt2_sparsity_results"
fi
if [ ! -d "sparsity_analysis" ]; then
    mkdir -p "sparsity_analysis"
fi

echo "Starting GPT2 model sparsification pipeline..."

# Step 1: Run the updated sparsification and evaluation script
echo "Step 1: Running model sparsification and evaluation..."
SPARSIFY_ARGS=(
    "--batch_size" "$BATCH_SIZE"
    "--block_size" "$BLOCK_SIZE"
    "--results_dir" "gpt2_sparsity_results"
)

if [ "$FORCE_EVAL" = true ]; then
    SPARSIFY_ARGS+=("--force_reevaluate")
    echo "Force reevaluation enabled, clearing previous results..."
    rm -f gpt2_sparsity_results/*.csv
    rm -f gpt2_sparsity_results/all_gpt2_models_sparsity.csv
fi

python sparsify_and_evaluate_gpt2.py "${SPARSIFY_ARGS[@]}"

# Check if the previous step succeeded
if [ $? -ne 0 ]; then
    echo "Error in sparsification and evaluation step. Exiting."
    exit 1
fi

# Step 2: Create specialized Pareto front visualization
echo "Step 2: Creating specialized Pareto front visualization..."
PARETO_ARGS=(
    "--results_dir" "gpt2_sparsity_results"
    "--output_dir" "sparsity_analysis"
)

python create_pareto_front.py "${PARETO_ARGS[@]}"

# Check if the previous step succeeded
if [ $? -ne 0 ]; then
    echo "Error in Pareto front visualization step. Exiting."
    exit 1
fi

# Step 3: Select optimal models
echo "Step 3: Selecting optimal models..."
SELECTION_ARGS=(
    "--results_dir" "gpt2_sparsity_results"
    "--output_file" "sparsity_analysis/optimal_models.json"
    "--min_sparsity" "$MIN_SPARSITY"
    "--max_performance_drop" "$MAX_PERF_DROP"
)

python select_optimal_models.py "${SELECTION_ARGS[@]}"

# Check if the previous step succeeded
if [ $? -ne 0 ]; then
    echo "Error in model selection step. Exiting."
    exit 1
fi

echo "Pipeline completed successfully!"
echo "Results are available in:"
echo "- gpt2_sparsity_results/: Raw sparsification results"
echo "- sparsity_analysis/: Analysis and visualizations"
echo "- sparsity_analysis/optimal_models.json: Recommended model selections"
echo "- sparsity_analysis/regularization_pareto_front.png: Specialized Pareto front visualization"