# Run the full GPT2 model sparsification evaluation pipeline

# Set default parameters
$REPO_ID = "GitWyd/SNN"
$BATCH_SIZE = 4
$BLOCK_SIZE = 1024
$EVAL_SUBSET = 200
$MIN_SPARSITY = 0.0
$MAX_PERF_DROP = 0.5
$VISUALIZE = $true
$FORCE_EVAL = $false

# Parse command line arguments
for ($i = 0; $i -lt $args.Count; $i++) {
    switch ($args[$i]) {
        "--batch_size" {
            $BATCH_SIZE = $args[++$i]
        }
        "--block_size" {
            $BLOCK_SIZE = $args[++$i]
        }
        "--eval_subset" {
            $EVAL_SUBSET = $args[++$i]
        }
        "--min_sparsity" {
            $MIN_SPARSITY = $args[++$i]
        }
        "--max_perf_drop" {
            $MAX_PERF_DROP = $args[++$i]
        }
        "--no_visualize" {
            $VISUALIZE = $false
        }
        "--force_reevaluate" {
            $FORCE_EVAL = $true
        }
        default {
            Write-Host "Unknown parameter: $($args[$i])"
            exit 1
        }
    }
}

# Create directories
if (-not (Test-Path -Path "gpt2_sparsity_results")) {
    New-Item -Path "gpt2_sparsity_results" -ItemType Directory | Out-Null
}
if (-not (Test-Path -Path "sparsity_analysis")) {
    New-Item -Path "sparsity_analysis" -ItemType Directory | Out-Null
}

Write-Host "Starting GPT2 model sparsification pipeline..."

# Step 1: Run the sparsification and evaluation script
Write-Host "Step 1: Running model sparsification and evaluation..."
$SparsifyArgs = @(
    "--repo_id", $REPO_ID,
    "--batch_size", $BATCH_SIZE,
    "--block_size", $BLOCK_SIZE,
    "--eval_subset_size", $EVAL_SUBSET,
    "--results_dir", "gpt2_sparsity_results"
)

if ($FORCE_EVAL) {
    $SparsifyArgs += "--force_reevaluate"
}

if ($FORCE_EVAL) {
    Write-Host "Force reevaluation enabled, clearing previous results..."
    Remove-Item -Path "gpt2_sparsity_results/*.csv" -Force
    Remove-Item -Path "gpt2_sparsity_results/all_gpt2_models_sparsity.csv" -Force
}

python sparsify_and_evaluate_gpt2.py $SparsifyArgs

# Check if the previous step succeeded
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in sparsification and evaluation step. Exiting."
    exit 1
}

# Step 2: Run the analysis script
Write-Host "Step 2: Analyzing results..."
$AnalysisArgs = @(
    "--results_dir", "gpt2_sparsity_results",
    "--output_dir", "sparsity_analysis"
)

python analyze_sparsity_results.py $AnalysisArgs

# Check if the previous step succeeded
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in analysis step. Exiting."
    exit 1
}

# Step 3: Select optimal models
Write-Host "Step 3: Selecting optimal models..."
$SelectionArgs = @(
    "--results_dir", "gpt2_sparsity_results",
    "--output_file", "sparsity_analysis/optimal_models.json",
    "--min_sparsity", $MIN_SPARSITY,
    "--max_performance_drop", $MAX_PERF_DROP
)
if ($VISUALIZE) {
    $SelectionArgs += "--visualize"
}

python select_optimal_models.py $SelectionArgs

# Check if the previous step succeeded
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in model selection step. Exiting."
    exit 1
}

Write-Host "Pipeline completed successfully!"
Write-Host "Results are available in:"
Write-Host "- gpt2_sparsity_results/: Raw sparsification results"
Write-Host "- sparsity_analysis/: Analysis and visualizations"
Write-Host "- sparsity_analysis/optimal_models.json: Recommended model selections"