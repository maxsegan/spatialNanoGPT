# PowerShell script to run the incremental sparsification and evaluation workflow
# This script takes the trained models and evaluates only new models, reusing existing results
# Now includes evaluation at higher sparsity levels (85%, 90%, 95%)

# Configuration
$EXPERIMENTS_DIR = "regularization_experiments"
$RESULTS_DIR = "sparsity_results"
$DATA_DIR = "data/shakespeare_char"  # Adjust based on your dataset

# Create the sparsity results directory if it doesn't exist
if (-not (Test-Path "$EXPERIMENTS_DIR\$RESULTS_DIR")) {
    New-Item -ItemType Directory -Path "$EXPERIMENTS_DIR\$RESULTS_DIR" | Out-Null
}

Write-Host "Starting incremental sparsity evaluation workflow with higher sparsity levels (up to 95%)..." -ForegroundColor Green
Write-Host "Experiments directory: $EXPERIMENTS_DIR" -ForegroundColor Cyan
Write-Host "Results will be saved to: $EXPERIMENTS_DIR\$RESULTS_DIR" -ForegroundColor Cyan

# Step 1: Run the incremental sparsification script
Write-Host "`nSTEP 1: Identifying new models and evaluating at different sparsity levels..." -ForegroundColor Yellow
python sparsify_and_evaluate_shakespeare.py `
    --experiments_dir=$EXPERIMENTS_DIR `
    --data_dir=$DATA_DIR `
    --results_dir=$RESULTS_DIR

if (-not $?) {
    Write-Host "Error running incremental sparsification script. Check the error message above." -ForegroundColor Red
    $continue = Read-Host "Do you want to continue to the comparison step anyway? (y/n)"
    if ($continue -ne "y") {
        exit 1
    }
}

# Check if any results were generated
$resultsExist = Test-Path "$EXPERIMENTS_DIR\$RESULTS_DIR\all_models_sparsity.csv"

if (-not $resultsExist) {
    Write-Host "No combined results file found. Skipping comparison step." -ForegroundColor Yellow
} else {
    # Step 2: Run the comparison script
    Write-Host "`nSTEP 2: Generating comprehensive comparisons across all models..." -ForegroundColor Yellow
    
    # Create comparisons directory if it doesn't exist
    $COMPARISONS_DIR = "$EXPERIMENTS_DIR\$RESULTS_DIR\comparisons"
    if (-not (Test-Path $COMPARISONS_DIR)) {
        New-Item -ItemType Directory -Path $COMPARISONS_DIR | Out-Null
    }
    
    python compare_sparsity_models_incremental.py `
        --results_csv="$EXPERIMENTS_DIR\$RESULTS_DIR\all_models_sparsity.csv" `
        --output_dir=$COMPARISONS_DIR

    if (-not $?) {
        Write-Host "Error running comparison script." -ForegroundColor Red
    } else {
        Write-Host "Comparison analysis completed successfully." -ForegroundColor Green
    }
}

# Check what results were generated
$resultsFiles = @(Get-ChildItem -Path "$EXPERIMENTS_DIR\$RESULTS_DIR" -Filter "*.csv" -ErrorAction SilentlyContinue)
$plotFiles = @(Get-ChildItem -Path "$EXPERIMENTS_DIR\$RESULTS_DIR" -Filter "*.png" -ErrorAction SilentlyContinue)
$comparisonFiles = @(Get-ChildItem -Path "$COMPARISONS_DIR" -Filter "*.*" -ErrorAction SilentlyContinue)

Write-Host "`nGenerated Files Summary:" -ForegroundColor Green

if ($resultsFiles.Count -eq 0) {
    Write-Host "  No CSV result files were generated." -ForegroundColor Yellow
} else {
    Write-Host "  CSV Results: $($resultsFiles.Count) files" -ForegroundColor Cyan
    $largestFiles = $resultsFiles | Sort-Object -Property Length -Descending | Select-Object -First 5
    foreach ($file in $largestFiles) {
        $sizeMB = [math]::Round($file.Length / 1MB, 2)
        if ($sizeMB -lt 0.01) {
            $sizeKB = [math]::Round($file.Length / 1KB, 2)
            Write-Host "    - $($file.Name) ($sizeKB KB)" -ForegroundColor Cyan
        } else {
            Write-Host "    - $($file.Name) ($sizeMB MB)" -ForegroundColor Cyan
        }
    }
    if ($resultsFiles.Count -gt 5) {
        Write-Host "    - ... and $($resultsFiles.Count - 5) more files" -ForegroundColor Cyan
    }
}

if ($plotFiles.Count -eq 0) {
    Write-Host "  No plot files were generated in the main results directory." -ForegroundColor Yellow
} else {
    Write-Host "  Plot Files: $($plotFiles.Count) files" -ForegroundColor Cyan
}

if ($comparisonFiles.Count -eq 0) {
    Write-Host "  No comparison files were generated." -ForegroundColor Yellow
} else {
    $comparisonCategories = @{
        "Plots" = @(Get-ChildItem -Path "$COMPARISONS_DIR" -Filter "*.png" -ErrorAction SilentlyContinue)
        "Data" = @(Get-ChildItem -Path "$COMPARISONS_DIR" -Filter "*.csv" -ErrorAction SilentlyContinue)
        "Text" = @(Get-ChildItem -Path "$COMPARISONS_DIR" -Filter "*.txt" -ErrorAction SilentlyContinue)
    }
    
    Write-Host "  Comparison Files: $($comparisonFiles.Count) files generated:" -ForegroundColor Cyan
    foreach ($category in $comparisonCategories.Keys) {
        $files = $comparisonCategories[$category]
        if ($files.Count -gt 0) {
            Write-Host "    - $category ($($files.Count) files)" -ForegroundColor Cyan
        }
    }
}

# Check if we have the best models report
$summaryFile = "$COMPARISONS_DIR\summary_findings.txt"
if (Test-Path $summaryFile) {
    Write-Host "`nKey Findings from Evaluation:" -ForegroundColor Green
    # Extract and display a few key stats from the summary file
    $summaryContent = Get-Content $summaryFile -Raw
    
    # Extract best model info using regex
    $bestModelInfo = $summaryContent -match "Best Overall Model: (.+)\r?\nRegularization Type: (.+)\r?\nSparsity: (.+)\r?\nValidation Loss: (.+)\r?\n"
    if ($bestModelInfo) {
        $bestModel = $matches[1]
        $bestRegType = $matches[2]
        $bestSparsity = $matches[3]
        $bestLoss = $matches[4]
        
        Write-Host "  Best Overall Model: $bestModel" -ForegroundColor Green
        Write-Host "    Type: $bestRegType, Sparsity: $bestSparsity, Loss: $bestLoss" -ForegroundColor Green
    }
    
    Write-Host "  For complete findings, see: $summaryFile" -ForegroundColor Cyan
}

# Open the results directory for convenience
if (Test-Path "$EXPERIMENTS_DIR\$RESULTS_DIR") {
    Write-Host "`nOpening results directory..." -ForegroundColor Green
    explorer "$EXPERIMENTS_DIR\$RESULTS_DIR"
}

Write-Host "`nIncremental sparsity evaluation complete!" -ForegroundColor Green
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "  1. Review the plots in the 'comparisons' folder to compare regularization strategies" -ForegroundColor Cyan
Write-Host "  2. Check 'best_models_by_sparsity.csv' to see the best model at each sparsity level" -ForegroundColor Cyan
Write-Host "  3. Look at 'summary_findings.txt' for key insights" -ForegroundColor Cyan
Write-Host "`nAdditional analysis can be performed on the CSV data for more detailed investigations." -ForegroundColor Cyan