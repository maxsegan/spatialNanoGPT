# PowerShell script for running regularization experiments with higher hyperparameters
# This is an updated version with increased parameter values

# Base configuration
$BASE_CONFIG = "config/train_shakespeare_char.py"
$OUTPUT_ROOT = "regularization_experiments"
$MAX_ITERS = 5000  # Reduce for faster experiments, increase for better convergence

# Create output directory
if (-not (Test-Path $OUTPUT_ROOT)) {
    New-Item -ItemType Directory -Path $OUTPUT_ROOT | Out-Null
}

# Log file
$LOG_FILE = "$OUTPUT_ROOT\experiment_log.txt"
"Starting regularization parameter sweep with higher values at $(Get-Date)" | Out-File -FilePath $LOG_FILE

# Function to run a training configuration
function Run-Training {
    param (
        [string]$name,
        [string]$l1_scale,
        [string]$weight_decay,
        [string]$spatial_cost_scale
    )
    
    $out_dir = "$OUTPUT_ROOT\$name"
    if (-not (Test-Path $out_dir)) {
        New-Item -ItemType Directory -Path $out_dir | Out-Null
    }
    
    "---------------------------------------------" | Tee-Object -FilePath $LOG_FILE -Append
    "Starting run: $name" | Tee-Object -FilePath $LOG_FILE -Append
    "L1 Scale: $l1_scale" | Tee-Object -FilePath $LOG_FILE -Append
    "Weight Decay (L2): $weight_decay" | Tee-Object -FilePath $LOG_FILE -Append
    "Spatial Cost Scale: $spatial_cost_scale" | Tee-Object -FilePath $LOG_FILE -Append
    "Output directory: $out_dir" | Tee-Object -FilePath $LOG_FILE -Append
    "Start time: $(Get-Date)" | Tee-Object -FilePath $LOG_FILE -Append
    
    # Run the training with specified parameters
    python train.py $BASE_CONFIG `
        --l1_scale=$l1_scale `
        --weight_decay=$weight_decay `
        --spatial_cost_scale=$spatial_cost_scale `
        --out_dir=$out_dir `
        --max_iters=$MAX_ITERS `
        --wandb_run_name=$name `
        --eval_interval=500
    
    "Finished run: $name at $(Get-Date)" | Tee-Object -FilePath $LOG_FILE -Append
    "---------------------------------------------" | Tee-Object -FilePath $LOG_FILE -Append
    "" | Tee-Object -FilePath $LOG_FILE -Append
}

# ======== L2 Regularization Experiments ========
"Running L2 regularization experiments..." | Tee-Object -FilePath $LOG_FILE -Append

# L2 of: 5e-1, 1, 5, 10, 50 (one step higher than before)
Run-Training -name "l2_5e-1" -l1_scale "0.0" -weight_decay "5e-1" -spatial_cost_scale "0.0"
Run-Training -name "l2_1" -l1_scale "0.0" -weight_decay "1" -spatial_cost_scale "0.0"
Run-Training -name "l2_5" -l1_scale "0.0" -weight_decay "5" -spatial_cost_scale "0.0"
Run-Training -name "l2_10" -l1_scale "0.0" -weight_decay "10" -spatial_cost_scale "0.0"
Run-Training -name "l2_50" -l1_scale "0.0" -weight_decay "50" -spatial_cost_scale "0.0"

# ======== L1 + L2 Regularization Experiments ========
"Running L1 + L2 regularization experiments..." | Tee-Object -FilePath $LOG_FILE -Append

# L1 (increased) + L2 (keep at 1e-1 as specified): 1e-7, 5e-7, 1e-6, 5e-6, 1e-5
Run-Training -name "l1_1e-7_l2_1e-1" -l1_scale "1e-7" -weight_decay "1e-1" -spatial_cost_scale "0.0"
Run-Training -name "l1_5e-7_l2_1e-1" -l1_scale "5e-7" -weight_decay "1e-1" -spatial_cost_scale "0.0"
Run-Training -name "l1_1e-6_l2_1e-1" -l1_scale "1e-6" -weight_decay "1e-1" -spatial_cost_scale "0.0"
Run-Training -name "l1_5e-6_l2_1e-1" -l1_scale "5e-6" -weight_decay "1e-1" -spatial_cost_scale "0.0"
Run-Training -name "l1_1e-5_l2_1e-1" -l1_scale "1e-5" -weight_decay "1e-1" -spatial_cost_scale "0.0"

# ======== L1 Only Regularization Experiments ========
"Running L1 only regularization experiments..." | Tee-Object -FilePath $LOG_FILE -Append

# L1 only: 1e-7, 5e-7, 1e-6, 5e-6, 1e-5
Run-Training -name "l1_1e-7_only" -l1_scale "1e-7" -weight_decay "0.0" -spatial_cost_scale "0.0"
Run-Training -name "l1_5e-7_only" -l1_scale "5e-7" -weight_decay "0.0" -spatial_cost_scale "0.0"
Run-Training -name "l1_1e-6_only" -l1_scale "1e-6" -weight_decay "0.0" -spatial_cost_scale "0.0"
Run-Training -name "l1_5e-6_only" -l1_scale "5e-6" -weight_decay "0.0" -spatial_cost_scale "0.0"
Run-Training -name "l1_1e-5_only" -l1_scale "1e-5" -weight_decay "0.0" -spatial_cost_scale "0.0"

# ======== Spatial Only Regularization Experiments ========
"Running spatial only regularization experiments..." | Tee-Object -FilePath $LOG_FILE -Append

# Spatial only: 100, 500
Run-Training -name "spatial_100_only" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "100.0"
Run-Training -name "spatial_500_only" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "500.0"

# ======== Spatial + L2 Regularization Experiments ========
"Running spatial + L2 regularization experiments..." | Tee-Object -FilePath $LOG_FILE -Append

# Spatial (increased) + L2 (keep at 1e-1 as specified): 1, 5, 10, 50, 100
Run-Training -name "spatial_100_l2_1e-1" -l1_scale "0.0" -weight_decay "1e-1" -spatial_cost_scale "100.0"
Run-Training -name "spatial_500_l2_1e-1" -l1_scale "0.0" -weight_decay "1e-1" -spatial_cost_scale "500.0"

"All experiments completed!" | Tee-Object -FilePath $LOG_FILE -Append
"Final completion time: $(Get-Date)" | Tee-Object -FilePath $LOG_FILE -Append
"Results are stored in: $OUTPUT_ROOT\" | Tee-Object -FilePath $LOG_FILE -Append

# Create a summary CSV of all experiments
$SUMMARY_CSV = "$OUTPUT_ROOT\experiment_summary.csv"
"Name,L1 Scale,Weight Decay (L2),Spatial Cost Scale,Output Directory" | Out-File -FilePath $SUMMARY_CSV -Encoding utf8

# L2 experiments
"l2_5e-1,0.0,5e-1,0.0,$OUTPUT_ROOT\l2_5e-1" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"l2_1,0.0,1.0,0.0,$OUTPUT_ROOT\l2_1" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"l2_5,0.0,5.0,0.0,$OUTPUT_ROOT\l2_5" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"l2_10,0.0,10.0,0.0,$OUTPUT_ROOT\l2_10" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"l2_50,0.0,50.0,0.0,$OUTPUT_ROOT\l2_50" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8

# L1 + L2 experiments
"l1_1e-7_l2_1e-1,1e-7,1e-1,0.0,$OUTPUT_ROOT\l1_1e-7_l2_1e-1" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"l1_5e-7_l2_1e-1,5e-7,1e-1,0.0,$OUTPUT_ROOT\l1_5e-7_l2_1e-1" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"l1_1e-6_l2_1e-1,1e-6,1e-1,0.0,$OUTPUT_ROOT\l1_1e-6_l2_1e-1" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"l1_5e-6_l2_1e-1,5e-6,1e-1,0.0,$OUTPUT_ROOT\l1_5e-6_l2_1e-1" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"l1_1e-5_l2_1e-1,1e-5,1e-1,0.0,$OUTPUT_ROOT\l1_1e-5_l2_1e-1" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8

# L1 only experiments
"l1_1e-7_only,1e-7,0.0,0.0,$OUTPUT_ROOT\l1_1e-7_only" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"l1_5e-7_only,5e-7,0.0,0.0,$OUTPUT_ROOT\l1_5e-7_only" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"l1_1e-6_only,1e-6,0.0,0.0,$OUTPUT_ROOT\l1_1e-6_only" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"l1_5e-6_only,5e-6,0.0,0.0,$OUTPUT_ROOT\l1_5e-6_only" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"l1_1e-5_only,1e-5,0.0,0.0,$OUTPUT_ROOT\l1_1e-5_only" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8

# Spatial only experiments
"spatial_100_only,0.0,0.0,1.0,$OUTPUT_ROOT\spatial_100_only" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"spatial_500_only,0.0,0.0,5.0,$OUTPUT_ROOT\spatial_500_only" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8

# Spatial + L2 experiments
"spatial_100_l2_1e-1,0.0,1e-1,1.0,$OUTPUT_ROOT\spatial_100_l2_1e-1" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"spatial_500_l2_1e-1,0.0,1e-1,5.0,$OUTPUT_ROOT\spatial_500_l2_1e-1" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8

"Experiment summary saved to $SUMMARY_CSV"