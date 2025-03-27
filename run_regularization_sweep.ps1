# PowerShell script for running regularization experiments with spatial mode variants
# Updated to include swappable and learnable spatial modes

# Base configuration
$BASE_CONFIG = "config/train_shakespeare_char.py"
$OUTPUT_ROOT = "regularization_experiments"
$MAX_ITERS = 5000  # Reduce for faster experiments, increase for better convergence
$OPTIMIZE_INTERVAL = 500  # Run Hungarian optimization every 250 iterations for swappable mode

# Create output directory
if (-not (Test-Path $OUTPUT_ROOT)) {
    New-Item -ItemType Directory -Path $OUTPUT_ROOT | Out-Null
}

# Log file
$LOG_FILE = "$OUTPUT_ROOT\experiment_log.txt"
"Starting regularization parameter sweep with spatial modes at $(Get-Date)" | Out-File -FilePath $LOG_FILE

# Function to run a training configuration
function Run-Training {
    param (
        [string]$name,
        [string]$l1_scale,
        [string]$weight_decay,
        [string]$spatial_cost_scale,
        [string]$spatial_mode = "fixed",
        [string]$optimize_interval = "500"
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
    "Spatial Mode: $spatial_mode" | Tee-Object -FilePath $LOG_FILE -Append
    if ($spatial_mode -eq "swappable") {
        "Optimize Interval: $optimize_interval" | Tee-Object -FilePath $LOG_FILE -Append
    }
    "Output directory: $out_dir" | Tee-Object -FilePath $LOG_FILE -Append
    "Start time: $(Get-Date)" | Tee-Object -FilePath $LOG_FILE -Append
    
    # Run the training with specified parameters
    if ($spatial_mode -eq "fixed") {
        python train.py $BASE_CONFIG `
            --l1_scale=$l1_scale `
            --weight_decay=$weight_decay `
            --spatial_cost_scale=$spatial_cost_scale `
            --out_dir=$out_dir `
            --max_iters=$MAX_ITERS `
            --wandb_run_name=$name `
            --eval_interval=500
    }
    else {
        python train.py $BASE_CONFIG `
            --l1_scale=$l1_scale `
            --weight_decay=$weight_decay `
            --spatial_cost_scale=$spatial_cost_scale `
            --spatial_mode=$spatial_mode `
            --optimize_interval=$optimize_interval `
            --out_dir=$out_dir `
            --max_iters=$MAX_ITERS `
            --wandb_run_name=$name `
            --eval_interval=500
    }
    
    "Finished run: $name at $(Get-Date)" | Tee-Object -FilePath $LOG_FILE -Append
    "---------------------------------------------" | Tee-Object -FilePath $LOG_FILE -Append
    "" | Tee-Object -FilePath $LOG_FILE -Append
}

# ======== Swappable Spatial Mode Experiments ========
"Running swappable spatial mode experiments..." | Tee-Object -FilePath $LOG_FILE -Append

Run-Training -name "swappable_spatial_1" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "1.0" -spatial_mode "swappable" -optimize_interval $OPTIMIZE_INTERVAL
Run-Training -name "swappable_spatial_5" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "5.0" -spatial_mode "swappable" -optimize_interval $OPTIMIZE_INTERVAL
Run-Training -name "swappable_spatial_10" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "10.0" -spatial_mode "swappable" -optimize_interval $OPTIMIZE_INTERVAL
Run-Training -name "swappable_spatial_50" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "50.0" -spatial_mode "swappable" -optimize_interval $OPTIMIZE_INTERVAL
Run-Training -name "swappable_spatial_100" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "100.0" -spatial_mode "swappable" -optimize_interval $OPTIMIZE_INTERVAL
Run-Training -name "swappable_spatial_200" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "200.0" -spatial_mode "swappable" -optimize_interval $OPTIMIZE_INTERVAL
Run-Training -name "swappable_spatial_300" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "300.0" -spatial_mode "swappable" -optimize_interval $OPTIMIZE_INTERVAL
Run-Training -name "swappable_spatial_400" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "400.0" -spatial_mode "swappable" -optimize_interval $OPTIMIZE_INTERVAL
Run-Training -name "swappable_spatial_500" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "500.0" -spatial_mode "swappable" -optimize_interval $OPTIMIZE_INTERVAL

# ======== Learnable Spatial Mode Experiments ========
"Running learnable spatial mode experiments..." | Tee-Object -FilePath $LOG_FILE -Append

Run-Training -name "learnable_spatial_1" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "1.0" -spatial_mode "learnable"
Run-Training -name "learnable_spatial_5" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "5.0" -spatial_mode "learnable"
Run-Training -name "learnable_spatial_10" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "10.0" -spatial_mode "learnable"
Run-Training -name "learnable_spatial_50" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "50.0" -spatial_mode "learnable"
Run-Training -name "learnable_spatial_100" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "100.0" -spatial_mode "learnable"
Run-Training -name "learnable_spatial_200" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "200.0" -spatial_mode "learnable"
Run-Training -name "learnable_spatial_300" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "300.0" -spatial_mode "learnable"
Run-Training -name "learnable_spatial_400" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "400.0" -spatial_mode "learnable"
Run-Training -name "learnable_spatial_500" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "500.0" -spatial_mode "learnable"

# ======== L1 Regularization Experiments ========
"Running L1 regularization experiments..." | Tee-Object -FilePath $LOG_FILE -Append

Run-Training -name "l1_75" -l1_scale "75.0" -weight_decay "0.0" -spatial_cost_scale "0.0"
Run-Training -name "l1_75_l2_1e-1" -l1_scale "75.0" -weight_decay "1e-1" -spatial_cost_scale "0.0"

# ======== Regular Spatial Experiments ========
"Running regular spatial regularization experiments..." | Tee-Object -FilePath $LOG_FILE -Append

Run-Training -name "spatial_200" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "200.0"
Run-Training -name "spatial_300" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "300.0"
Run-Training -name "spatial_400" -l1_scale "0.0" -weight_decay "0.0" -spatial_cost_scale "400.0"

"All experiments completed!" | Tee-Object -FilePath $LOG_FILE -Append
"Final completion time: $(Get-Date)" | Tee-Object -FilePath $LOG_FILE -Append
"Results are stored in: $OUTPUT_ROOT\" | Tee-Object -FilePath $LOG_FILE -Append

# Create a summary CSV of all experiments
$SUMMARY_CSV = "$OUTPUT_ROOT\experiment_summary.csv"
"Name,L1 Scale,Weight Decay (L2),Spatial Cost Scale,Spatial Mode,Output Directory" | Out-File -FilePath $SUMMARY_CSV -Encoding utf8

# Swappable spatial experiments
"swappable_spatial_1,0.0,0.0,1.0,swappable,$OUTPUT_ROOT\swappable_spatial_1" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"swappable_spatial_5,0.0,0.0,5.0,swappable,$OUTPUT_ROOT\swappable_spatial_5" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"swappable_spatial_10,0.0,0.0,10.0,swappable,$OUTPUT_ROOT\swappable_spatial_10" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"swappable_spatial_50,0.0,0.0,50.0,swappable,$OUTPUT_ROOT\swappable_spatial_50" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"swappable_spatial_100,0.0,0.0,100.0,swappable,$OUTPUT_ROOT\swappable_spatial_100" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"swappable_spatial_200,0.0,0.0,200.0,swappable,$OUTPUT_ROOT\swappable_spatial_200" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"swappable_spatial_300,0.0,0.0,300.0,swappable,$OUTPUT_ROOT\swappable_spatial_300" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"swappable_spatial_400,0.0,0.0,400.0,swappable,$OUTPUT_ROOT\swappable_spatial_400" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"swappable_spatial_500,0.0,0.0,500.0,swappable,$OUTPUT_ROOT\swappable_spatial_500" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8

# Learnable spatial experiments
"learnable_spatial_1,0.0,0.0,1.0,learnable,$OUTPUT_ROOT\learnable_spatial_1" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"learnable_spatial_5,0.0,0.0,5.0,learnable,$OUTPUT_ROOT\learnable_spatial_5" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"learnable_spatial_10,0.0,0.0,10.0,learnable,$OUTPUT_ROOT\learnable_spatial_10" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"learnable_spatial_50,0.0,0.0,50.0,learnable,$OUTPUT_ROOT\learnable_spatial_50" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"learnable_spatial_100,0.0,0.0,100.0,learnable,$OUTPUT_ROOT\learnable_spatial_100" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"learnable_spatial_200,0.0,0.0,200.0,learnable,$OUTPUT_ROOT\learnable_spatial_200" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"learnable_spatial_300,0.0,0.0,300.0,learnable,$OUTPUT_ROOT\learnable_spatial_300" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"learnable_spatial_400,0.0,0.0,400.0,learnable,$OUTPUT_ROOT\learnable_spatial_400" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"learnable_spatial_500,0.0,0.0,500.0,learnable,$OUTPUT_ROOT\learnable_spatial_500" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8

# L1 experiments
"l1_75,75.0,0.0,0.0,fixed,$OUTPUT_ROOT\l1_75" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"l1_75_l2_1e-1,75.0,1e-1,0.0,fixed,$OUTPUT_ROOT\l1_75_l2_1e-1" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8

# Regular spatial experiments
"spatial_200,0.0,0.0,200.0,fixed,$OUTPUT_ROOT\spatial_200" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"spatial_300,0.0,0.0,300.0,fixed,$OUTPUT_ROOT\spatial_300" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8
"spatial_400,0.0,0.0,400.0,fixed,$OUTPUT_ROOT\spatial_400" | Out-File -FilePath $SUMMARY_CSV -Append -Encoding utf8

"Experiment summary saved to $SUMMARY_CSV"