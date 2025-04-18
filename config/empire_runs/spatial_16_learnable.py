# config for training GPT-2 (124M) with spatial regularization
wandb_log = True
wandb_project = 'spatial-gpt2'
wandb_run_name = 'spatial_16_learnable'
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 6
block_size = 1024
gradient_accumulation_steps = 2 * 5 * 8
# this makes total number of tokens be 300B / 6
max_iters = 100000
lr_decay_iters = 100000
# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10
# weight decay (L2 regularization)
weight_decay = 0
# regularization
spatial_cost_scale = 16
spatial_mode = "learnable"
optimize_interval = 500
l1_scale = 0.0
