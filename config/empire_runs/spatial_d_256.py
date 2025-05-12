# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'spatial-gpt2'
wandb_run_name='spatial_d_256'
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8
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
spatial_cost_scale = 256
spatial_mode = "fixed"
optimize_interval = 500
spatial_d_value = 0.1
l1_scale = 0
