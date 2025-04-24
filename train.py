"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from regularized_gpt import RegularizedGPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 100000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# regularization
optimize_interval = 500
spatial_mode = "fixed"
spatial_cost_scale = 1e-5
spatial_d_value = 0.0
l1_scale = 0.0
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
# Set manual seed - will be overridden later if resuming from checkpoint
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, wandb_run_name + 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    
    # Restore RNG states to ensure reproducible sampling
    if ddp:
        # For DDP, load the RNG state specific to this rank
        if 'cpu_rng_states' in checkpoint and 'ddp_world_size' in checkpoint:
            saved_world_size = checkpoint['ddp_world_size']
            
            # If the current world size matches the saved one, we can restore exact states
            if saved_world_size == ddp_world_size:
                # Convert numpy back to tensor for CPU state
                cpu_rng_state = torch.from_numpy(checkpoint['cpu_rng_states'][ddp_rank])
                # Ensure the state is the right size (trim excess padding)
                if 'cpu_rng_state_sizes' in checkpoint:
                    state_size = checkpoint['cpu_rng_state_sizes'][ddp_rank]
                    cpu_rng_state = cpu_rng_state[:state_size]
                torch.set_rng_state(cpu_rng_state)
                
                # Handle GPU states if available
                if torch.cuda.is_available() and 'gpu_rng_states' in checkpoint:
                    gpu_rng_state = torch.from_numpy(checkpoint['gpu_rng_states'][ddp_rank])
                    # Ensure the state is the right size (trim excess padding)
                    if 'gpu_rng_state_sizes' in checkpoint:
                        state_size = checkpoint['gpu_rng_state_sizes'][ddp_rank]
                        gpu_rng_state = gpu_rng_state[:state_size]
                    torch.cuda.set_rng_state(gpu_rng_state)
                
                print(f"Rank {ddp_rank}: Restored RNG state from checkpoint")
            else:
                # If world sizes don't match, we can't restore exact per-rank states
                # Instead, set a deterministic but different seed for each rank
                print(f"Warning: Previous run had {saved_world_size} processes, now using {ddp_world_size}.")
                print(f"Rank {ddp_rank}: Using deterministic seed derived from checkpoint iter_num")
                rand_seed = 1337 + iter_num * 1000 + ddp_rank
                torch.manual_seed(rand_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(rand_seed)
    else:
        # For non-DDP, restore the single process state
        if 'cpu_rng_state' in checkpoint:
            torch.set_rng_state(torch.from_numpy(checkpoint['cpu_rng_state']))
            if torch.cuda.is_available() and 'gpu_rng_state' in checkpoint:
                torch.cuda.set_rng_state(torch.from_numpy(checkpoint['gpu_rng_state']))
            print("Restored RNG state from checkpoint")
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

regularized_model = RegularizedGPT(model, A=1.0, B=1.0, D=spatial_d_value, spatial_cost_scale=spatial_cost_scale, l1_scale=l1_scale, spatial_mode=spatial_mode, device=device)
model.to(device)
raw_model = regularized_model.model  # This is the underlying GPT model for optimizer
if init_from == 'resume':
    print("Loading regularization settings from checkpoint")
    
    # Load spatial matrices for swappable mode
    if spatial_mode == "swappable" and 'linear_distance_matrices' in checkpoint and 'value_distance_matrices' in checkpoint:
        spatial_net = regularized_model.module.spatial_net if ddp else regularized_model.spatial_net
        
        # Load linear distance matrices
        for i, matrix in enumerate(checkpoint['linear_distance_matrices']):
            if i < len(spatial_net.linear_distance_matrices):
                spatial_net.linear_distance_matrices[i] = matrix.to(device)
        
        # Load value distance matrices
        for i, matrix in enumerate(checkpoint['value_distance_matrices']):
            if i < len(spatial_net.value_distance_matrices):
                spatial_net.value_distance_matrices[i] = matrix.to(device)
        
        print("Loaded optimized distance matrices from checkpoint")
    
    # Load spatial state dict for learnable mode
    elif spatial_mode == "learnable" and 'spatial_state' in checkpoint:
        spatial_net = regularized_model.module.spatial_net if ddp else regularized_model.spatial_net
        spatial_net.load_state_dict(checkpoint['spatial_state'])
        print("Loaded learnable spatial parameters from checkpoint")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = regularized_model
    regularized_model = torch.compile(regularized_model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    regularized_model = DDP(regularized_model, device_ids=[ddp_local_rank])
raw_model = regularized_model.module.model if ddp else regularized_model.model

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    regularized_model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = regularized_model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    regularized_model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
                "l1_scale": l1_scale,
                "spatial_cost_scale": spatial_cost_scale,
                "spatial_d_value": spatial_d_value,  # Added D value to logging
                "weight_decay": weight_decay,
                "regularization_cost": cost.item() if ('cost' in locals() or 'cost' in globals()) and cost is not None else 0.0,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                # Set up the checkpoint
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'regularization': {
                        'l1_scale': l1_scale,
                        'spatial_cost_scale': spatial_cost_scale,
                        'spatial_d_value': spatial_d_value,  # Added D value to saved config
                        'weight_decay': weight_decay,
                        'spatial_mode': spatial_mode,
                    },
                }
                
                # Handle RNG states for each process
                if ddp:
                    # Get local RNG states
                    cpu_rng_state = torch.get_rng_state()
                    gpu_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                    
                    # Serialize the RNG states to tensors of fixed size
                    cpu_state_size = cpu_rng_state.numel()
                    max_cpu_state_size = torch.tensor([cpu_state_size], device=device)
                    torch.distributed.all_reduce(max_cpu_state_size, op=torch.distributed.ReduceOp.MAX)
                    max_cpu_state_size = max_cpu_state_size.item()
                    
                    # Pad CPU RNG state if needed
                    if cpu_state_size < max_cpu_state_size:
                        padded_cpu_state = torch.zeros(max_cpu_state_size, dtype=cpu_rng_state.dtype, device=device)
                        padded_cpu_state[:cpu_state_size] = cpu_rng_state.to(device)
                        cpu_rng_state_tensor = padded_cpu_state
                    else:
                        cpu_rng_state_tensor = cpu_rng_state.to(device)
                    
                    # Create a tensor list to gather all CPU RNG states
                    cpu_states_gathered = [torch.zeros_like(cpu_rng_state_tensor) for _ in range(ddp_world_size)]
                    torch.distributed.all_gather(cpu_states_gathered, cpu_rng_state_tensor)
                    
                    # Process GPU RNG states if available
                    if gpu_rng_state is not None:
                        gpu_state_size = gpu_rng_state.numel()
                        max_gpu_state_size = torch.tensor([gpu_state_size], device=device)
                        torch.distributed.all_reduce(max_gpu_state_size, op=torch.distributed.ReduceOp.MAX)
                        max_gpu_state_size = max_gpu_state_size.item()
                        
                        # Pad GPU RNG state if needed
                        if gpu_state_size < max_gpu_state_size:
                            padded_gpu_state = torch.zeros(max_gpu_state_size, dtype=gpu_rng_state.dtype, device=device)
                            padded_gpu_state[:gpu_state_size] = gpu_rng_state.to(device)
                            gpu_rng_state_tensor = padded_gpu_state
                        else:
                            gpu_rng_state_tensor = gpu_rng_state.to(device)
                        
                        # Create a tensor list to gather all GPU RNG states
                        gpu_states_gathered = [torch.zeros_like(gpu_rng_state_tensor) for _ in range(ddp_world_size)]
                        torch.distributed.all_gather(gpu_states_gathered, gpu_rng_state_tensor)
                        
                        # Store in checkpoint (only the master process will use this)
                        checkpoint['gpu_rng_states'] = [state.cpu().numpy() for state in gpu_states_gathered]
                        checkpoint['gpu_rng_state_sizes'] = [gpu_state_size] * ddp_world_size
                    
                    # Store in checkpoint (only the master process will use this)
                    checkpoint['cpu_rng_states'] = [state.cpu().numpy() for state in cpu_states_gathered]
                    checkpoint['cpu_rng_state_sizes'] = [cpu_state_size] * ddp_world_size
                    checkpoint['ddp_world_size'] = ddp_world_size
                else:
                    # For non-DDP, just store the single process state
                    checkpoint['cpu_rng_state'] = torch.get_rng_state().numpy()
                    if torch.cuda.is_available():
                        checkpoint['gpu_rng_state'] = torch.cuda.get_rng_state().numpy()
                if spatial_mode in ["learnable", "swappable"]:
                    spatial_net = regularized_model.module.spatial_net if ddp else regularized_model.spatial_net
                    
                    if spatial_mode == "learnable":
                        checkpoint['spatial_state'] = spatial_net.state_dict()
                    elif spatial_mode == "swappable":
                        linear_matrices = [matrix.detach().cpu() for matrix in spatial_net.linear_distance_matrices]
                        checkpoint['linear_distance_matrices'] = linear_matrices
                        
                        # Save value distance matrices
                        value_matrices = [matrix.detach().cpu() for matrix in spatial_net.value_distance_matrices]
                        checkpoint['value_distance_matrices'] = value_matrices
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, wandb_run_name + 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            regularized_model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = regularized_model(X, Y)
            # Scale the entire loss (model + cost) for gradient accumulation
            cost = regularized_model.module.get_cost() if ddp else regularized_model.get_cost()
            total_loss = (loss + cost) / gradient_accumulation_steps

        X, Y = get_batch('train')
        scaler.scale(total_loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(regularized_model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    if iter_num % optimize_interval == 0:
        # Run Hungarian optimization for swappable
        if spatial_mode == "swappable":
            if master_process:
                print(f"Running Hungarian optimization at step {iter_num}")
            regularized_model.module.spatial_net.optimize() if ddp else regularized_model.spatial_net.optimize()
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
