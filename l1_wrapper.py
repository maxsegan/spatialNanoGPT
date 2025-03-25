import torch
import torch.nn as nn

class L1RegularizedGPT(nn.Module):
    """Wrapper for GPT model with L1 regularization"""
    def __init__(self, model, l1_scale=0.0, device="cuda"):
        super().__init__()
        self.model = model
        self.l1_scale = l1_scale
        self.device = device
    
    def forward(self, idx, targets=None):
        logits, loss = self.model(idx, targets)
        return logits, loss
    
    def get_cost(self):
        """Compute L1 regularization cost across all model weights"""
        if self.l1_scale <= 0:
            return torch.tensor(0.0).to(self.device)
            
        total_l1 = 0.0
        total_params = 0
        
        # Apply L1 regularization to all weight matrices (excluding biases, embeddings)
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1 and not any(x in name for x in ['wpe', 'wte']):
                total_l1 += torch.sum(torch.abs(param))
                total_params += param.numel()
        
        # Return scaled L1 cost
        return self.l1_scale * total_l1 / total_params
    
    def estimate_mfu(self, *args, **kwargs):
        """Pass through MFU estimation to underlying model"""
        return self.model.estimate_mfu(*args, **kwargs)