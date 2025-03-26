import torch
import torch.nn as nn
from spatial_wrapper import SpatialNet
from l1_wrapper import L1RegularizedGPT

class RegularizedGPT(nn.Module):
    """
    A wrapper class that can apply L1, L2, and spatial regularization to a GPT model.
    - L2 regularization is applied through the optimizer's weight_decay
    - L1 regularization is calculated explicitly
    - Spatial regularization is calculated using a SpatialNet
    """
    def __init__(
        self, 
        model, 
        l1_scale=0.0, 
        spatial_cost_scale=0.0,
        A=1.0, 
        B=1.0, 
        D=1.0, 
        spatial_mode="fixed",  # "fixed", "learnable", and in the future "swappable"
        device="cuda"
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.l1_scale = l1_scale
        
        self.spatial_enabled = spatial_cost_scale > 0
        if self.spatial_enabled:
            if spatial_mode == "fixed" or spatial_mode == "swappable":
                self.spatial_net = SpatialNet(
                    model=model,
                    A=A,
                    B=B,
                    D=D,
                    spatial_cost_scale=spatial_cost_scale,
                    device=device
                )
            elif spatial_mode == "learnable":
                from spatial_wrapper_learn import SpatialNet as LearnableSpatialNet
                self.spatial_net = LearnableSpatialNet(
                    model=model,
                    A=A,
                    B=B,
                    D=D,
                    spatial_cost_scale=spatial_cost_scale,
                    device=device
                )
        
        self.l1_enabled = l1_scale > 0
        if self.l1_enabled:
            self.l1_regularizer = L1RegularizedGPT(
                model=model,
                l1_scale=l1_scale,
                device=device
            )
    
    def forward(self, idx, targets=None):
        """Forward pass through the model"""
        if self.spatial_enabled:
            return self.spatial_net(idx, targets)
        else:
            return self.model(idx, targets)
    
    def get_cost(self):
        """Compute the combined regularization cost"""
        total_cost = 0.0
        
        if self.spatial_enabled:
            total_cost += self.spatial_net.get_cost()
        
        if self.l1_enabled:
            total_cost += self.l1_regularizer.get_cost()
            
        return total_cost
    
    def estimate_mfu(self, *args, **kwargs):
        """Pass through MFU estimation to underlying model"""
        return self.model.estimate_mfu(*args, **kwargs)