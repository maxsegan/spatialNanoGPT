import torch
import torch.nn as nn

import os
import torch
import math

import torch.nn.functional as F

import torch

def compute_tensor_stats(list_of_lists):
    """
    Computes min, max, mean, and standard deviation for a list of lists of tensors.
    
    Args:
        list_of_lists (List[List[torch.Tensor]]): A list of lists of tensors.
        
    Returns:
        tuple: (min_value, max_value, mean_value, std_value)
    """
    # Flatten each tensor in each sublist to a 1D tensor.
    flattened_tensors = [tensor.view(-1) for sublist in list_of_lists for tensor in sublist]
    
    # Concatenate all flattened tensors into one long tensor.
    all_values = torch.cat(flattened_tensors, dim=0)
    
    # Compute the statistics.
    min_val = all_values.min().item()
    max_val = all_values.max().item()
    mean_val = all_values.mean().item()
    std_val = all_values.std().item()
    
    return min_val, max_val, mean_val, std_val




def collision_penalty(x_in, y_in, x_out, y_out, threshold):
    """
    Computes a repulsive (collision) penalty for all neurons combined
    (both input and output) using torch.cdist for efficiency.

    Args:
        x_in, y_in (torch.Tensor): 1D tensors for input neuron x and y positions.
        x_out, y_out (torch.Tensor): 1D tensors for output neuron x and y positions.
        threshold (float): The minimum allowed distance between neurons.
        lambda_factor (float): Scaling factor for the penalty.

    Returns:
        torch.Tensor: A scalar penalty value.
    """
    # Get device from input tensors
    device = x_in.device
    # Concatenate input and output positions to treat them as one group.
    x_all = torch.cat((x_in, x_out), dim=0)
    y_all = torch.cat((y_in, y_out), dim=0)
    
    # Stack into a tensor of shape [N_total, 2]
    positions = torch.stack((x_all, y_all), dim=1)
    N_total = positions.size(0)
    
    # Compute the pairwise Euclidean distance matrix using torch.cdist
    dists = torch.cdist(positions, positions, p=2)
    
    # Set the diagonal to infinity to avoid self-collision penalty.
    mask = torch.eye(N_total, device=device, dtype=torch.bool)
    dists = dists.masked_fill(mask, float('inf'))

    
    # Compute the repulsive penalty for distances below the threshold.
    penalty = F.relu(threshold - dists) ** 2
    
    # Each pair is counted twice in the symmetric distance matrix, so divide the sum by 2.
    total_penalty = penalty.sum() / 2.0
    return total_penalty


# Function to compute distance matrix for a given linear layer
def compute_distance_matrix(N, M, A, B, D, device="cuda"):
    # Existing code
    x_in = torch.linspace(-A / 2, A / 2, N)
    y_in = torch.full((N,), -D / 2)
    x_out = torch.linspace(-B / 2, B / 2, M)
    y_out = torch.full((M,), D / 2)

    # Convert to learnable parameters
    x_in = nn.Parameter(x_in.to(device))
    y_in = nn.Parameter(y_in.to(device))
    x_out = nn.Parameter(x_out.to(device))
    y_out = nn.Parameter(y_out.to(device))

    return nn.ParameterList([x_in, y_in, x_out, y_out])

def compute_distance_matrix_cdist(o_X, o_Y, i_X, i_Y):
    """
    Uses torch.cdist to compute the pairwise Euclidean distance matrix.
    """
    device = o_X.device  # Get device from input tensor
    inputs = torch.stack((i_X, i_Y), dim=1)
    outputs = torch.stack((o_X, o_Y), dim=1)
    return torch.cdist(inputs, outputs)

class SpatialNet(nn.Module):
    def __init__(self, model, A, B, D, spatial_cost_scale=1,device="cuda"):
        super(SpatialNet, self).__init__()
        self.model = model
        self.linear_layers = []
        self.value_networks = []
        self.value_distance_matrices = nn.ModuleList( [])
        self.linear_distance_matrices = nn.ModuleList( [])
        self.A = A
        self.B = B
        self.D = D
        self.spatial_cost_scale = spatial_cost_scale  # Scaling factor for spatial cost
        self.device=device
        self._extract_layers(model)


    def _extract_layers(self, module):
        for name, layer in module.named_children():
            # If it's a linear layer, add it directly
            if isinstance(layer, nn.Linear):
                self.linear_layers.append(layer)
                N, M = layer.in_features, layer.out_features
                distance_matrix = compute_distance_matrix(N, M, self.A, self.B, self.D)
                self.linear_distance_matrices.append(distance_matrix)
            # If it's the CausalSelfAttention, we handle c_attn and c_proj
            elif layer.__class__.__name__ == "CausalSelfAttention":
                # c_attn is a single linear that produces Q,K,V stacked together.
                c_attn = layer.c_attn
                c_proj = layer.c_proj

                # Add c_proj as a normal linear
                self.linear_layers.append(c_proj)
                N, M = c_proj.in_features, c_proj.out_features
                dist_matrix_cp = compute_distance_matrix(N, M, self.A, self.B, self.D)
                self.linear_distance_matrices.append(dist_matrix_cp)

                # Now handle the value projection part of c_attn
                # c_attn.weight shape: [3*n_embd, n_embd]
                # Value projection is the last n_embd x n_embd chunk
                with torch.no_grad():
                    n_embd = layer.n_embd
                    value_proj_weight = c_attn.weight[2 * n_embd : 3 * n_embd, :]  # W_V
                    # If bias exists:
                    value_proj_bias = None
                    if c_attn.bias is not None:
                        value_proj_bias = c_attn.bias[2 * n_embd : 3 * n_embd]

                # Store this as a "value network"
                # We'll store the reference directly to c_attn parameters since we want them optimized
                # We'll just keep track of indices. But for simplicity, treat them like a network.
                self.value_networks.append(
                    (c_attn.weight[2 * n_embd : 3 * n_embd, :], value_proj_bias)
                )
                # Distance matrix for value projection
                N_v, M_v = value_proj_weight.size(1), value_proj_weight.size(
                    0
                )  # in_features, out_features
                dist_matrix_v = compute_distance_matrix(
                    N_v, M_v, self.A, self.B, self.D
                )
                self.value_distance_matrices.append(dist_matrix_v)

            else:
                # Recursively search children
                self._extract_layers(layer)


    def get_cost(self,quadratic=False):
        total_cost = 0.0
        total_params = 0

        collision_cost = 0
        collision_threshold = self.D

        # Compute cost for linear layers
        for layer, dist_coords in zip(self.linear_layers, self.linear_distance_matrices):
            collision_cost+=collision_penalty(dist_coords[0],dist_coords[1],dist_coords[2],dist_coords[3], collision_threshold)
            dist_matrix = compute_distance_matrix_cdist(dist_coords[0],dist_coords[1],dist_coords[2],dist_coords[3])
            weight_abs = torch.abs(layer.weight)
            if quadratic:
                total_cost += torch.sum(weight_abs * dist_matrix.to(self.device)*dist_matrix.to(self.device))
            else:
                total_cost += torch.sum(weight_abs * dist_matrix.to(self.device))
            total_params += weight_abs.numel()

        # Compute cost for value projection layers
        for value_proj, dist_coords in zip(self.value_networks, self.value_distance_matrices):
            collision_cost+=collision_penalty(dist_coords[0],dist_coords[1],dist_coords[2],dist_coords[3], collision_threshold)

            dist_matrix = compute_distance_matrix_cdist(dist_coords[0],dist_coords[1],dist_coords[2],dist_coords[3])
            weight_abs = torch.abs(value_proj[0])
            if quadratic:
                total_cost += torch.sum(weight_abs * dist_matrix*dist_matrix)
            else:
                total_cost += torch.sum(weight_abs * dist_matrix)
            total_params += weight_abs.numel()

        # Apply the scaling factor to the spatial cost
        #print(self.spatial_cost_scale * total_cost / total_params,collision_cost/total_params)
        return self.spatial_cost_scale * total_cost / total_params + self.spatial_cost_scale * collision_cost / total_params

    def get_stats(self):
        return compute_tensor_stats(self.linear_distance_matrices+self.value_distance_matrices )
    
    def forward(self, idx, targets=None):
        return self.model(idx, targets=targets)
    