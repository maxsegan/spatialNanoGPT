import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment

def alternative_hungarian_optimization(W, C, max_iter=100, tol=1e-6, verbose=False):
    """
    Finds row and column permutations for matrix C to minimize mean(W * C_perm),
    where * is the elementwise product. Returns the new permuted C matrix.
    
    Parameters:
      W: 2D numpy array of shape (m, n) with nonnegative entries.
      C: 2D numpy array of shape (m, n) with nonnegative entries.
      max_iter: maximum number of alternating iterations.
      tol: tolerance for convergence (based on change in objective).
      verbose: if True, prints progress.
      
    Returns:
      row_perm: permutation of row indices for C (as a numpy array).
      col_perm: permutation of column indices for C (as a numpy array).
      obj: final objective value.
      C_new: the permuted version of C, i.e., C[np.ix_(row_perm, col_perm)].
    """
    W=W.detach().cpu().numpy()
    C=C.detach().cpu().numpy() 
    m, n = W.shape
    
    # Initialize with the identity permutation.
    row_perm = np.arange(m)
    col_perm = np.arange(n)
    
    def compute_objective(row_perm, col_perm):
        # Compute mean of elementwise product after applying the permutations.
        C_perm = C[np.ix_(row_perm, col_perm)]
        return np.mean(W * C_perm)
    
    obj_prev = compute_objective(row_perm, col_perm)
    if verbose:
        print("Initial objective: ", obj_prev)
    
    for iteration in range(max_iter):
        # --- Step 1: Optimize row permutation (with fixed col_perm) ---
        # Vectorized computation of cost for rows:
        # cost_rows[i, k] = dot(W[i, :], C[k, col_perm])
        cost_rows = np.dot(W, C[:, col_perm].T)
        _, new_row_perm = linear_sum_assignment(cost_rows)
        row_perm = new_row_perm  # new_row_perm gives the row from C assigned to row i in W
        
        # --- Step 2: Optimize column permutation (with fixed row_perm) ---
        # Vectorized computation of cost for columns:
        # cost_cols[j, l] = dot(W[:, j], C[row_perm, l])
        cost_cols = np.dot(W.T, C[row_perm, :])
        _, new_col_perm = linear_sum_assignment(cost_cols)
        col_perm = new_col_perm
        
        # --- Check convergence ---
        obj_current = compute_objective(row_perm, col_perm)
        if verbose:
            print(f"Iteration {iteration+1}: objective = {obj_current}",flush=True)
        
        if abs(obj_prev - obj_current) < tol:
            break
        obj_prev = obj_current
        
    # Create the new permuted version of C.
    C_new = C[np.ix_(row_perm, col_perm)]
    return torch.tensor(C_new)


def compute_distance_matrix(N, M, A, B, D):
    x_in = torch.linspace(-A / 2, A / 2, N)
    y_in = torch.full((N,), -D / 2)
    x_out = torch.linspace(-B / 2, B / 2, M)
    y_out = torch.full((M,), D / 2)

    x_in = x_in.view(N, 1)
    y_in = y_in.view(N, 1)
    x_out = x_out.view(1, M)
    y_out = y_out.view(1, M)

    distance_matrix = torch.sqrt((x_out - x_in) ** 2 + (y_out - y_in) ** 2)
    return distance_matrix.T


class SpatialNet(nn.Module):
    """
    This class wraps a GPT model from nanoGPT and adds a spatialized cost term.

    It finds all linear layers in the model (including those inside the MLP and
    CausalSelfAttention modules) and also extracts the V projection weights from
    the attention (c_attn) layer to create a "value network" cost similar to the
    ViT example.
    """

    def __init__(
        self, model, A=1.0, B=1.0, D=1.0, spatial_cost_scale=1e-4, device="cuda"
    ):
        super(SpatialNet, self).__init__()
        self.model = model
        self.linear_layers = []
        self.value_networks = []
        self.value_distance_matrices = []
        self.linear_distance_matrices = []
        self.A = A
        self.B = B
        self.D = D
        self.spatial_cost_scale = spatial_cost_scale
        self.device = device

        self._extract_layers(model)
        self.linear_distance_matrices = [
            m.to(device) for m in self.linear_distance_matrices
        ]
        self.value_distance_matrices = [
            m.to(device) for m in self.value_distance_matrices
        ]

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

    def optimize(self):
        print("init",self.get_cost(),flush=True)
        # Compute cost for linear layers
        new_dist_matrices= []
        i=0
        total=len(self.linear_distance_matrices)+len(self.value_distance_matrices),
        for layer, dist_matrix in zip(self.linear_layers, self.linear_distance_matrices):
            i+=1
            #print(i,total,dist_matrix.shape,flush=True)
            weight_abs = torch.abs(layer.weight)
            new_dist_matrices.append(alternative_hungarian_optimization(weight_abs, dist_matrix))
        self.linear_distance_matrices=new_dist_matrices
        new_dist_matrices= []
        # Compute cost for value projection layers
        for value_proj, dist_matrix in zip(self.value_networks, self.value_distance_matrices):
            i+=1
            #print(i,total,dist_matrix.shape,flush=True)

            weight_abs = torch.abs(value_proj[0])
            new_dist_matrices.append(alternative_hungarian_optimization(weight_abs, dist_matrix))
        self.value_distance_matrices=new_dist_matrices
        print("final",self.get_cost(),flush=True)

    def get_cost(self):
        total_cost = 0.0
        total_params = 0
        costs = []
        param_counts = []

        # Compute cost for linear layers
        for layer, dist_matrix in zip(
            self.linear_layers,
            self.linear_distance_matrices):
            weight_abs = torch.abs(layer.weight)
            costs.append(torch.sum(weight_abs * dist_matrix))
            param_counts.append(weight_abs.numel())

        # Compute cost for value projection layers
        for (value_proj_weight, _), dist_matrix in zip(
            self.value_networks, self.value_distance_matrices
        ):
            weight_abs = torch.abs(value_proj_weight)
            costs.append(torch.sum(weight_abs * dist_matrix))
            param_counts.append(weight_abs.numel())
        if costs:
            total_cost = torch.stack(costs).sum()
            total_params = sum(param_counts)
        # Apply the scaling factor to the spatial cost
        if total_params > 0 or self.spatial_cost_scale > 0:
            return self.spatial_cost_scale * total_cost / total_params
        else:
            return torch.tensor(0.0).to(self.device)

    def forward(self, idx, targets=None):
        return self.model(idx, targets=targets)

