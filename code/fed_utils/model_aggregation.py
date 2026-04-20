import torch
import os
import logging
import time
from torch.nn.functional import normalize
import numpy as np
from collections import OrderedDict 
from scipy.optimize import minimize
from peft.utils import set_peft_model_state_dict, get_peft_model_state_dict


def FedAvg(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    num_clients = len(selected_clients_set)
    weights_array = torch.tensor([1.0 / num_clients for _ in selected_clients_set], dtype=torch.float32)
    weighted_global_weights = None
    
    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(epoch), f"local_output_{client_id}", "adapter_model.bin")
        single_weights = torch.load(single_output_dir, map_location='cpu')
        
        if k == 0:
            weighted_global_weights = {
                key: single_weights[key].to(dtype=compute_dtype) * weights_array[k] 
                for key in single_weights.keys()
            }
        else:
            for key in single_weights.keys():
                weighted_global_weights[key] += single_weights[key].to(dtype=compute_dtype) * weights_array[k]
        
        del single_weights

    set_peft_model_state_dict(model, weighted_global_weights, "default")
    del weighted_global_weights
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model



def RecoFed_aggregation_het_rank(
    model,
    global_model_state_before_round: dict,
    client_update_deltas: dict,
    peft_config,
    c: float = 0.2,
    global_learning_rate: float = 1.0
):
    total_start_time = time.perf_counter()
    auxiliary_time_cost = 0.0 

    if not client_update_deltas: 
        return model, 0.0

    client_ids = sorted(client_update_deltas.keys())
    num_tasks = len(client_ids)
    device = model.device

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    
    # Preload the base parameters.
    A_all = {k: v.to(device, dtype=compute_dtype) for k, v in global_model_state_before_round.items() if "lora_A" in k}
    B_all = {k: v.to(device, dtype=compute_dtype) for k, v in global_model_state_before_round.items() if "lora_B" in k}

    def _get_lora_names(sd): 
        return [k.replace('.lora_A.weight', '') for k in sd.keys() if '.lora_A.weight' in k]
    lora_names = _get_lora_names(global_model_state_before_round)

    # Pass 1: compute the Gram matrix H_ij = <g_i, g_j>.
    logging.info("Step 1/3: Pass 1 - computing the Gram matrix with streaming...")
    
    gram_matrix = torch.zeros((num_tasks, num_tasks), device=device, dtype=torch.float32)
    layer_shapes = {}

    for layer_name in lora_names:
        key_A = f"{layer_name}.lora_A.weight"
        key_B = f"{layer_name}.lora_B.weight"
        if key_A not in A_all: continue
        
        shape_A = A_all[key_A].shape
        shape_B = B_all[key_B].shape
        shape = (shape_B[0], shape_A[1])
        num_elm = shape[0] * shape[1]
        layer_shapes[layer_name] = shape
        
        # Temporary buffer: [num_clients, layer_size].
        layer_updates = torch.zeros((num_tasks, num_elm), device=device, dtype=compute_dtype)
        has_update = False
        
        for i, cid in enumerate(client_ids):
            upd = client_update_deltas[cid]
            if key_A in upd and key_B in upd:
                has_update = True
                dA = upd[key_A].to(device, dtype=compute_dtype)
                dB = upd[key_B].to(device, dtype=compute_dtype)
                
                W_new = (B_all[key_B] + dB) @ (A_all[key_A] + dA)
                W_old = B_all[key_B] @ A_all[key_A]
                
                diff = (W_new - W_old).flatten()
                layer_updates[i] = diff
                
                del W_new, W_old, diff, dA, dB

        if has_update:
            gram_matrix.add_(torch.matmul(layer_updates.float(), layer_updates.float().T))
        
        del layer_updates

    logging.info("Step 2/3: solving optimal weights w*...")
    
    H_np = gram_matrix.cpu().float().numpy()
    g0_norm_sq = H_np.sum() / (num_tasks ** 2)
    phi_sqrt = np.sqrt(c**2 * g0_norm_sq)
    
    def obj(w):
        term1 = (1/num_tasks) * w.T @ H_np @ np.ones(num_tasks)
        term2 = phi_sqrt * np.sqrt(max(1e-12, w.T @ H_np @ w))
        return term1 + term2
        
    # Constraints: sum(w) = 1, 0 <= w <= 1.
    res = minimize(obj, np.ones(num_tasks)/num_tasks, method='SLSQP', 
                   bounds=[(0,1)]*num_tasks, constraints={'type':'eq','fun':lambda w:sum(w)-1})
    
    w_opt = torch.tensor(res.x if res.success else np.ones(num_tasks)/num_tasks, device=device, dtype=torch.float32)
    
    # Consensus update = sum(rho_i * g_i).
    g_w_norm = torch.sqrt(w_opt @ gram_matrix @ w_opt)
    scalar = (phi_sqrt / g_w_norm).item() if g_w_norm > 1e-8 else 0.0
    rho = (1.0 / num_tasks) + scalar * w_opt
    rho = rho.to(dtype=compute_dtype)

    logging.info(f"Optimal weights: {w_opt.cpu().numpy().round(4)}")

    logging.info("Step 3/3: Pass 2 - fused accumulation, SVD projection, and similarity stats...")
    
    new_state = OrderedDict()
    rank_pattern = peft_config.rank_pattern or {}
    default_rank = peft_config.r
    
    client_dot_products = {cid: 0.0 for cid in client_ids}
    client_norm_sq = {cid: 0.0 for cid in client_ids}
    consensus_norm_sq = 0.0
    
    for layer_name in lora_names:
        if layer_name not in layer_shapes: continue
        shape = layer_shapes[layer_name]
        num_elm = shape[0] * shape[1]
        
        key_A = f"{layer_name}.lora_A.weight"
        key_B = f"{layer_name}.lora_B.weight"

        # Restore the base parameters.
        A_old = global_model_state_before_round[key_A].to(device, dtype=compute_dtype)
        B_old = global_model_state_before_round[key_B].to(device, dtype=compute_dtype)
        W_old = B_old @ A_old 

        # Cache only this layer's client diffs for later dot products.
        current_layer_client_diffs = {} 

        d_layer_accum = torch.zeros(num_elm, device=device, dtype=compute_dtype)
        has_update = False

        # Collect and accumulate this layer's updates.
        for i, cid in enumerate(client_ids):
            upd = client_update_deltas[cid]
            if key_A in upd and key_B in upd:
                has_update = True
                dA = upd[key_A].to(device, dtype=compute_dtype)
                dB = upd[key_B].to(device, dtype=compute_dtype)
                
                W_new = (B_old + dB) @ (A_old + dA)
                diff = (W_new - W_old).flatten()
                
                if abs(rho[i]) > 1e-6:
                    d_layer_accum.add_(diff, alpha=rho[i])
                
                diff_cpu = diff.cpu().float()
                current_layer_client_diffs[cid] = diff_cpu
                
                client_norm_sq[cid] += torch.sum(diff_cpu ** 2).item()

                del W_new, dA, dB, diff
            else:
                current_layer_client_diffs[cid] = torch.zeros(num_elm, dtype=torch.float32)

        if not has_update:
            new_state[key_A] = global_model_state_before_round[key_A]
            new_state[key_B] = global_model_state_before_round[key_B]
            del current_layer_client_diffs, d_layer_accum, W_old, A_old, B_old
            continue

        d_layer_float = d_layer_accum.float()
        
        W_target = W_old.float() + (global_learning_rate * d_layer_float.view(shape))
        
        module_key = next((k for k in rank_pattern.keys() if k in layer_name), None)
        r = rank_pattern.get(module_key, default_rank) if module_key else default_rank
        
        try:
            U, S, V = torch.pca_lowrank(W_target, q=r+10, center=False, niter=4)
            sqS = torch.sqrt(S[:r].clamp(min=1e-8))
            new_A = (torch.diag(sqS) @ V[:, :r].T)
            new_B = (U[:, :r] @ torch.diag(sqS))
            
            new_state[key_A] = new_A.to(global_model_state_before_round[key_A].dtype)
            new_state[key_B] = new_B.to(global_model_state_before_round[key_B].dtype)
            
            W_final_projected = new_B.float().to(device) @ new_A.float().to(device)
            diff_final_layer = (W_final_projected - W_old.float()).flatten().cpu()
            
            consensus_norm_sq += torch.sum(diff_final_layer ** 2).item()
            
            for cid in client_ids:
                client_vec = current_layer_client_diffs[cid]
                dot_val = torch.dot(client_vec, diff_final_layer).item()
                client_dot_products[cid] += dot_val

            del W_final_projected, diff_final_layer, U, S, V, new_A, new_B

        except Exception as e:
            logging.warning(f"SVD failed for layer {layer_name}: {e}")
            new_state[key_A] = global_model_state_before_round[key_A]
            new_state[key_B] = global_model_state_before_round[key_B]

        del d_layer_accum, W_target, d_layer_float, W_old, A_old, B_old, current_layer_client_diffs

    t_aux_start = time.perf_counter()
    
  
    for k, v in global_model_state_before_round.items():
        if k not in new_state: 
            new_state[k] = v.to(device)
            
    set_peft_model_state_dict(model, new_state, "default")
    
    auxiliary_time_cost += (time.perf_counter() - t_aux_start)
    pure_time = (time.perf_counter() - total_start_time) - auxiliary_time_cost
    
    logging.info(f"*** Core algorithm time: {pure_time:.4f}s ***")
    
    return model
