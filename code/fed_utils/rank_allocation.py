import torch
import logging
import numpy as np
from CKA import cka
import json

def get_feature_map(args, model, tokenizer, device=torch.device("cuda:0"), dataloader=None):
    """
    Capture each transformer layer's output feature map with forward hooks.
    """
    model.config.use_cache = False
    layers = get_layers(model)
    
    # Preallocate storage for each layer's output feature map.
    features = [
        torch.zeros(
            args.nsamples, 
            args.seqlen, 
            model.config.hidden_size, 
            dtype=next(iter(model.parameters())).dtype
        ) for _ in layers
    ]
    
    hooks = []
    nsamples_processed = 0

    def make_hook(i):
        def hook_fn(module, input, output):
            # This storage path expects one sample per batch.
            if nsamples_processed < args.nsamples:
                features[i][nsamples_processed] = output[0][0].detach().cpu()
        return hook_fn

    for i, layer in enumerate(layers):
        hook = layer.register_forward_hook(make_hook(i))
        hooks.append(hook)

    print("Running forward passes with hooks to capture feature maps...")
    
    for batch in dataloader:
        if nsamples_processed >= args.nsamples:
            break
        
        input_ids = batch[0][0:1].to(device)
        attention_mask = batch[1][0:1].to(device)
        
        # Create position_ids for LLaMA/Qwen-style models.
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        
        try:
            model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        except Exception as e:
            print(f"Forward pass failed: {e}")
            pass
        
        nsamples_processed += 1

    for hook in hooks:
        hook.remove()
        
    return features


def get_layers(model):
    """
    Get the transformer layer list from either a base model or a PEFT wrapper.
    """
    logging.info(f"--> [get_layers] Looking for layers in model type: {type(model)}")
    
    if hasattr(model, "get_base_model"):
        logging.info("--> [get_layers] PEFT model detected; unwrapping base model...")
        base_model = model.get_base_model()
    else:
        base_model = model
    
    logging.info(f"--> [get_layers] Base model type after unwrapping: {type(base_model)}")

    # Path A: Llama, Qwen, Mistral, and most modern decoder-only models.
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        logging.info("--> [get_layers] Found layers at .model.layers.")
        return base_model.model.layers
    
    # Path B: OPT, GPT-2, and similar models.
    elif hasattr(base_model, "model") and hasattr(base_model.model, "decoder") and hasattr(base_model.model.decoder, "layers"):
        logging.info("--> [get_layers] Found layers at .model.decoder.layers.")
        return base_model.model.decoder.layers
        
    # Path C: fallback for models with layers directly on the base object.
    elif hasattr(base_model, "layers"):
        logging.info("--> [get_layers] Found layers directly on the base model.")
        return base_model.layers
        
    else:
        error_msg = f"Could not find a '.layers' attribute in model type {type(base_model)}."
        logging.error(error_msg)
        raise AttributeError(error_msg)


def calculate_importance_from_features(args, model, tokenizer, device=torch.device("cuda:0"), 
                                       calib_dataloader=None):
    """
    Compute token-level feature importance with token subsampling to avoid CPU OOM.
    """
    feature = get_feature_map(args, model, tokenizer, device=device, dataloader=calib_dataloader)

    max_tokens = 4096
    
    N_total = feature[0].shape[0] * feature[0].shape[1]
    
    if N_total > max_tokens:
        print(f"Total token count is {N_total}; subsampling {max_tokens} tokens for CKA to avoid OOM...")
        torch.manual_seed(42) 
        indices = torch.randperm(N_total)[:max_tokens]
    else:
        indices = torch.arange(N_total)

    for i in range(len(feature)):
        flat_feat = feature[i].view(-1, model.config.hidden_size)
        feature[i] = flat_feat[indices]

    # Calculate the similarity matrix using CKA
    print("Calculating similarity matrix...")
    similar_matrix = torch.zeros(len(feature), len(feature))
    for i in range(len(feature)):
        for j in range(i, len(feature)): # Optimization: only compute upper triangle
            with torch.no_grad():
                sim = cka.cka(cka.gram_linear(feature[i].float()), cka.gram_linear(feature[j].float()))
                similar_matrix[i, j] = sim
                similar_matrix[j, i] = sim

    # Convert similarity to importance scores
    def sum_list_except_self(row, self_idx):
        return torch.sum(row) - row[self_idx]

    temp = [sum_list_except_self(similar_matrix[i], i) for i in range(len(feature))]
    total_similarity_sum = sum(temp)
    
    # Normalize similarities
    normalized_similarity = [x / total_similarity_sum for x in temp]
    print(f"normalized_similarity: {normalized_similarity}")
    beta = 5
    important = [torch.exp(-1 * beta * s) for s in normalized_similarity]
    
    important = np.array([t.numpy() for t in important]) 
    # Log the calculated scores
    importance_scores_dict = {f"layers.{i}": float(score) for i, score in enumerate(important)}
    print("--- [Pruning] Calculated Layer Importance Scores ---")
    print(json.dumps(importance_scores_dict, indent=4))
    print("--------------------------------------------------")

    # Clear memory-intensive variables
    del feature
    del similar_matrix
    torch.cuda.empty_cache()
    return importance_scores_dict



def allocate_ranks_by_importance(importance_scores_dict, target_avg_rank, min_rank=4, max_rank=16):
    """
    Allocate ranks by layer importance using a greedy budget assignment.
    """
    if not isinstance(importance_scores_dict, dict):
         logging.error(f"allocate_ranks_by_importance expected a dict, got {type(importance_scores_dict)}")
         return None

    # Sort by parsed layer index to keep layer order stable.
    try:
        sorted_items = sorted(importance_scores_dict.items(), key=lambda item: int(item[0].split('.')[-1]))
    except (ValueError, IndexError):
        logging.error("Could not parse layer indices from importance score keys.")
        return None
        
    importance_scores = np.array([score for key, score in sorted_items])

    num_layers = len(importance_scores)
    if num_layers == 0:
        logging.warning("Importance score list is empty; cannot allocate ranks.")
        return []
        
    total_rank_budget = int(target_avg_rank * num_layers)

    ranks = np.full(num_layers, min_rank, dtype=int)
    
    if np.sum(ranks) > total_rank_budget:
        logging.warning(f"Minimum rank sum ({min_rank}) exceeds the target budget. All ranks will be set to {min_rank}.")
        return ranks.tolist()

    remaining_budget = total_rank_budget - np.sum(ranks)
    logging.info(f"Total rank budget: {total_rank_budget}, remaining budget after initialization: {remaining_budget}")

    sorted_indices = np.argsort(-importance_scores)

    for idx in sorted_indices:
        if remaining_budget <= 0:
            break
        
        can_add = max_rank - ranks[idx]
        add_amount = min(can_add, remaining_budget)
        
        ranks[idx] += add_amount
        remaining_budget -= add_amount

    if remaining_budget > 0:
        logging.warning(f"All layers reached max rank, but {remaining_budget} budget remains.")

    logging.info("--- [Rank Allocation Result] ---")
    for i in range(num_layers):
        logging.info(f"  Layer {i}: assigned rank = {ranks[i]} (importance: {importance_scores[i]:.4f})")
    
    final_total_rank = np.sum(ranks)
    final_avg_rank = np.mean(ranks)
    logging.info(f"Final total rank: {final_total_rank} (budget: {total_rank_budget})")
    logging.info(f"Final average rank: {final_avg_rank:.4f} (target: {target_avg_rank})")
    logging.info("------------------------------------")
    
    return ranks.tolist()
