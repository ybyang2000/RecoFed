from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import logging
import json

def global_evaluation(model, data_files, generate_and_tokenize_prompt, batch_size, device):
    model.eval()
    data = load_dataset("json", data_files=data_files)
    val_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = val_data.with_format('torch')
    data_loader = DataLoader(val_data, batch_size=batch_size)
    loss = []
    
    # Track problematic samples.
    problem_samples = {
        "nan_losses": [],
        "inf_losses": [],
        "exception_batches": [],
        "input_has_nan": [],
        "input_has_inf": []
    }
    
    # Validate the evaluation data.
    logging.info(f"[Evaluation debug] Loaded {len(val_data)} evaluation samples")
    if len(val_data) == 0:
        logging.error("[Evaluation debug] Evaluation dataset is empty.")
        return float('nan')
    
    # Inspect the first sample.
    if len(val_data) > 0:
        sample = val_data[0]
        logging.info(f"[Evaluation debug] First sample keys: {sample.keys()}")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                logging.info(f"[Evaluation debug] {k}: shape={v.shape}, dtype={v.dtype}")
                if torch.isnan(v).any():
                    logging.error(f"[Evaluation debug] Input data {k} contains NaN.")
                if torch.isinf(v).any():
                    logging.error(f"[Evaluation debug] Input data {k} contains Inf.")
    
    nan_batch_count = 0
    total_batch_count = 0
    
    for i, inputs in enumerate(tqdm(data_loader)):
        total_batch_count += 1
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in inputs.items() if k in ['input_ids', 'attention_mask', 'labels']}
            
            # # Debug info
            # logging.info(f"[Evaluation debug] Batch {i} size: {len(batch.get('input_ids', []))}")
            # if 'input_ids' in batch:
            #     logging.info(f"[Evaluation debug] input_ids shape: {batch['input_ids'].shape}")
            #     logging.info(f"[Evaluation debug] input_ids range: [{batch['input_ids'].min()}, {batch['input_ids'].max()}]")
            #     # Check special values.
            #     if batch['input_ids'].max() >= 152000 or batch['input_ids'].min() < 0:
            #         logging.warning(f"[Evaluation debug] Batch {i} input_ids values are out of range.")
            
            # Check whether inputs contain invalid values.
            input_has_nan = False
            input_has_inf = False
            for k, v in batch.items():
                if torch.isnan(v).any():
                    logging.error(f"[Evaluation debug] Batch {i} input {k} contains NaN.")
                    problem_samples["input_has_nan"].append(i)
                    input_has_nan = True
                if torch.isinf(v).any():
                    logging.error(f"[Evaluation debug] Batch {i} input {k} contains Inf.")
                    problem_samples["input_has_inf"].append(i)
                    input_has_inf = True
            
            if input_has_nan or input_has_inf:
                nan_batch_count += 1
                continue
            
            try:
                output = model(**batch)
                batch_loss = output[0].cpu()
                
                # Check whether the loss is NaN or Inf.
                if torch.isnan(batch_loss):
                    logging.warning(f"Encountered NaN loss: {batch_loss}. Skipping this batch.")
                    nan_batch_count += 1
                    problem_samples["nan_losses"].append(i)
                    logging.info(f"[Debug] Model output details: {type(output)}, len={len(output) if hasattr(output, '__len__') else 'N/A'}")
                    if hasattr(output, 'loss'):
                        logging.info(f"[Debug] output.loss = {output.loss}")
                    
                    if 'input_ids' in batch:
                        input_ids = batch['input_ids'].cpu()
                        logging.info(f"[Debug] Problem batch {i} input_ids (first 10 tokens): {input_ids[0][:10] if len(input_ids) > 0 else 'N/A'}")
                    continue
                    
                if torch.isinf(batch_loss):
                    logging.warning(f"Encountered Inf loss: {batch_loss}. Skipping this batch.")
                    nan_batch_count += 1
                    problem_samples["inf_losses"].append(i)
                    if 'input_ids' in batch:
                        input_ids = batch['input_ids'].cpu()
                        logging.info(f"[Debug] Problem batch {i} input_ids (first 10 tokens): {input_ids[0][:10] if len(input_ids) > 0 else 'N/A'}")
                    continue
                    
                loss.append(batch_loss)
            except Exception as e:
                logging.warning(f"Evaluation exception: {e}. Skipping this batch.")
                nan_batch_count += 1
                problem_samples["exception_batches"].append(i)
                logging.exception(e)
                
                if 'input_ids' in batch:
                    input_ids = batch['input_ids'].cpu()
                    logging.info(f"[Debug] Exception batch {i} input_ids (first 10 tokens): {input_ids[0][:10] if len(input_ids) > 0 else 'N/A'}")
                continue
                
    logging.info(f"[Evaluation summary] Total batches: {total_batch_count}, NaN/exception batches: {nan_batch_count}, valid batches: {len(loss)}")
    
    # Save problematic sample info.
    try:
        with open("problem_samples.json", "w") as f:
            json.dump(problem_samples, f, indent=2)
        logging.info("[Evaluation debug] Problem sample info saved to problem_samples.json")
    except Exception as e:
        logging.error(f"[Evaluation debug] Failed to save problem sample info: {e}")
    
    # Log problem-sample stats by type.
    for problem_type, samples in problem_samples.items():
        if samples:
            logging.info(f"[Evaluation debug] {problem_type}: {samples}")
    
    if len(loss) == 0:
        logging.error("All evaluation batches returned invalid loss values (NaN/Inf).")
        return float('nan')
        
    final_loss = sum(loss) / len(loss)
    
    if torch.isnan(final_loss) or torch.isinf(final_loss):
        logging.error(f"Final evaluation loss is invalid: {final_loss}")
        return float('nan')
        
    return final_loss
