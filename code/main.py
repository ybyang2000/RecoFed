import os
custom_temp_dir = "/home/dell/tmp_cache" 

if not os.path.exists(custom_temp_dir):
    os.makedirs(custom_temp_dir)

os.environ["TMPDIR"] = custom_temp_dir
os.environ["TEMP"] = custom_temp_dir
os.environ["TMP"] = custom_temp_dir
import sys
import json
sys.setrecursionlimit(10000)
os.environ['HF_HUB_OFFLINE'] = '0'  # Explicitly enable online mode for huggingface_hub.
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Keep Transformers in offline mode.
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"  # Hugging Face mirror endpoint.
os.environ["WANDB_DISABLED"] = "true"  # Disable wandb.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Restrict the run to GPU 1.
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
from typing import List
from tqdm import tqdm
import fire
import gc
import logging
import re
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for server runs.
import torch
from data.data import get_loaders
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    get_peft_model_state_dict,
)
from fed_utils import calculate_importance_from_features, allocate_ranks_by_importance
from fed_utils import FedAvg, client_selection, global_evaluation, GeneralClient, RecoFed_aggregation_het_rank
import datasets
from utils.prompter import Prompter
import swanlab
from types import SimpleNamespace

datasets.utils.logging.set_verbosity_error()

# Configure logging for both file and console output.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("training_run.log", mode='w'),  # Overwrite the previous run log.
        logging.StreamHandler(sys.stdout)
    ]
)

def fl_finetune(
        # model/data params
        global_model: str = '',
        data_path: str = './data',
        output_dir: str = './lora-shepherd/',
        # FL hyperparamas
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 0.1,
        num_communication_rounds: int = 50,
        num_clients: int = 10,
        subsets: int = 0,
        # Local training hyperparams
        local_batch_size: int = 64,  # 64,
        local_micro_batch_size: int = 4,
        local_num_epochs: int = 1,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        val_data_path: str = "./data/dataset1/flan_test_200_selected_nstrict_1.jsonl",
        local_save_steps: int = 3,
        use_importance_rank_allocation: bool = False,  # Enable importance-based rank allocation.
        target_avg_rank: int = 8,                      # Target average rank after dynamic allocation.
        rank_alloc_min: int = 6,                       # Minimum allocated rank.
        rank_alloc_max: int = 16,                      # Maximum allocated rank.
        calibration_data_path: str = "./data/calibration_data.jsonl",
        nsamples: int = 130,  # Number of calibration samples for feature maps.
        cutoff_len: int = 512,
        local_model: bool = False,
        glocal: bool = False,
        # LoRA hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.01,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        aggregation_method: str = 'recofed',
        cagrad_c: float = 0.4,  # CAGrad hyperparameter c.
        # llm hyperparams
        train_on_inputs: bool = False,
        group_by_length: bool = False,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
        use_8bit: bool = False,  # Enable 8-bit quantization.
):
    args = SimpleNamespace(
        nsamples=nsamples,
        seqlen=cutoff_len,
    )
    # Set the initial LoRA rank.
    if use_importance_rank_allocation:
        initial_rank = target_avg_rank*2
        logging.info(f"Layer-importance rank allocation enabled. Round 0 uses uniform high rank: {initial_rank}")
    else:
        initial_rank = target_avg_rank if 'target_avg_rank' in locals() else lora_r 
        logging.info(f"Layer-importance rank allocation disabled. All rounds use fixed rank: {initial_rank}")

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"local_model:{local_model}\n"
            f"glocal:{glocal}\n"
        )
    assert (
        global_model
    ), "Please specify a --global_model, e.g. --global_modell='decapoda-research/llama-7b-hf'"
    data_path = os.path.join(data_path, str(num_clients))
    assert os.path.exists(data_path), "Please generate the data files for each client"

    # set up the global model & toknizer
    local_batch_size = int(local_batch_size)
    local_micro_batch_size = int(local_micro_batch_size)
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    
    swanlab.init(
        project="feddpa",
        config={
            "learning_rate": local_learning_rate,
            "architecture": "feddpa",
            "dataset": "dataset1",
            "method": aggregation_method
        }
    )

    # Load the base model.
    quantization_config = None
    if use_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        logging.info("8-bit quantization is ENABLED for training.")


    model = AutoModelForCausalLM.from_pretrained(
        global_model,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        global_model,
        use_fast=False,
        padding_side='left',
        trust_remote_code=True,
        add_eos_token=True,
        add_bos_token=True,
        local_files_only=False
    )


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.config.pretraining_tp = 1


    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"] if 'input' in data_point.keys() else None,
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        #print(tokenized_full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"] if 'input' in data_point.keys() else None,
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    final_lora_config = None
    if use_importance_rank_allocation:
        logging.info("--- [One-time analysis] Start calculating initial layer importance ---")

        logging.info(f"Creating a temporary PEFT model for importance calculation (uniform rank r={initial_rank})...")
        temp_peft_model = None
        try:
            temp_config = LoraConfig(r=initial_rank, lora_alpha=lora_alpha, target_modules=lora_target_modules, task_type="CAUSAL_LM")
            temp_peft_model = get_peft_model(model, temp_config)
            calib_nsamples = 8
            model_seqlen = cutoff_len
            calib_dataloader, _ = get_loaders(
                "glue_mnli", nsamples=calib_nsamples, seed=42,
                seqlen=model_seqlen, tokenizer=tokenizer
            )
            
            logging.info("Temporarily merging adapters for calculation...")
            merged_model = temp_peft_model.merge_and_unload()
            merged_model.eval()
            
            importance_scores = calculate_importance_from_features(
                args, merged_model, tokenizer, 
                device=torch.device("cuda:0"), calib_dataloader=calib_dataloader
            )
            temp_peft_model.unmerge_adapter()
            if importance_scores and len(importance_scores) > 0:
                new_ranks_list = allocate_ranks_by_importance(
                    importance_scores, target_avg_rank, rank_alloc_min, rank_alloc_max
                )
                logging.info("Calculation complete. Unloading the temporary adapter...")
                logging.info(f"Finding target modules matching '{lora_target_modules}'...")
                all_target_modules = [
                    name for name, module in model.named_modules()
                    if any(target in name for target in lora_target_modules) and isinstance(module, torch.nn.Linear)
                ]
                logging.info(f"Found {len(all_target_modules)} LoRA-applicable target modules.")


                if all_target_modules:
                    rank_pattern = {}
                    for module_name in all_target_modules:
                        match = re.search(r'\.layers\.(\d+)\.', module_name)
                        if match:
                            layer_idx = int(match.group(1))
                            if layer_idx < len(new_ranks_list):
                                rank_pattern[module_name] = new_ranks_list[layer_idx]
                    
                    logging.info(f"[Diagnostic] Created rank_pattern entries: {len(rank_pattern)}")
                    if rank_pattern:
                        logging.info(f"[Diagnostic] rank_pattern sample: {list(rank_pattern.items())[:3]}")

                    logging.info("Creating the final heterogeneous-rank LoRA config...")
                    final_lora_config = LoraConfig(
                        lora_alpha=lora_alpha,
                        target_modules=lora_target_modules, 
                        lora_dropout=lora_dropout,
                        bias="none",
                        task_type="CAUSAL_LM",
                        rank_pattern=rank_pattern 
                    )

        except Exception as e:
            logging.error(f"Layer importance and rank allocation failed: {e}", exc_info=True)
            logging.warning("Falling back to the default uniform-rank config.")
            final_lora_config = None

        finally:
            logging.info("Cleaning up the temporary PEFT model...")
            if temp_peft_model is not None:
                del merged_model, temp_peft_model, temp_config
            
            torch.cuda.empty_cache()
            gc.collect()
            logging.info("Temporary model cleanup complete.")

    # Fall back to a uniform-rank config when heterogeneous rank allocation is disabled or fails.
    if final_lora_config is None:
        logging.info(f"Creating a uniform-rank ({lora_r}) LoRA config...")
        final_lora_config = LoraConfig(
            base_model_name_or_path=global_model,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    logging.info("Applying the final PEFT config to the base model...")
    model = get_peft_model(model, final_lora_config)

    model.print_trainable_parameters()

    
    logging.info("\n" + "="*80)
    logging.info("========== Verifying Trainable LoRA Parameters ==========")

    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(f"Trainable parameter: {name} | shape: {param.shape}")

    previously_selected_clients_set = set()
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))
    for epoch in tqdm(range(num_communication_rounds), desc="Federated Rounds"):
        global_adapter_state_before_round = get_peft_model_state_dict(model, adapter_name="default")
        logging.info(f"\n==================== Communication Round {epoch} Start ====================")
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy, other_info=epoch, subsets=subsets)
        client_update_deltas = {}
        initial_params = {}
        for client_id in selected_clients_set:
            client_init_params = {}
            client = GeneralClient(client_id, model, data_path, output_dir, args)
            logging.info(f"\n--- Processing client {client_id} ---")
            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)

            logging.info(f"[Round {epoch}] Client {client_id}: training global adapter 'default'")
            client_init_params = client.initiate_local_training(epoch=epoch, output_dir=output_dir, client_id=client_id)
            client.build_local_trainer(tokenizer, local_micro_batch_size, gradient_accumulation_steps, local_num_epochs, local_learning_rate, group_by_length, ddp)
            initial_params[client_id] = client_init_params
            problem_batches = []
            
            original_training_step = client.local_trainer.training_step
            
            def training_step_with_nan_check(trainer_self, model, inputs, num_items_in_batch):
                try:
                    loss = original_training_step(model, inputs, num_items_in_batch) 
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        logging.warning(f"Client {client_id} encountered NaN or Inf loss: {loss}")
                        problem_batches.append({
                            "round": epoch,
                            "loss_value": str(loss.item())
                        })
                        return torch.tensor(1.0, requires_grad=True, device=loss.device)
                    return loss
                except Exception as e:
                    logging.error(f"Client {client_id} training failed: {e}", exc_info=True)
                    problem_batches.append({
                        "round": epoch,
                        "exception": str(e)
                    })
                    return torch.tensor(1.0, requires_grad=True, device=inputs[list(inputs.keys())[0]].device)
            
            import types
            client.local_trainer.training_step = types.MethodType(training_step_with_nan_check, client.local_trainer)

            client.train()

            if problem_batches:
                try:
                    problem_file = os.path.join(output_dir, f"client_{client_id}_problem_batches.json")
                    with open(problem_file, "w") as f:
                        json.dump(problem_batches, f, indent=2)
                    logging.info(f"Round {epoch}: saved problem batch info for client {client_id} to {problem_file}")
                except Exception as e:
                    logging.error(f"Round {epoch}: failed to save problem batch info for client {client_id}: {e}")

            model, local_dataset_len_dict, previously_selected_clients_set, _ ,update_delta= client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set,final_lora_config
            )
            client_update_deltas[client_id] = update_delta

   
        if aggregation_method.lower() == "recofed":
         
            model = RecoFed_aggregation_het_rank(
                model=model,
                global_model_state_before_round=global_adapter_state_before_round,
                client_update_deltas=client_update_deltas,
                c=cagrad_c,
                global_learning_rate=1.0,
                peft_config=final_lora_config
            )
        else:
            model, _ = FedAvg(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch)
            
        global_adapter_path = os.path.join(output_dir, str(epoch))
        os.makedirs(global_adapter_path, exist_ok=True)
        model.peft_config["default"].save_pretrained(global_adapter_path)

        default_weights = get_peft_model_state_dict(model, adapter_name="default")
        torch.save(default_weights, os.path.join(global_adapter_path, "adapter_model.bin"))

        logging.info(f"[Evaluation] Starting model evaluation for round {epoch}")
        
        eval_loss = global_evaluation(model, val_data_path, generate_and_tokenize_prompt, 1, 'cuda')
        logging.info(f'Communication round: {epoch}, Eval Loss: {eval_loss}')
       
            
        swanlab.log({"eval_loss": eval_loss if not (torch.isnan(torch.tensor(eval_loss)) or torch.isinf(torch.tensor(eval_loss))) else 0})
        
        logging.info(f"==================== Communication Round {epoch} End ====================")
if __name__ == "__main__":
    fire.Fire(fl_finetune)
