import transformers
import os
from datasets import load_dataset
import logging
from collections import OrderedDict
import torch
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict, 
)
import numpy as np



class GeneralClient:
    def __init__(self, client_id, model, data_path, output_dir, args):
        self.client_id = client_id
        self.model = model
        self.args = args
        self.local_data_path = os.path.join(data_path, "local_training_{}.json".format(self.client_id))
        self.local_data = load_dataset("json", data_files=self.local_data_path, cache_dir=os.path.join(output_dir, ".cache"))
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))
        self.global_params_old = {}

    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size):
        original_columns = self.local_data["train"].column_names
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            self.local_train_dataset = local_train_val["train"].shuffle().map(
                generate_and_tokenize_prompt, remove_columns=original_columns
            )
            self.local_eval_dataset = local_train_val["test"].shuffle().map(
                generate_and_tokenize_prompt, remove_columns=original_columns
            )
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(
                generate_and_tokenize_prompt, remove_columns=original_columns
            )
            self.local_eval_dataset = None
        self.local_val_set_size = local_val_set_size

    def build_local_trainer(
        self, 
        tokenizer, 
        local_micro_batch_size, 
        gradient_accumulation_steps, 
        local_num_epochs, 
        local_learning_rate, 
        group_by_length, 
        ddp,
    ):
        """
        Create the local Trainer instance.
        """
        self.model.set_adapter('default')
        self.train_args = transformers.TrainingArguments(
                per_device_train_batch_size=local_micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=0,
                num_train_epochs=local_num_epochs,
                learning_rate=local_learning_rate,
                fp16=False,
                bf16=True,
                logging_steps=1,
                optim="adamw_torch",
                eval_strategy="steps" if self.local_val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=200 if self.local_val_set_size > 0 else None,
                save_steps=200,
                report_to="none",
                output_dir=self.local_output_dir,
                save_total_limit=1,
                # gradient_checkpointing=False,
                load_best_model_at_end=True if self.local_val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                dataloader_drop_last=False
            )
        trainer_class = transformers.Trainer
        trainer_extra_kwargs = {}


        self.local_trainer = trainer_class(
            model=self.model,
            args=self.train_args,
            train_dataset=self.local_train_dataset,
            eval_dataset=self.local_eval_dataset,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
            **trainer_extra_kwargs
        )
    
    def train(self):
        """
        Run local training.
        """
        if not hasattr(self, 'local_trainer'):
            raise RuntimeError("Trainer has not been built. Call build_local_trainer first.")
        
        logging.info("Client %d starts local training...", self.client_id)
        self.model.set_adapter('default')
        self.model.train()
        self.local_trainer.train()
        logging.info("Client %d finished local training.", self.client_id)

   

    #     return self.global_params_old
    def initiate_local_training(self, epoch=0, output_dir=str, client_id=0):
        """
        Create a true deep-copy snapshot before local training starts.
        """
        self.model.config.use_cache = False
        # Personalized training can load the previous round's model weights here.
        # if epoch > 0:
        #     single_output_dir = os.path.join(output_dir, str(epoch-1), "local_output_{}".format(client_id),
        #                                  "adapter_model.bin")
        #     single_weights = torch.load(single_output_dir)
        #     set_peft_model_state_dict(self.model, single_weights, "default")
     
        state_dict_before = get_peft_model_state_dict(self.model, adapter_name="default")
        
        self.global_params_old = OrderedDict()
        for key, tensor in state_dict_before.items():
            self.global_params_old[key] = tensor.clone().detach()
        
        test_param_name = 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'
        if test_param_name in self.global_params_old:
            norm_before = torch.norm(self.global_params_old[test_param_name])
            logging.info(f"[Debug] Client {self.client_id} before training (cloned): parameter '{test_param_name}' norm is {norm_before.item()}")
        else:
            logging.warning(f"[Debug] Parameter {test_param_name} was not found in the backup.")

        return self.global_params_old

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set, config):
        # Get the weights after training.
        params_dict_new = get_peft_model_state_dict(self.model, adapter_name='default')
        
        test_param_name_clean = 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'
        if test_param_name_clean in params_dict_new:
            norm_after_train = torch.norm(params_dict_new[test_param_name_clean])
            logging.info(f"[Debug] Client {self.client_id} after training: parameter '{test_param_name_clean}' norm is {norm_after_train.item()} (expected to change)")

        # Calculate the update delta.
        update_delta = OrderedDict()
        for name in params_dict_new.keys():
            if name in self.global_params_old:
                update_delta[name] = params_dict_new[name].cpu() - self.global_params_old[name].cpu()


        single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
        adapter_name = 'default'
        os.makedirs(single_output_dir, exist_ok=True)
        new_adapter_weight = get_peft_model_state_dict(
                self.model, adapter_name=adapter_name
            )
        torch.save(new_adapter_weight, single_output_dir + "/adapter_model.bin")
        config.save_pretrained(single_output_dir)
        
        set_peft_model_state_dict(self.model, self.global_params_old, "default")
        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        previously_selected_clients_set.add(self.client_id)
        weights_after_restore = get_peft_model_state_dict(self.model, adapter_name='default')
        if test_param_name_clean in weights_after_restore:
            norm_after_restore = torch.norm(weights_after_restore[test_param_name_clean])
            logging.info(f"[Debug] Client {self.client_id} after restore: parameter '{test_param_name_clean}' norm is {norm_after_restore.item()} (expected to match the pre-training value)")
            
            norm_before_train_value = torch.norm(self.global_params_old[test_param_name_clean]).item()
            if not np.isclose(norm_before_train_value, norm_after_restore.item()):
                logging.error(f"[Debug failure] Restored norm ({norm_after_restore.item()}) does not match pre-training norm ({norm_before_train_value}). Model state was not restored correctly.")
            else:
                 logging.info("[Debug success] Model state was restored correctly.")
        return self.model, local_dataset_len_dict, previously_selected_clients_set, self.client_id, update_delta
    
