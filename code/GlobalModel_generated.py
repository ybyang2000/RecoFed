import os
# -------------------- [Environment Setup] --------------------
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# -------------------- [Imports] --------------------
import fire
import torch
import logging
import json
import sys
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from utils.prompter import Prompter

# -------------------- [Logging] --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("inference.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

# -------------------- [Device Detection] --------------------
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# -------------------- [Dataset] --------------------
class EvalDataset(Dataset):
    def __init__(self, file, prompter, tokenizer):
        self.prompter = prompter
        self.tokenizer = tokenizer
        with open(file, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx].strip()
        ques = json.loads(line)
        prompt = self.prompter.generate_prompt(ques['instruction'], None)
        return prompt, ques

# -------------------- [File Helper] --------------------
def writeFile(s, path):
    with open(path, 'a+', encoding='utf-8') as f1:
        f1.write(s + '\n')

# -------------------- [Main] --------------------
def main(
    base_model: str,
    lora_weights_path: str,
    test_file: str,
    output_file: str,
    load_8bit: bool = False,
    prompt_template: str = "alpaca",
    batch_size: int = 8,
):
    prompter = Prompter(prompt_template)

    # --- Model loading ---
    quantization_config = None
    if load_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        logging.info("8-bit quantization is ENABLED for inference.")

    torch_dtype = "auto"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    elif not load_8bit:
        torch_dtype = torch.float16

    logging.info(f"Loading base model: {base_model}")
   
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)


    # model = LlamaForCausalLM.from_pretrained(
    #     base_model,
    #     load_in_8bit=True,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )

    # tokenizer = LlamaTokenizer.from_pretrained(base_model)
    # tokenizer.pad_token_id = (
    #     0
    # )
    # tokenizer.padding_side = "left"

    if lora_weights_path:
        logging.info(f"Loading LoRA weights from: {lora_weights_path}")
        if os.path.isfile(lora_weights_path):
            lora_dir = os.path.dirname(lora_weights_path)
        else:
            lora_dir = lora_weights_path
            
        config_path = os.path.join(lora_dir, "adapter_config.json")
        if os.path.exists(config_path):
            model = PeftModel.from_pretrained(
                model,
                lora_dir,
                is_trainable=False,
            )
            logging.info("Successfully loaded LoRA weights using config.")
        else:
            logging.warning("adapter_config.json not found, attempting to load weights directly.")
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                is_trainable=False,
            )
            logging.info("Successfully loaded LoRA weights directly.")

    # --- Tokenizer configuration ---
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Keep model config aligned with the tokenizer.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id


    print(f"Tokenizer configured: pad_token='{tokenizer.pad_token}', bos_token='{tokenizer.bos_token}', eos_token='{tokenizer.eos_token}'")
    logging.info(f"Tokenizer configured: pad_token='{tokenizer.pad_token}', bos_token='{tokenizer.bos_token}', eos_token='{tokenizer.eos_token}'")


    model.eval()

    # --- Evaluation helper ---
    def evaluate(
        prompter,
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=128,
        input_ids=None,
        **kwargs,
    ):
        inputs = None
        if input_ids is None:
            prompt = prompter.generate_prompt(instruction, input)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
        else:
            input_ids = input_ids.to(device)
        
        attention_mask = None
        if inputs is not None:
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
            )
        
        sequences = generation_output.sequences
        outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        answers = [prompter.get_response(output) for output in outputs]
        return answers

    # --- Inference loop ---
    logging.info(f"===== Starting Inference on {test_file} =====")
    eval_dataset = EvalDataset(test_file, prompter, tokenizer)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    for prompts, original_data_batch in tqdm(dataloader, desc="Batch Inference"):
        responses = evaluate(prompter, instruction=None, input_ids=tokenizer(list(prompts), return_tensors="pt", padding=True).input_ids)
        
        for i in range(len(prompts)):
            original_data = {k: v[i] for k, v in original_data_batch.items()}
            response = responses[i]
            
            result = {
                "text": original_data['instruction'].item() if isinstance(original_data['instruction'], torch.Tensor) else original_data['instruction'],
                "answer": response,
                "category": original_data['category'].item() if isinstance(original_data['category'], torch.Tensor) else original_data['category']
            }
            writeFile(json.dumps(result, ensure_ascii=False), output_file)
    
    logging.info(f"Inference complete. Results saved to {output_file}")

if __name__ == "__main__":
    fire.Fire(main)
