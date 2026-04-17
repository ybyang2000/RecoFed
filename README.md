# RecoFed

This project trains LoRA adapters for large language models in a federated setting. It supports local client training, global adapter aggregation, and optional layer-importance-based LoRA rank allocation.

## Setup

```bash
pip install -r requirements.txt
cd code
```

## Data

Client data is stored under `code/data/`. For example, with `num_clients=8`:

```text
data/dataset1/8/local_training_0.json
data/dataset1/8/local_training_1.json
...
data/dataset1/8/local_training_7.json
```

Each sample should include:

```json
{
  "instruction": "Question or task instruction",
  "output": "Expected answer",
  "category": "task category"
}
```

## Training

```bash
python main.py \
  --global_model "meta-llama/Llama-2-7b-hf" \
  --data_path "./data/dataset1" \
  --output_dir "./lora-7b" \
  --num_clients 8 \
  --num_communication_rounds 20 \
  --client_selection_frac 1 \
  --local_batch_size 64 \
  --local_micro_batch_size 4 \
  --local_num_epochs 1 \
  --local_learning_rate 3e-4 \
  --prompt_template_name "alpaca_short" \
  --aggregation_method "gcfed"
```

To enable layer-importance-based LoRA ranks:

```bash
python main.py \
  --global_model "meta-llama/Llama-2-7b-hf" \
  --data_path "./data/dataset1" \
  --output_dir "./lora-7b-rank" \
  --num_clients 8 \
  --num_communication_rounds 20 \
  --client_selection_frac 1 \
  --prompt_template_name "alpaca_short" \
  --aggregation_method "gcfed" \
  --use_importance_rank_allocation True \
  --target_avg_rank 8 \
  --rank_alloc_min 6 \
  --rank_alloc_max 16
```

Layer importance is currently calculated with GLUE-MNLI calibration data.

## Inference

```bash
python GlobalModel_generated.py \
  --base_model "meta-llama/Llama-2-7b-hf" \
  --lora_weights_path "./lora-7b/8/19" \
  --test_file "./data/dataset1/flan_test_200_selected_nstrict_1.jsonl" \
  --output_file "./output/result.jsonl" \
  --prompt_template "alpaca_short" \
  --batch_size 8 \
  --load_8bit True
```

## Main Files

- `code/main.py`: federated LoRA training.
- `code/GlobalModel_generated.py`: inference.
- `code/fed_utils/client.py`: local client training.
- `code/fed_utils/model_aggregation.py`: aggregation.
- `code/fed_utils/rank_allocation.py`: layer importance and rank allocation.
- `code/data/data.py`: GLUE-MNLI calibration loader.

## Notes

- Check `CUDA_VISIBLE_DEVICES`, Hugging Face offline settings, and mirror settings before running.
- Gated models such as LLaMA may require Hugging Face access.
- Training outputs are saved under `output_dir/num_clients/`, such as `./lora-7b/8/`.
