<div align="center">

# RecoFed

**Federated LoRA fine-tuning with RecoFed aggregation and layer-aware rank allocation**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Ready-ee4c2c)
![LoRA](https://img.shields.io/badge/LoRA-PEFT-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

## Overview

RecoFed trains LoRA adapters for large language models in a federated setting. Each client trains locally, then the server aggregates client updates with the RecoFed aggregation method. The project also supports optional layer-importance-based LoRA rank allocation.

## Highlights

- 🔗 Federated LoRA training for instruction-tuned LLMs
- 🧩 RecoFed aggregation for client adapter updates
- 📊 Optional layer-importance-based heterogeneous LoRA ranks
- 🧪 GLUE-MNLI calibration for layer-importance estimation
- 🚀 Inference script for saved global adapters

## Quick Start

```bash
pip install -r requirements.txt
cd code
```

## Data Layout

Client data lives under `code/data/`.

```text
data/
├── dataset1/
│   ├── 2/
│   ├── 4/
│   ├── 6/
│   └── 8/
└── dataset2/
    ├── 2/
    └── 8/
```

Example client file:

```text
data/dataset1/8/local_training_0.json
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

Run RecoFed training:

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
  --aggregation_method "recofed"
```

Enable layer-aware LoRA ranks:

```bash
python main.py \
  --global_model "meta-llama/Llama-2-7b-hf" \
  --data_path "./data/dataset1" \
  --output_dir "./lora-7b-rank" \
  --num_clients 8 \
  --num_communication_rounds 20 \
  --client_selection_frac 1 \
  --prompt_template_name "alpaca_short" \
  --aggregation_method "recofed" \
  --use_importance_rank_allocation True \
  --target_avg_rank 8 \
  --rank_alloc_min 6 \
  --rank_alloc_max 16
```

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

## Project Structure

```text
code/
├── main.py                         # Federated LoRA training
├── GlobalModel_generated.py        # Inference
├── data/
│   └── data.py                     # GLUE-MNLI calibration loader
├── fed_utils/
│   ├── client.py                   # Local client training
│   ├── model_aggregation.py        # RecoFed and FedAvg aggregation
│   ├── rank_allocation.py          # Layer importance and rank allocation
│   └── evaluation.py               # Evaluation
├── templates/
└── utils/
```

## Notes

- Check `CUDA_VISIBLE_DEVICES`, Hugging Face offline settings, and mirror settings before running.
- Gated models such as LLaMA may require Hugging Face access.
- Training outputs are saved under `output_dir/num_clients/`, such as `./lora-7b/8/`.
- Model weights, logs, caches, and runtime outputs are excluded by `.gitignore`.
