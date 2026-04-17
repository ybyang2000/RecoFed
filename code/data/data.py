import numpy as np
import random
import torch
from datasets import load_dataset


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_glue_mnli(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("glue", "mnli", split="train")
    valdata = load_dataset("glue", "mnli", split="validation_matched")

    label_names = ["entailment", "neutral", "contradiction"]

    def format_example(example):
        label = label_names[example["label"]] if example["label"] >= 0 else "unknown"
        return (
            f"Premise: {example['premise']}\n"
            f"Hypothesis: {example['hypothesis']}\n"
            f"Label: {label}"
        )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        sample = traindata[random.randint(0, len(traindata) - 1)]
        trainenc = tokenizer(
            format_example(sample),
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=seqlen,
        )
        trainloader.append((trainenc.input_ids, trainenc.attention_mask))

    validation_text = "\n\n".join(format_example(item) for item in valdata.select(range(min(256, len(valdata)))))
    valenc = tokenizer(
        validation_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=seqlen,
    )
    valenc = TokenizerWrapper(valenc.input_ids)
    return trainloader, valenc
    

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if "glue_mnli" in name or "mnli" in name:
        return get_glue_mnli(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"Unsupported calibration dataset: {name}. Only GLUE-MNLI is supported.")
