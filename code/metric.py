from typing import Dict, List
from t5.evaluation import metrics
import tqdm
import json

def read_list(file, k):
    """
    Read grouped values from a JSONL file.

    Empty lines, malformed JSON lines, and records missing required keys are skipped.
    """
    dic = {}
    try:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: file not found - {file}")
        return dic

    for line in lines:
        line = line.strip()

        if line:
            try:
                data = json.loads(line)

                if 'category' in data and k in data:
                    if data['category'] not in dic:
                        dic[data['category']] = []

                    tmpd = data[k]
                    if isinstance(tmpd, str) and tmpd.endswith('</s>'):
                        tmpd = tmpd.split('</s>')[0]

                    dic[data['category']].append(tmpd)
                else:
                    print(f"Warning: skipping line missing 'category' or '{k}': '{line}'")

            except json.JSONDecodeError:
                print(f"Warning: skipping malformed JSON line: '{line}'")
                
    return dic

# Multi-rouge/multi-bleu. When there are multiple references, we want to get the
# rouge score that is highest. According to the authors, this is how it was done
# in the GEM paper.
# Source: https://github.com/google/BIG-bench/blob/main/bigbench/api/task_metrics.py
def rouge_fn(targets: List[List[str]], predictions: List[str]) -> Dict[str, float]:
    """Computes ROUGE by taking the max ROUGE-N per reference and N."""
    # Following strategy from https://www.aclweb.org/anthology/W04-1013/.
    # Identify best reference per response and ROUGE type.
    rouge_types = ["rouge1", "rouge2", "rougeLsum"]
    max_references = {rouge_type: [] for rouge_type in rouge_types}
    for targ_for_resp, resp in tqdm.tqdm(zip(targets, predictions), total=len(targets)):
        # Compute individual scores per example/ref pair.
        resp_scores = [metrics.rouge([t], [resp]) for t in targ_for_resp]
        # Find best scoring references for generated output and ROUGE type.
        for rouge_type in rouge_types:
            best_score_index = max(range(len(resp_scores)), key=lambda x: resp_scores[x][rouge_type])
            best_ref = targ_for_resp[best_score_index]
            # Add the reference to the new reference list.
            max_references[rouge_type].append(best_ref)
    # Compute metric for each of the reference lists for a ref type.
    results = {}
    for rouge_type in rouge_types:
        results[rouge_type] = metrics.rouge(max_references[rouge_type], predictions)[rouge_type]
    return results

def rouge(targets, predictions):
    if not targets or not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeLsum": 0.0}
    results = metrics.rouge(targets, predictions)
    return results

def get_result(targets, predictions, save):
    results = {}
    total_target = []
    total_pre = []
    for k in targets.keys():
        if k in predictions:
            result = rouge(targets[k], predictions[k])
            results[k] = result
            total_target.extend(targets[k])
            total_pre.extend(predictions[k])
        else:
            print(f"Warning: category '{k}' was not found in predictions; skipping it.")
    
    results['total'] = rouge(total_target, total_pre)
    print(results)
    with open(save, 'w') as f:
        f.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    targets_file = 'data/dataset1/flan_test_200_selected_nstrict_1.jsonl'
    predictions_file = 'output/result1.jsonl'
    eval_output_file = 'output/eval.json'

    print("--- Reading target file ---")
    targets = read_list(targets_file, 'output')

    print("--- Reading prediction file ---")
    predictions = read_list(predictions_file, 'answer')

    print("\n--- Computing evaluation scores ---")
    get_result(targets, predictions, eval_output_file)
    print(f"\nEvaluation complete. Results saved to {eval_output_file}")
