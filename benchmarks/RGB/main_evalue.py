#!/usr/bin/env python
import os
import json
import random
import math
import argparse
import subprocess
import yaml
import numpy as np
import tqdm
import re

def local_ollama_generate(prompt, model_name, debug=False):
    """
    Calls the local model using `ollama run <model_name>` passing the prompt via stdin.
    Captures and returns the standard output as the model's answer.
    """
    cmd = ["ollama", "run", model_name]
    try:
        process = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True
        )
        output = process.stdout.strip()
        if debug:
            print(f"[DEBUG] Return code: {process.returncode}")
            print(f"[DEBUG] STDOUT: {output}")
            print(f"[DEBUG] STDERR: {process.stderr.strip()}")
        return output
    except Exception as e:
        if debug:
            print("[ERROR] Exception while calling ollama:", e)
        return ""

def processdata(instance, noise_rate, passage_num, filename, correct_rate=0):
    """
    Process a data instance to extract the query, answer, and documents.
    This version mimics the original logic handling different dataset types.
    """
    query = instance['query']
    ans = instance['answer']

    neg_num = math.ceil(passage_num * noise_rate)
    pos_num = passage_num - neg_num

    if '_int' in filename:
        for i in instance['positive']:
            random.shuffle(i)
        print(len(instance['positive']))
        docs = [i[0] for i in instance['positive']]
        if len(docs) < pos_num:
            maxnum = max([len(i) for i in instance['positive']])
            for i in range(1, maxnum):
                for j in instance['positive']:
                    if len(j) > i:
                        docs.append(j[i])
                        if len(docs) == pos_num:
                            break
                if len(docs) == pos_num:
                    break
        neg_num = passage_num - len(docs)
        if neg_num > 0:
            negative = instance['negative'][:neg_num]
            docs += negative
    elif '_fact' in filename:
        correct_num = math.ceil(passage_num * correct_rate)
        pos_num = passage_num - neg_num - correct_num
        indexs = list(range(len(instance['positive'])))
        selected = random.sample(indexs, min(len(indexs), pos_num))
        docs = [instance['positive_wrong'][i] for i in selected]
        remain = [i for i in indexs if i not in selected]
        if correct_num > 0 and len(remain) > 0:
            docs += [instance['positive'][i] for i in random.sample(remain, min(len(remain), correct_num))]
        if neg_num > 0:
            docs += instance['negative'][:neg_num]
    else:
        if noise_rate == 1:
            neg_num = passage_num
            pos_num = 0
        else:
            if neg_num > len(instance['negative']):
                neg_num = len(instance['negative'])
                pos_num = passage_num - neg_num
            elif pos_num > len(instance['positive']):
                pos_num = len(instance['positive'])
                neg_num = passage_num - pos_num

        positive = instance['positive'][:pos_num]
        negative = instance['negative'][:neg_num]
        docs = positive + negative

    random.shuffle(docs)
    return query, ans, docs

def checkanswer(prediction, ground_truth):
    """
    Check if the ground truth is present in the prediction.
    Returns a list of labels: 1 for a match, 0 otherwise. For cases where the 
    ground truth is a list of lists, it performs the check on each sub‐list.
    """
    prediction = prediction.lower()
    if type(ground_truth) is not list:
        ground_truth = [ground_truth]
    labels = []
    for inst in ground_truth:
        flag = True
        if type(inst) == list:
            flag = False 
            inst = [i.lower() for i in inst]
            for i in inst:
                if i in prediction:
                    flag = True
                    break
        else:
            inst = inst.lower()
            if inst not in prediction:
                flag = False
        labels.append(int(flag))
    return labels

def clean_thinks(text, debug=False):
    """
    Remove everything between <think> and </think> tags in the model output.
    Returns the cleaned text.
    """
    # Store original text for comparison if debugging is enabled
    original_text = text
    
    # Remove content between <think> tags
    pattern = r'<think>.*?</think>'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL).strip()
    
    # Debug info if requested
    if debug and original_text != cleaned_text:
        print("\n[DEBUG] Removed <think> sections:")
        print("-" * 50)
        print("Original:", original_text)
        print("-" * 50)
        print("Cleaned:", cleaned_text)
        print("-" * 50)
    
    return cleaned_text

def predict(query, ground_truth, docs, model_name, system, instruction, dataset, debug=False):
    """
    Constructs a prompt using the system prompt and instruction. If docs exist,
    they are joined into a single string. The function then calls the local OLLAMA model
    and checks for the correctness of the predicted answer.
    """
    if len(docs) == 0:
        text = instruction.format(QUERY=query, DOCS='')
        prediction = local_ollama_generate(text, model_name, debug)
    else:
        docs_joined = '\n'.join(docs)
        text = instruction.format(QUERY=query, DOCS=docs_joined)
        text = system + "\n" + text
        prediction = local_ollama_generate(text, model_name, debug)
    
    # Clean <think> sections from the prediction
    prediction = clean_thinks(prediction, debug)
    
    if 'zh' in dataset:
        prediction = prediction.replace(" ", "")
    if '信息不足' in prediction or 'insufficient information' in prediction:
        labels = [-1]
    else:
        labels = checkanswer(prediction, ground_truth)
    factlabel = 0
    if '事实性错误' in prediction or 'factual errors' in prediction:
        factlabel = 1
    return labels, prediction, factlabel

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RGB benchmark with different models"
    )
    parser.add_argument(
        '--ollama_model', type=str, required=True,
        help='model name that was evaluated'
    )
    parser.add_argument(
        '--dataset', type=str, default='en',
        choices=['en', 'zh', 'en_int', 'zh_int', 'en_fact', 'zh_fact'],
        help='evaluation dataset'
    )
    parser.add_argument(
        '--noise_rate', type=float, default=0.0,
        help='rate of noisy passages'
    )
    parser.add_argument('--correct_rate', type=float, default=0.0, help='rate of correct passages')
    parser.add_argument('--factchecking', type=bool, default=False, help='whether to perform fact checking')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    parser.add_argument('--passage_num', type=int, default=5, help='number of passages to use')
    args = parser.parse_args()

    model_name = args.ollama_model
    noise_rate = args.noise_rate
    passage_num = args.passage_num

    # Load dataset from data/RGB folder
    dataset_file = f"data/RGB/{args.dataset}.json"
    if not os.path.exists(dataset_file):
        print(f"Dataset file not found: {dataset_file}")
        return

    instances = []
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            instances.append(json.loads(line))

    # Determine base result folder based on language (only English supported now)
    resultpath = 'result-en'

    # Embedded YAML configurations (only English)
    instruction_yaml = """
en:
  system: "You are an accurate and reliable AI assistant that can answer questions with the help of external documents. Please note that external documents may contain noisy or factually incorrect information. If the document contains the correct answer, you will provide an accurate answer. If the document does not contain the answer, you will generate 'I can not answer the question because of the insufficient information in documents.'. If there are inconsistencies with the facts in some of the documents, please respond 'There are factual errors in the provided documents.' and provide the correct answer."
  instruction: "Document:\\n{DOCS} \\n\\nQuestion:\\n{QUERY}"
"""
    instruction_fact_yaml = """
en:
  system: "You are an accurate and reliable AI assistant that can answer questions with the help of external documents. If some documents contain factual errors, first mention 'There are factual errors in the provided documents.' then provide the correct answer based on the accurate ones."
  instruction: "Document:\\n{DOCS} \\n\\nQuestion:\\n{QUERY}"
"""

    # Load the prompt configuration from the embedded YAML (only English supported)
    if args.factchecking:
        prompt = yaml.load(instruction_fact_yaml, Loader=yaml.FullLoader)['en']
        resultpath = os.path.join(resultpath, 'fact')
    else:
        prompt = yaml.load(instruction_yaml, Loader=yaml.FullLoader)['en']

    system = prompt['system']
    instruction = prompt['instruction']

    # Prepare output directories under results/RGB
    outputs_dir = os.path.join("results", "RGB", "outputs", args.ollama_model)
    metrics_dir = os.path.join("results", "RGB", "metrics", args.ollama_model)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # File paths remain the same, but now they'll go into model-specific folders
    evaluate_file = os.path.join(
        outputs_dir,
        f"prediction_{args.dataset}_{args.ollama_model}_noise{args.noise_rate}_passage{args.passage_num}_correct{args.correct_rate}.json"
    )

    output_file = os.path.join(
        metrics_dir,
        f"prediction_{args.dataset}_{args.ollama_model}_noise{args.noise_rate}_passage{args.passage_num}_correct{args.correct_rate}_metrics.json"
    )
    
    # Load already processed instances if the file exists
    useddata = {}
    if os.path.exists(evaluate_file):
        with open(evaluate_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                data = json.loads(line)
                useddata[data['id']] = data

    results_list = []
    with open(evaluate_file, 'w', encoding='utf-8') as f_out:
        for instance in tqdm.tqdm(instances, desc="Processing instances"):
            # Skip reprocessing if instance is already in useddata and matches
            if (instance['id'] in useddata and 
                instance['query'] == useddata[instance['id']]['query'] and 
                instance['answer'] == useddata[instance['id']]['ans']):
                results_list.append(useddata[instance['id']])
                f_out.write(json.dumps(useddata[instance['id']], ensure_ascii=False) + '\n')
                continue
            try:
                random.seed(2333)
                if passage_num == 0:
                    query = instance['query']
                    ans = instance['answer']
                    docs = []
                else:
                    query, ans, docs = processdata(instance, noise_rate, passage_num, args.dataset, args.correct_rate)
                label, prediction, factlabel = predict(query, ans, docs, model_name, system, instruction, args.dataset, args.debug)
                instance['label'] = label
                newinstance = {
                    'id': instance['id'],
                    'query': query,
                    'ans': ans,
                    'label': label,
                    'prediction': prediction,
                    'docs': docs,
                    'noise_rate': noise_rate,
                    'factlabel': factlabel
                }
                results_list.append(newinstance)
                f_out.write(json.dumps(newinstance, ensure_ascii=False) + '\n')
            except Exception as e:
                print("Error processing instance:", e)
                continue

    # Compute overall metric (tt) as in the original
    tt = 0
    for r in results_list:
        label = r['label']
        if noise_rate == 1 and label[0] == -1:
            tt += 1
        elif 0 not in label and 1 in label:
            tt += 1
    print("Success Rate:", tt / len(results_list))
    scores = {
        'all_rate': tt / len(results_list) if len(results_list) > 0 else 0,
        'noise_rate': noise_rate,
        'tt': tt,
        'nums': len(results_list),
    }
    if '_fact' in args.dataset:
        fact_tt = 0
        correct_tt = 0
        for r in results_list:
            if r['factlabel'] == 1:
                fact_tt += 1
                if 0 not in r['label']:
                    correct_tt += 1
        fact_check_rate = fact_tt / len(results_list) if len(results_list) > 0 else 0
        correct_rate_value = correct_tt / fact_tt if fact_tt > 0 else 0
        scores['fact_check_rate'] = fact_check_rate
        scores['correct_rate'] = correct_rate_value
        scores['fact_tt'] = fact_tt
        scores['correct_tt'] = correct_tt

    # Save metrics under results/RGB/metrics
    with open(output_file, 'w', encoding='utf-8') as f_metrics:
        json.dump(scores, f_metrics, ensure_ascii=False, indent=4)
    
    print(f"Outputs saved to: {evaluate_file}")
    print(f"Metrics saved to: {output_file}")

if __name__ == '__main__':
    main()