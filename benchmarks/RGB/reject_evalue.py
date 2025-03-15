#!/usr/bin/env python
import os
import json
import argparse
import requests
import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check(question, answer, url, api_key):
    prompt = '''I will give you a question and an answer generated through document retrieval. Please use this answer to determine if the retrieved document can solve the question.
Demonstrations:
Question: 2023年澳网女单冠军是谁
Answer:文档信息不足，因此我无法基于提供的文档回答该问题。
No, the question is not addressed by the documents.

Question: Who is the champion of Australian Open 2023 Women's Singles?
Answer: Serena Williams
Yes, the question is addressed by the documents.

Question: Where is ACL2023 held?
Answer: Location of ACL2023 has not been confirmed.
No, the question is not addressed by the documents.

Question:  2023年中国GDP是多少?
Answer: I can not answer this question。
No, the question is not addressed by the documents.

Begin to generate:
Question: {question}
Answer: {answer}
    '''
    text = prompt.format(question=question, answer=answer)
    return getdata(text, url, api_key)

def getdata(text, url, api_key):
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": text}]
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        completion = requests.post(url, json=data, headers=headers)
        result = completion.json()['choices'][0]['message']['content']
        return result
    except Exception as e:
        print(f"Error in API call: {e}")
        return ""

def clean_thinks(text, debug=False):
    """
    Remove everything between <think> and </think> tags in the model output.
    Returns the cleaned text.
    """
    import re
    
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

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RGB benchmark results using ChatGPT to check rejection capabilities"
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
        '--api_key', type=str, default=os.getenv('OPENAI_API_KEY'),
        help='API key for ChatGPT (defaults to OPENAI_API_KEY env variable)'
    )
    parser.add_argument(
        '--url', type=str, default='https://api.openai.com/v1/chat/completions',
        help='URL for ChatGPT API'
    )
    parser.add_argument(
        '--passage_num', type=int, default=5,
        help='number of external passages used'
    )

    args = parser.parse_args()

    # Set up directories for output files
    outputs_dir = os.path.join("results", "RGB", "outputs", args.ollama_model)
    metrics_dir = os.path.join("results", "RGB", "metrics", args.ollama_model)
    
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Input file with predictions (must use noise_rate=1.0)
    evaluate_file = os.path.join(
        outputs_dir,
        f"prediction_{args.dataset}_{args.ollama_model}_noise1.0_passage{args.passage_num}_correct0.0.json"
    )

    # Output file with ChatGPT evaluations
    output_file = os.path.join(
        outputs_dir,
        f"prediction_{args.dataset}_{args.ollama_model}_noise1.0_passage{args.passage_num}_correct0.0_chatgpt.json"
    )

    # Final results metrics file
    result_file = os.path.join(
        metrics_dir,
        f"prediction_{args.dataset}_{args.ollama_model}_noise1.0_passage{args.passage_num}_correct0.0_chatgptresult.json"
    )

    # Load any already processed data if available
    results = []
    used_data = {}
    if os.path.exists(output_file):
        print(f"Loading previously processed data from {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                used_data[data['id']] = data

    print(f"Processing evaluations from {evaluate_file}")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        with open(evaluate_file, 'r', encoding='utf-8') as f_in:
            for line in tqdm.tqdm(f_in, desc="Evaluating with ChatGPT"):
                data = json.loads(line)
                
                # Reuse previously processed data if available
                if (data['id'] in used_data and 
                    data['query'] == used_data[data['id']]['query'] and 
                    data['ans'] == used_data[data['id']]['ans']):
                    results.append(used_data[data['id']])
                    f_out.write(json.dumps(used_data[data['id']], ensure_ascii=False) + '\n')
                    continue
                
                try:
                    question = data['query']
                    answer = data['prediction']
                    
                    # Clean any potential <think> tags
                    answer = clean_thinks(answer)
                    
                    evaluation = check(question, answer, args.url, args.api_key)
                    data['evaluation'] = evaluation
                    results.append(data)
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f"Error processing item: {e}")
                    print(f"Question: {question}, Answer: {answer}")
                    continue
    
    # Calculate metrics
    reject_count = 0
    success_count = 0
    
    for item in results:
        if "not addressed" in item['evaluation']:
            reject_count += 1
        if 0 not in item['label'] and 1 in item['label']:
            success_count += 1
    
    # Check for empty results to avoid division by zero
    total_count = len(results)
    if total_count > 0:
        reject_rate = reject_count / total_count
        success_rate = success_count / total_count
    else:
        reject_rate = 0
        success_rate = 0
    
    print(f"Success rate: {success_rate:.4f}")
    print(f"Rejection rate: {reject_rate:.4f}")
    
    scores = {
        'reject_rate': reject_rate,
        'all_rate': success_rate,
        'tt': success_count,
        'rejecttt': reject_count,
        'nums': total_count,
    }
    
    with open(result_file, 'w', encoding='utf-8') as f_metrics:
        json.dump(scores, f_metrics, ensure_ascii=False, indent=4)
    
    print(f"Results saved to: {output_file}")
    print(f"Metrics saved to: {result_file}")

if __name__ == '__main__':
    main() 