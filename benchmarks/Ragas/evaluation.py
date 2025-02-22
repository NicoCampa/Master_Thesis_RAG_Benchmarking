import re
import pandas as pd
import json
from datasets import Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import textwrap

def clean_thinks(dataset: Dataset) -> Dataset:
    """
    Remove everything between <think> and </think> tags in the answer text.
    """
    df = pd.DataFrame(dataset)
    pattern = r'<think>.*?</think>'
    df['answer'] = df['answer'].apply(lambda x: re.sub(pattern, '', x, flags=re.DOTALL).strip())
    print(f"\nCleaned answer:\n{df['answer'].iloc[0]}")
    return Dataset.from_pandas(df)

def evaluate_results(cleaned_dataset, evaluator_llm):
    """
    Run evaluation using ragas evaluation framework.
    """
    print("Starting evaluation of results with metrics...")
    from benchmarks.Ragas.ragas import evaluate
    from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
    result = evaluate(
        dataset=cleaned_dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=evaluator_llm
    )
    print("Evaluation completed.")
    return result

def save_metrics(result, filename):
    """
    Save the average evaluation metrics into a JSON file.
    """
    df = result.to_pandas()
    metrics_avg = df[['context_precision', 'context_recall', 'faithfulness', "answer_relevancy"]].mean().to_dict()
    with open(filename, 'w') as f:
        json.dump(metrics_avg, f, indent=4)
    print(f"Saved metrics averages to {filename}")

def plot_heatmap(result, image_filename):
    """
    Plot a heatmap of context_precision, context_recall, faithfulness, and answer_relevancy.
    Save the image to the provided filename.
    """
    df = result.to_pandas()
    heatmap_data = df[["context_precision", "context_recall", "faithfulness", "answer_relevancy"]]
    
    # Create figure with adjusted size
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    cmap = LinearSegmentedColormap.from_list("green_red", ["red", "green"])
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", linewidths=0.5, cmap=cmap)
    
    # Wrap long question texts
    questions = df["question"].tolist()
    wrapped_questions = ['\n'.join(textwrap.wrap(q, width=50)) for q in questions]
    
    # Set y-tick labels with wrapped text
    plt.yticks(ticks=range(len(wrapped_questions)), 
              labels=wrapped_questions, 
              rotation=0,
              va='center')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(image_filename), exist_ok=True)
    
    # Save figure
    plt.savefig(image_filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved evaluation heatmap image to {image_filename}") 