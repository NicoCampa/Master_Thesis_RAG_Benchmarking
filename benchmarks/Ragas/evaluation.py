import re
import pandas as pd
import json
from datasets import Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import textwrap
import numpy as np
from typing import Dict
import matplotlib.colors as mcolors

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
    from ragas import evaluate
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

def get_visualization_paths(base_filename):
    """
    Generate standardized paths for all visualizations.
    """
    # Get the directory and filename without extension
    dir_path = os.path.dirname(base_filename)
    base_name = os.path.splitext(os.path.basename(base_filename))[0]
    
    return {
        'heatmap': os.path.join(dir_path, f"{base_name}_heatmap.png"),
        'radar': os.path.join(dir_path, f"{base_name}_radar.png"),
        'split_heatmaps': [
            os.path.join(dir_path, f"{base_name}_heatmap_part{i}.png") 
            for i in range(1, 4)  # Assuming max 3 parts, adjust as needed
        ]
    }

def plot_heatmap(result, image_filename):
    """
    Plot a heatmap of evaluation metrics with improved visualization.
    """
    # Convert to DataFrame if it's not already one
    df = result if isinstance(result, pd.DataFrame) else result.to_pandas()
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    
    # Truncate and clean questions for better display
    questions = df["question"].tolist()
    max_question_length = 50  # Slightly shorter for better readability
    questions = [textwrap.fill(q[:max_question_length], width=40) + ('...' if len(q) > max_question_length else '') 
                for q in questions]
    
    # Calculate figure size based on number of questions
    n_questions = len(questions)
    fig_height = max(8, n_questions * 0.5)  # Increased height multiplier
    
    # Create figure with better proportions
    plt.figure(figsize=(12, fig_height))
    
    # Create custom colormap with better color transitions
    custom_cmap = LinearSegmentedColormap.from_list("custom", 
        ["#FF4444", "#FFFF44", "#44FF44"], N=256)
    
    # Create heatmap with improved styling
    ax = sns.heatmap(
        df[metrics],
        annot=True,
        fmt=".2f",
        cmap=custom_cmap,
        vmin=0,
        vmax=1,
        center=0.5,
        linewidths=1,
        square=True,  # Make cells square
        cbar_kws={
            "label": "Score",
            "orientation": "vertical",
            "pad": 0.02,
        }
    )
    
    # Customize axes with better formatting
    plt.xlabel("Evaluation Metrics", fontsize=12, labelpad=10)
    plt.ylabel("Questions", fontsize=12, labelpad=10)
    
    # Format x-axis labels
    plt.xticks(
        [x + 0.5 for x in range(len(metrics))],
        [m.replace("_", " ").title() for m in metrics],
        rotation=30,
        ha='right',
        fontsize=10
    )
    
    # Format y-axis labels with better spacing
    ax.set_yticks([y + 0.5 for y in range(len(questions))])
    ax.set_yticklabels(questions, rotation=0, ha='right', fontsize=9)
    
    # Add title with model name
    model_name = os.path.basename(image_filename).split('_')[0]
    plt.title(f"RAG Evaluation Metrics - {model_name}", pad=20, fontsize=14)
    
    # Adjust layout with better margins
    plt.tight_layout()
    
    # Save figure with high quality
    plt.savefig(image_filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_radar_charts(result, image_filename):
    df = result.to_pandas()
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    
    n_questions = len(df)
    n_cols = min(3, n_questions)
    n_rows = (n_questions + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, 5 * n_rows))
    
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        ax = plt.subplot(n_rows, n_cols, idx, projection='polar')
        
        values = row[metrics].values
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        # Add background grid with better visibility
        ax.grid(True, alpha=0.3)
        
        # Plot with better styling
        ax.plot(angles, values, 'o-', linewidth=2, label='Scores')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=8)
        ax.set_ylim(0, 1)
        
        # Add score labels
        for angle, value in zip(angles[:-1], values[:-1]):
            ax.text(angle, value, f'{value:.2f}', 
                   ha='center', va='bottom')
        
        # Truncate question for title
        title = textwrap.fill(row['question'][:50], width=30)
        ax.set_title(title + '...', pad=20, fontsize=10)
    
    plt.tight_layout(h_pad=3, w_pad=2)
    plt.savefig(image_filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_split_heatmaps(result, base_filename, questions_per_plot=15):
    """
    Split the heatmap into multiple plots for better readability.
    """
    # Convert to DataFrame if it's not already one
    df = result if isinstance(result, pd.DataFrame) else result.to_pandas()
    total_questions = len(df)
    num_plots = (total_questions + questions_per_plot - 1) // questions_per_plot
    
    for i in range(num_plots):
        start_idx = i * questions_per_plot
        end_idx = min((i + 1) * questions_per_plot, total_questions)
        
        # Create subplot for this chunk
        plot_heatmap(
            df.iloc[start_idx:end_idx], 
            base_filename.replace('.png', f'_part{i+1}.png')
        )

def plot_average_metrics(result, image_filename):
    """
    Plot average metrics using a radar chart with improved aesthetics.
    """
    # Convert to DataFrame if it's not already one
    df = result if isinstance(result, pd.DataFrame) else result.to_pandas()
    
    # Calculate averages
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    avg_scores = df[metrics].mean()
    
    # Set up the figure with white background
    plt.figure(figsize=(12, 12), facecolor='white')
    ax = plt.subplot(111, projection='polar')
    ax.set_facecolor('white')

    # Prepare the radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    values = np.concatenate((avg_scores, [avg_scores[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    # Plot data with improved visibility
    ax.fill(angles, values, color='#e8f5e9', alpha=0.25)
    ax.plot(angles, values, 'o-', linewidth=3, color='#1b5e20')
    
    # Add value annotations with better visibility
    for angle, value in zip(angles[:-1], values[:-1]):
        text_radius = value + 0.05
        # Add rounded box around values
        ax.text(angle, text_radius, f'{value:.3f}', 
                ha='center', va='center',
                color='black',
                fontsize=13,
                fontweight='bold',
                bbox=dict(
                    facecolor='white',
                    edgecolor='#2d6a4f',
                    boxstyle='round,pad=0.5',
                    alpha=0.9
                ))
    
    # Set chart properties with improved formatting
    ax.set_xticks(angles[:-1])
    metric_labels = [m.replace('_', '\n').title() for m in metrics]
    ax.set_xticklabels(metric_labels,
                       color='black',
                       fontsize=14,
                       fontweight='bold')
    
    # Remove radial labels and set limits
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    
    # Add subtle grid
    ax.grid(True, color='#cccccc', alpha=0.3)
    
    # Add title with improved formatting
    model_name = os.path.basename(image_filename).split('_')[0]
    plt.title('RAG Evaluation Metrics\n' + model_name, 
              color='black',
              pad=30,
              y=1.08,
              fontsize=18,
              fontweight='bold')
    
    # Save figure
    plt.savefig(image_filename, 
                bbox_inches='tight',
                dpi=300,
                facecolor='white',
                edgecolor='none')
    plt.close()

def plot_radar_metrics(metrics: Dict[str, float], model_name: str, save_path: str = None):
    # Set up the figure
    plt.figure(figsize=(10, 10), facecolor='white')
    ax = plt.subplot(111, projection='polar')
    ax.set_facecolor('white')

    # Get the metrics and angles
    categories = list(metrics.keys())
    values = list(metrics.values())
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    values += values[:1]
    angles += angles[:1]

    # Plot with gradient
    # Create a gradient from darker to lighter green
    colors = [(0.0, '#1a6b3b'),  # Darker green
              (1.0, '#44c28d')]  # Lighter green
    cmap = LinearGradientMap(colors)
    
    # Plot the data with gradient fill
    ax.fill(angles, values, alpha=0.25, color=cmap(0.6))
    ax.plot(angles, values, 'o-', linewidth=2, color=cmap(0.8))

    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Add grid
    ax.grid(color='gray', alpha=0.2)

    # Set title with better spacing
    plt.title('RAG Evaluation Metrics\n' + model_name.split('/')[-1], 
              pad=20, 
              y=1.05,
              fontsize=12)

    # Adjust layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return plt.gcf()

# Custom gradient map class
class LinearGradientMap:
    def __init__(self, colors):
        self.colors = colors

    def __call__(self, val):
        for i in range(len(self.colors)-1):
            if val <= self.colors[i+1][0]:
                c1 = np.array(mcolors.to_rgb(self.colors[i][1]))
                c2 = np.array(mcolors.to_rgb(self.colors[i+1][1]))
                alpha = (val - self.colors[i][0]) / (self.colors[i+1][0] - self.colors[i][0])
                return tuple(c1 * (1-alpha) + c2 * alpha)
        return mcolors.to_rgb(self.colors[-1][1])

def evaluate_dataset(dataset_path, evaluator_llm):
    """
    Main evaluation function.
    """
    # ... existing code ...

    # Get base filename for results
    base_filename = dataset_path.replace('.json', '')  # or however you want to name it
    metrics_file = f"{base_filename}_results.json"
    
    # Run evaluation
    result = evaluate_results(cleaned_dataset, evaluator_llm)
    
    # Save metrics and generate all visualizations
    save_metrics(result, metrics_file)
    plot_average_metrics(result, base_filename)  # This will generate all plots
    
    return result 