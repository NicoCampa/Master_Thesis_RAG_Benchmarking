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
import time
from openai import RateLimitError

def clean_thinks(dataset: Dataset) -> Dataset:
    """
    Remove everything between <think> and </think> tags in the answer text.
    Shows both original and cleaned answers for comparison.
    """
    df = pd.DataFrame(dataset)
    
    # Simple pattern to match exactly between <think> tags
    pattern = r'<think>.*?</think>'
    
    # Store original answer for comparison
    original_answer = df['answer'].iloc[0]
    
    # Print original answer first
    print("\nOriginal answer:")
    print("-" * 50)
    print(original_answer)
    print("-" * 50)
    
    # Clean answers
    df['answer'] = df['answer'].apply(lambda x: re.sub(pattern, '', x, flags=re.DOTALL).strip())
    
    # Store cleaned answer
    cleaned_answer = df['answer'].iloc[0]
    
    # Print cleaned answer
    print("\nCleaned answer:")
    print("-" * 50)
    print(cleaned_answer)
    print("-" * 50)
    
    # Print if any changes were made
    if original_answer != cleaned_answer:
        print("\nNote: Changes were made - <think> tags were removed")
    else:
        print("\nNote: No <think> tags were found - answer unchanged")
    
    return Dataset.from_pandas(df)

def evaluate_results(cleaned_dataset, evaluator_llm):
    """
    Run evaluation using ragas evaluation framework with simpler batching.
    """
    print("Starting evaluation with simplified batching...")
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
    import time
    import pandas as pd
    from datasets import Dataset
    
    # Define batch size and delay
    BATCH_SIZE = 10
    DELAY_SECONDS = 5
    
    # Convert to pandas for easier slicing
    df = cleaned_dataset.to_pandas()
    total_questions = len(df)
    
    print(f"Total questions to evaluate: {total_questions}")
    print(f"Processing in batches of {BATCH_SIZE} with {DELAY_SECONDS}s delay between batches")
    
    # Initialize a results dataframe to store all scores
    results_df = pd.DataFrame({
        'question': df['question'],
        'answer': df['answer'],
        'contexts': df['contexts'],
        'ground_truth': df['ground_truth'],
        'context_precision': float('nan'),
        'context_recall': float('nan'),
        'faithfulness': float('nan'),
        'answer_relevancy': float('nan')
    })
    
    # Process in batches
    for start_idx in range(0, total_questions, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total_questions)
        print(f"\nProcessing batch: questions {start_idx+1} to {end_idx}")
        
        # Create a subset of the original dataset
        batch_dataset = Dataset.from_pandas(df.iloc[start_idx:end_idx])
        
        # Set up metrics
        metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
        
        # Evaluate batch with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                batch_result = evaluate(
                    dataset=batch_dataset,
                    metrics=metrics,
                    llm=evaluator_llm
                )
                
                # Extract results into dataframe
                batch_df = batch_result.to_pandas()
                
                # Copy scores back to the main results dataframe
                for metric in ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']:
                    results_df.loc[start_idx:end_idx-1, metric] = batch_df[metric].values
                
                print(f"✓ Successfully evaluated batch")
                break
                
            except Exception as e:
                print(f"Error during batch evaluation (attempt {attempt+1}/{max_retries}): {str(e)}")
                
                if attempt < max_retries - 1:
                    wait_time = DELAY_SECONDS * (2 ** attempt)
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to evaluate batch after {max_retries} attempts")
                    # Don't raise - continue with other batches
        
        # Wait between batches to avoid rate limits
        if end_idx < total_questions:
            print(f"Waiting {DELAY_SECONDS} seconds before next batch...")
            time.sleep(DELAY_SECONDS)
    
    # Create final dataset with all results
    final_dataset = Dataset.from_pandas(results_df)
    
    # Run one final evaluation just to create a proper Result object
    # But with empty dataset so it's just for structure
    try:
        print("\nCreating final Result object...")
        empty_result = evaluate(
            dataset=Dataset.from_dict({
                'question': [],
                'answer': [],
                'contexts': [],
                'ground_truth': []
            }),
            metrics=metrics,
            llm=evaluator_llm
        )
        
        # Get the proper Result object representation and replace its dataset
        from ragas.evaluation import Result
        final_result = Result(empty_result._asdict())
        final_result.dataset = final_dataset
        
        print("Success: Final evaluation object created")
        return final_result
        
    except Exception as e:
        print(f"Error creating final Result object: {str(e)}")
        # Create a simple wrapper class as fallback
        class SimpleResult:
            def __init__(self, df):
                self.df = df
                
            def to_pandas(self):
                return self.df
                
            def _asdict(self):
                return {"dataset": final_dataset, "metrics": metrics}
                
        return SimpleResult(results_df)

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

def plot_radar_performance(result, image_filename):
    """
    Creates a clean, modern radar chart for model performance metrics.
    """
    # Extract model name and retrieval strategy from filename
    filename_parts = os.path.basename(image_filename).split('_')
    model_name = filename_parts[0]
    retrieval_strategy = filename_parts[1] if len(filename_parts) > 1 else ""
    
    # Format model name for display
    model_display = model_name.replace('-', ' ').replace(':', ' ')
    
    # Get metrics data
    df = result if isinstance(result, pd.DataFrame) else result.to_pandas()
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    display_names = ["Context\nPrecision", "Context\nRecall", "Faithfulness", "Answer\nRelevancy"]
    avg_scores = df[metrics].mean()
    
    # Set up colors
    background_color = '#ffffff'
    primary_color = '#006400'  # Dark green
    secondary_color = '#90EE90'  # Light green
    text_color = '#333333'
    grid_color = '#cccccc'
    
    # Create figure and polar axes
    fig = plt.figure(figsize=(12, 12), facecolor=background_color)
    ax = fig.add_subplot(111, polar=True)
    
    # Set background color
    ax.set_facecolor(background_color)
    
    # Number of metrics and angles
    N = len(metrics)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add the values
    values = avg_scores.tolist()
    values += values[:1]  # Close the loop
    
    # Draw the chart elements
    ax.plot(angles, values, 'o-', linewidth=3, color=primary_color, markersize=10)
    ax.fill(angles, values, color=secondary_color, alpha=0.4)
    
    # Draw circular grid lines
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], color=text_color, size=0)  # Hide tick labels
    
    # Draw grid lines but with very light colors
    ax.grid(True, color=grid_color, linestyle='-', linewidth=0.5, alpha=0.7)
    
    # Set y-axis limit
    ax.set_ylim(0, 1)
    
    # Add metric labels at each point
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  # Hide default labels, we'll add our own
    
    # Add custom metric labels and values around the chart
    for i, (display_name, angle, value) in enumerate(zip(display_names, angles[:-1], values[:-1])):
        # Calculate positions
        angle_deg = np.rad2deg(angle)
        
        # Add metric name
        ha = 'center'
        if angle_deg < 90 or angle_deg > 270:
            ha = 'left'
        elif 90 < angle_deg < 270:
            ha = 'right'
            
        # Adjust label position based on angle quadrant
        label_distance = 1.3
        ax.text(angle, label_distance, display_name, 
                size=14, fontweight='bold', ha=ha, va='center', 
                color=text_color)
        
        # Add value in box
        box_distance = 1.05  # Distance from center
        x = box_distance * np.cos(angle)
        y = box_distance * np.sin(angle)
        
        # Create text box
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor=primary_color, linewidth=2)
        ax.text(angle, value + 0.05, f"{value:.3f}", 
                size=14, fontweight='bold', ha='center', va='center', color=text_color,
                bbox=bbox_props)
    
    # Combine model name and retrieval strategy for title
    title = f"{model_display}"
    if retrieval_strategy:
        title += f"\n{retrieval_strategy}"
    
    # Add title
    ax.set_title(title, size=24, fontweight='bold', color=text_color, pad=30)
    
    # Add subtle credit text
    plt.figtext(0.95, 0.01, "Generated with RAGAS", ha='right', va='bottom', 
                fontsize=8, color='#999999')
    
    # Save the figure
    plt.savefig(image_filename, dpi=300, bbox_inches='tight', facecolor=background_color)
    plt.close()
    
    print(f"Radar chart saved to: {image_filename}")

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

def get_color_scheme(strategy):
    """
    Returns color scheme based on retrieval strategy.
    """
    schemes = {
        'dense': {
            'main': '#d35400',  # Strong orange
            'fill': '#ffeee6',  # Light orange
            'edge': '#a04000'   # Dark orange
        },
        'hybrid': {
            'main': '#0d472e',  # Strong green
            'fill': '#e1f5e9',  # Light green
            'edge': '#0a3622'   # Dark green
        },
        'sparse': {
            'main': '#5b2c6f',  # Strong purple
            'fill': '#f4e6f5',  # Light purple
            'edge': '#4a235a'   # Dark purple
        }
    }
    return schemes.get(strategy, schemes['hybrid'])  # Default to hybrid if strategy not found

def get_model_name_from_file(filename):
    """Extract model name from the output file path."""
    # Extract filename from path and remove extension
    base = os.path.basename(filename)
    parts = base.split('_')
    # For filenames like "model-name_retriever_radar_average.png"
    # Return everything before the retriever type
    if len(parts) >= 2:
        return parts[0]
    return "Unknown Model"

def get_retriever_from_file(filename):
    """Extract retriever type from the output file path."""
    # Extract filename from path and remove extension
    base = os.path.basename(filename)
    parts = base.split('_')
    # For filenames like "model-name_retriever_radar_average.png"
    if len(parts) >= 2:
        return parts[1]
    return "unknown"

def plot_average_metrics(result, output_file):
    """
    Creates a clean radar chart for model performance metrics with horizontal labels.
    """
    # Extract model name and retrieval strategy from filename
    filename_parts = os.path.basename(output_file).split('_')
    model_name = filename_parts[0]
    retrieval_strategy = filename_parts[1] if len(filename_parts) > 1 else ""
    
    # Format model name for display - putting retrieval strategy on same line
    model_display = f"{model_name} {retrieval_strategy}"
    
    # Get metrics data
    df = result if isinstance(result, pd.DataFrame) else result.to_pandas()
    
    # Define metrics in clockwise order starting from top
    metrics = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]
    display_names = ["Context Recall", "Context\nPrecision", "Faithfulness", "Answer\nRelevancy"]
    
    # Get values in the correct order
    avg_scores = df[metrics].mean().values
    
    # Set up colors
    background_color = '#ffffff'
    primary_color = '#006400'  # Dark green
    secondary_color = '#90EE90'  # Light green
    text_color = '#333333'
    grid_color = '#cccccc'
    
    # Create figure and polar axes
    fig = plt.figure(figsize=(12, 12), facecolor=background_color)
    ax = fig.add_subplot(111, polar=True)
    
    # Set background color
    ax.set_facecolor(background_color)
    
    # Number of metrics and angles - starting from the top
    N = len(metrics)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    theta = np.roll(theta, -N//4)  # Start from top
    
    # Close the polygon by appending first values
    theta_closed = np.append(theta, theta[0])
    values_closed = np.append(avg_scores, avg_scores[0])
    
    # Draw the chart elements
    ax.plot(theta_closed, values_closed, 'o-', linewidth=3, color=primary_color, markersize=10)
    ax.fill(theta_closed, values_closed, color=secondary_color, alpha=0.4)
    
    # Draw circular grid lines
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], color=text_color, size=0)  # Hide tick labels
    
    # Draw grid lines but with very light colors
    ax.grid(True, color=grid_color, linestyle='-', linewidth=0.5, alpha=0.7)
    
    # Set y-axis limit
    ax.set_ylim(0, 1)
    
    # Hide default axis tick labels
    ax.set_xticks(theta)
    ax.set_xticklabels([])
    
    # Add custom metric labels and values
    for i, (name, angle, value) in enumerate(zip(display_names, theta, avg_scores)):
        # Calculate label position
        label_distance = 1.2  # Base distance
        
        # Set alignment based on quadrant
        if i == 0:  # Top - Context Recall
            ha = 'center'
            va = 'bottom'
            label_distance = 1.05  # Moved closer to circle (was 1.15)
        elif i == 1:  # Right - Context Precision
            ha = 'left'
            va = 'center'
            label_distance = 1.3  # Unchanged
        elif i == 2:  # Bottom - Faithfulness
            ha = 'center'
            va = 'top'
            label_distance = 1.05  # Moved closer to circle (was 1.15)
        elif i == 3:  # Left - Answer Relevancy
            ha = 'right'
            va = 'center'
            label_distance = 1.3  # Unchanged
        
        # Add metric name
        ax.text(
            angle, label_distance,
            name, 
            size=18, 
            fontweight='bold',
            ha=ha, va=va, 
            color=text_color
        )
        
        # Value box positioning remains the same
        bbox_props = dict(
            boxstyle='round,pad=0.5', 
            facecolor='white', 
            edgecolor=primary_color, 
            linewidth=2
        )
        
        # Position value box at data point with slight offset
        value_offset = 0.05
        box_radius = value + value_offset
        
        ax.text(
            angle, box_radius, 
            f"{value:.3f}", 
            size=18, 
            fontweight='bold',
            ha='center', va='center', 
            color=text_color, 
            bbox=bbox_props
        )
    
    # Add single-line title with higher position
    ax.set_title(model_display, size=28, fontweight='bold', color=text_color, pad=50)  # Increased pad from 40 to 50
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=background_color)
    plt.close()
    
    print(f"Radar chart saved to: {output_file}")

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

def evaluate_results_by_category(cleaned_dataset, evaluator_llm):
    """
    Run evaluation using ragas framework and break down results by question type.
    """
    print("Starting evaluation with category breakdown...")
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
    import pandas as pd
    
    # First, get the standard evaluation results
    result = evaluate_results(cleaned_dataset, evaluator_llm)
    
    # Now, let's add category analysis
    print("\nBreaking down results by question type...")
    
    # Get the evaluation dataframe
    eval_df = result.to_pandas()
    
    # Get the original dataset with question types
    # Assume question_types are available in a column named 'question_type'
    dataset_df = cleaned_dataset.to_pandas()
    
    # If the dataset doesn't have question_type column, try to extract from qa.csv
    if 'question_type' not in dataset_df.columns:
        try:
            qa_df = pd.read_csv('./data/Ragas/qa.csv', sep=';')
            
            # Create a mapping from question to question_type
            question_to_type = dict(zip(qa_df['question'], qa_df['question_type']))
            
            # Add question_type to the evaluation dataframe
            dataset_df['question_type'] = dataset_df['question'].map(question_to_type)
            
            print(f"Successfully mapped {sum(dataset_df['question_type'].notna())} out of {len(dataset_df)} questions to their types")
        except Exception as e:
            print(f"Could not load question types from qa.csv: {str(e)}")
            return result
    
    # Now add the question types to the evaluation dataframe
    # We need to ensure the order is preserved
    question_types = []
    for q in eval_df['question']:
        found = False
        for idx, row in dataset_df.iterrows():
            if row['question'] == q:
                question_types.append(row.get('question_type', 'unknown'))
                found = True
                break
        if not found:
            question_types.append('unknown')
    
    eval_df['question_type'] = question_types
    
    # Group by question type and calculate mean for each metric
    metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
    category_metrics = eval_df.groupby('question_type')[metrics].mean()
    
    # Print the categorical results
    print("\n--- Results by Question Type ---")
    print(category_metrics.round(3))
    
    # Calculate count of each question type for weighted average validation
    type_counts = eval_df['question_type'].value_counts()
    print("\nQuestion type distribution:")
    for qtype, count in type_counts.items():
        print(f"  {qtype}: {count} questions ({count/len(eval_df)*100:.1f}%)")
        
    # Return enhanced result with category breakdowns
    result.category_metrics = category_metrics
    return result

def create_model_subdirectories(model_name):
    """
    Create organized subdirectories for a specific model's outputs.
    """
    import os
    
    # Define the base paths
    base_paths = [
        os.path.join("results", "Ragas", "images", model_name),
        os.path.join("results", "Ragas", "metrics", model_name),
        os.path.join("results", "Ragas", "outputs", model_name)
    ]
    
    # Create each directory
    for path in base_paths:
        os.makedirs(path, exist_ok=True)
        print(f"Ensured directory exists: {path}")
    
    return base_paths

def plot_metrics_by_category(result, model_name, retriever_type):
    """
    Create bar chart visualization for metrics broken down by question category.
    """
    if not hasattr(result, 'category_metrics'):
        print("No category metrics available for plotting")
        return
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Get the category metrics
    cat_metrics = result.category_metrics
    
    # Create model subdirectories
    image_dir = os.path.join("results", "Ragas", "images", model_name)
    os.makedirs(image_dir, exist_ok=True)
    
    # Set up plot configurations
    metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
    display_names = {
        'context_precision': 'Context Precision',
        'context_recall': 'Context Recall', 
        'faithfulness': 'Faithfulness',
        'answer_relevancy': 'Answer Relevancy'
    }
    
    # Bar chart comparison of question types
    plt.figure(figsize=(12, 8))
    
    # Set up the bar positions
    categories = cat_metrics.index
    x = np.arange(len(categories))
    width = 0.2  # Width of the bars
    
    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width - width*1.5, 
                cat_metrics[metric], 
                width, 
                label=display_names[metric])
    
    # Customize the chart
    plt.xlabel('Question Type', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title(f'Performance by Question Type: {model_name} {retriever_type}', fontsize=16)
    plt.xticks(x, categories, rotation=0, fontsize=12)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, metric in enumerate(metrics):
        for j, v in enumerate(cat_metrics[metric]):
            plt.text(j + i*width - width*1.5, v + 0.02, f'{v:.2f}', 
                    ha='center', va='bottom', fontsize=10, rotation=0)
    
    # Save the chart
    category_chart = os.path.join(image_dir, f"{retriever_type}_category_comparison.png")
    plt.tight_layout()
    plt.savefig(category_chart, dpi=300)
    plt.close()
    print(f"Saved category comparison chart to {category_chart}")

def save_metrics_by_category(result, model_name, retriever_type):
    """
    Save metrics broken down by question category to a JSON file.
    """
    import json
    import os
    
    if not hasattr(result, 'category_metrics'):
        print("No category metrics available for saving")
        return
    
    # Create model subdirectory
    metrics_dir = os.path.join("results", "Ragas", "metrics", model_name)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Create the metrics dictionary
    metrics_by_category = {
        'overall': result.to_pandas()[['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']].mean().to_dict(),
        'by_category': result.category_metrics.to_dict('index')
    }
    
    # Save to JSON
    filename = os.path.join(metrics_dir, f"{retriever_type}_results.json")
    with open(filename, 'w') as f:
        json.dump(metrics_by_category, f, indent=4)
    
    print(f"Saved category metrics to {filename}") 