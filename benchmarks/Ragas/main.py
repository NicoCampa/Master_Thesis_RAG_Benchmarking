#!/usr/bin/env python
import warnings

# Filter all deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Filter specific LangChain warnings
try:
    from langchain.warnings import LangChainDeprecationWarning
    warnings.filterwarnings('ignore', category=LangChainDeprecationWarning)
except ImportError:
    pass

# Filter specific messages
warnings.filterwarnings('ignore', message=".*deprecated.*")
warnings.filterwarnings('ignore', message=".*will be removed.*")
warnings.filterwarnings('ignore', message=".*Please replace deprecated imports.*")
warnings.filterwarnings('ignore', message=".*As of langchain-core.*")
warnings.filterwarnings('ignore', message=".*You can use the langchain cli.*")
warnings.filterwarnings('ignore', message=".*For example, replace imports.*")
warnings.filterwarnings('ignore', message=".*pydantic.*")

from dotenv import load_dotenv
load_dotenv()
import os
import argparse
import json
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import re

def main():
    parser = argparse.ArgumentParser(description="Run RAG benchmark pipeline.")
    parser.add_argument(
        "--ollama_model",
        type=str,
        default="deepseek-r1:1.5b",
        help="Ollama model name to use.",
    )
    parser.add_argument(
        "--retrieval_strategy",
        type=str,
        default="dense",
        choices=["dense", "sparse", "hybrid"],
        help="Retrieval strategy to use.",
    )
    args = parser.parse_args()
    print(f"Selected Ollama Model: {args.ollama_model}, Retrieval Strategy: {args.retrieval_strategy}")
    
    # Check for API key in the environment
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return
    
    print("Starting RAG benchmark pipeline...")
    print("Loading documents and splitting into chunks...")
    from benchmarks.Ragas.data_loader import load_documents, load_qa
    docs, chunks = load_documents()
    print(f"Loaded {len(docs)} documents and created {len(chunks)} chunks.")
    
    print("Setting up retrievers...")
    from benchmarks.Ragas.retrievers import setup_retrievers
    retriever_dict = setup_retrievers(chunks)
    
    # Initialize model globally before chain creation
    from langchain_ollama.llms import OllamaLLM
    model = OllamaLLM(model=args.ollama_model)
    
    print("Preparing the RAG chain...")
    from benchmarks.Ragas.chain import rag_chain
    chain_instance = rag_chain(retriever_dict, args.retrieval_strategy, model=model)
    
    print("Loading QA pairs from CSV...")
    questions, ground_truth = load_qa()
    print(f"Loaded {len(questions)} questions.")
    
    # Running chain for each question with a progress bar
    all_questions = []
    all_answers = []
    all_contexts = []
    detailed_output = []  # For JSON output
    
    print("Running RAG chain for each question...")
    for i, query in enumerate(tqdm(questions, desc="Processing questions")):
        all_questions.append(query)
        
        # Get answer from chain
        answer_result = chain_instance.invoke({"question": query})
        answer = answer_result.get("answer", answer_result) if isinstance(answer_result, dict) else answer_result
        all_answers.append(answer)
        
        # Get contexts using the selected retrieval strategy
        retriever_output = retriever_dict[args.retrieval_strategy].invoke({"query": query})
        
        # Extract documents with proper error handling
        if isinstance(retriever_output, dict) and "documents" in retriever_output:
            docs_list = retriever_output.get("documents", [])
        else:
            # Fallback
            print(f"Warning: Unexpected retriever output format: {type(retriever_output)}")
            docs_list = []
        
        # Format contexts for RAGAS - this is key
        if not docs_list:
            print(f"Warning: No documents retrieved for query: '{query[:50]}...'")
            all_contexts.append(["No relevant context found."])
        else:
            # Each context is stored as a list of strings (RAGAS format)
            context_texts = [doc.page_content for doc in docs_list]
            all_contexts.append(context_texts)
        
        # Add entry for JSON output
        detailed_output.append({
            "id": i,
            "question": query,
            "answer": answer,
            "ground_truth": ground_truth[i],
            "context": "\n\n".join([doc.page_content for doc in docs_list or []]) if docs_list else "No relevant context found."
        })
    
    # Create output directory and save detailed JSON
    os.makedirs("results/Ragas/outputs", exist_ok=True)
    output_json_path = os.path.join("results", "Ragas", "outputs", f"{args.ollama_model}_{args.retrieval_strategy}_responses.json")
    with open(output_json_path, 'w') as f:
        json.dump(detailed_output, f, indent=4)
    print(f"Saved detailed responses to {output_json_path}")
    
    # Create dataset with the correctly formatted data
    ragas_data = {
        "question": all_questions,
        "answer": all_answers,
        "contexts": all_contexts,
        "ground_truth": ground_truth
    }
    dataset = Dataset.from_dict(ragas_data)
    
    print("First entry of the results:")
    first_entry = {
        "question": all_questions[0],
        "answer": all_answers[0],
        "contexts": all_contexts[0],
        "ground_truth": ground_truth[0],
    }
    print(json.dumps(first_entry, indent=4))
    
    print("Cleaning answers to remove <think> sections...")
    from benchmarks.Ragas.evaluation import clean_thinks, evaluate_results, save_metrics, plot_average_metrics
    cleaned_dataset = clean_thinks(dataset)
    
    # First ensure model name is properly formatted for directory naming
    model_dir_name = re.sub(r'[^\w\-]', '_', args.ollama_model)

    # Create model subdirectories at the beginning of the evaluation section
    print("Creating organized output directories...")
    from benchmarks.Ragas.evaluation import create_model_subdirectories
    create_model_subdirectories(model_dir_name)

    # Update the evaluation section to use the new functions
    print("Evaluating dataset with RAG metrics by category...")
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    from benchmarks.Ragas.evaluation import (
        evaluate_results_by_category, 
        plot_metrics_by_category, 
        save_metrics_by_category,
        plot_average_metrics
    )

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    result = evaluate_results_by_category(cleaned_dataset, evaluator_llm)

    print("Average evaluation metrics:")
    df_metrics = result.to_pandas()[['context_precision', 'context_recall', 'faithfulness', "answer_relevancy"]].mean()
    print(df_metrics)

    # Save metrics and visualization with new organization
    save_metrics_by_category(result, model_dir_name, args.retrieval_strategy)
    plot_metrics_by_category(result, model_dir_name, args.retrieval_strategy)

    # Plot overall radar chart in the model subdirectory
    image_filename = os.path.join("results", "Ragas", "images", model_dir_name, f"{args.retrieval_strategy}_radar_average.png")
    plot_average_metrics(result, image_filename)

    # Save detailed JSON outputs to the outputs directory
    detailed_output_path = os.path.join("results", "Ragas", "outputs", model_dir_name, f"{args.retrieval_strategy}_responses.json")
    os.makedirs(os.path.dirname(detailed_output_path), exist_ok=True)
    with open(detailed_output_path, 'w') as f:
        json.dump(detailed_output, f, indent=4)
    print(f"Saved detailed responses to {detailed_output_path}")
    
    print("RAG benchmark pipeline completed.")

if __name__ == "__main__":
    main() 