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
    results_data = {"question": [], "answer": [], "contexts": [], "ground_truth": ground_truth}
    print("Running RAG chain for each question...")
    for query in tqdm(questions, desc="Processing questions"):
        results_data["question"].append(query)
        answer_result = chain_instance.invoke({"question": query})
        answer = answer_result.get("answer", answer_result) if isinstance(answer_result, dict) else answer_result
        results_data["answer"].append(answer)
        retriever_output = retriever_dict["hybrid"].invoke({"query": query})
        docs_list = retriever_output.get("documents", [])
        results_data["contexts"].append([doc.page_content for doc in docs_list])
    
    # Create the dataset from the dictionary
    dataset = Dataset.from_dict(results_data)
    print("First entry of the results:")
    first_entry = {
        "question": results_data["question"][0],
        "answer": results_data["answer"][0],
        "contexts": results_data["contexts"][0],
        "ground_truth": results_data["ground_truth"][0],
    }
    print(json.dumps(first_entry, indent=4))
    
    print("Cleaning answers to remove <think> sections...")
    from benchmarks.Ragas.evaluation import clean_thinks, evaluate_results, save_metrics, plot_heatmap
    cleaned_dataset = clean_thinks(dataset)
    
    print("Evaluating dataset with RAG metrics...")
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    result = evaluate_results(cleaned_dataset, evaluator_llm)
    
    print("Average evaluation metrics:")
    df_metrics = result.to_pandas()[['context_precision', 'context_recall', 'faithfulness', "answer_relevancy"]].mean()
    print(df_metrics)
    
    # Create directories if they don't exist
    os.makedirs("results/Ragas/metrics", exist_ok=True)
    os.makedirs("results/Ragas/images", exist_ok=True)
    
    # Save metrics results to a JSON file in results/Ragas/metrics
    output_json = os.path.join("results", "Ragas", "metrics", f"{args.ollama_model}_{args.retrieval_strategy}_results.json")
    save_metrics(result, output_json)
    
    # Plot heatmap and save the image file in results/Ragas/images
    image_filename = os.path.join("results", "Ragas", "images", f"{args.ollama_model}_{args.retrieval_strategy}.png")
    plot_heatmap(result, image_filename)
    
    print("RAG benchmark pipeline completed.")

if __name__ == "__main__":
    main() 