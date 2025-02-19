# MasterThesisRAG

**Benchmarking Retrieval-Augmented Generation (RAG)**

This project provides a comprehensive benchmarking framework for RAG systems focused on:

1. **Single-Hop QA** – using Natural Questions  
2. **Multi-Hop QA** – using HotpotQA

## Table of Contents

- [Directory Structure](#directory-structure)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Directory Structure

- **configs/**: Experiment configurations  
- **data/**: Datasets  
- **database/**: Retrieval databases (e.g. Chroma, BM25)  
- **scripts/**: Core scripts for benchmarking  
- **results/**: Experiment results  

## Key Features

- **Retrieval Methods**: Dense (DPR), Sparse (BM25), Hybrid  
- **Generative Models**: Nemotron-mini, Llama 3.2, Gemma 2, Qwen 2.5  
- **Evaluation Frameworks**:
  - **RAGAS**: Faithfulness, Answer Relevance, Context Precision  
  - **RGB**: Noise Robustness, Integration, Counterfactual Analysis  

## Installation

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the benchmarking scripts from the `scripts/` directory. For example:

## Contributing

Contributions are welcome! For details on how to contribute, please see our [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the MIT License.

![Alt text](/images/pipelineRAG.png)