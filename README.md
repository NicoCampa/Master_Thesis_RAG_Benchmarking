# MasterThesisRAG
**Benchmarking Retrieval-Augmented Generation (RAG)**

This project benchmarks RAG systems on:
1. **Single-Hop QA**: Using Natural Questions
2. **Multi-Hop QA**: Using HotpotQA

### **Directory Structure**
- **configs/**: Experiment configurations
- **data/**: Datasets
- **database/**: Retrieval databases (Chroma, BM25)
- **scripts/**: Core scripts for benchmarking
- **results/**: Experiment results

### **Key Features**
- **Retrieval Methods**: Dense (DPR), Sparse (BM25), Hybrid
- **Generative Models**: Nemotron-mini, Llama 3.2, Gemma 2, Qwen 2.5
- **Evaluation Frameworks**:
  - **RAGAS**: Faithfulness, Answer Relevance, Context Precision.
  - **RGB**: Noise Robustness, Integration, Counterfactual Analysis.