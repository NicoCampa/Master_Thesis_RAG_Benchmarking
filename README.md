# MasterThesisRAG
**Benchmarking Retrieval-Augmented Generation (RAG)**

This project benchmarks RAG systems on:
1. **Single-Hop QA**: Using Natural Questions (NQ).
2. **Multi-Hop QA**: Using HotpotQA.

### **Directory Structure**
- **configs/**: Experiment configurations.
- **data/**: Datasets (NQ, HotpotQA).
- **database/**: Retrieval databases (Chroma, BM25).
- **scripts/**: Core scripts for benchmarking.
- **results/**: Experiment results.

### **Key Features**
- **Retrieval Methods**: Dense (DPR), Sparse (BM25), Hybrid.
- **Generative Models**: LLaMA (Ollama), OpenAI GPT-4.
- **Evaluation Frameworks**:
  - **RAGAS**: Faithfulness, Answer Relevance, Context Precision.
  - **RGB**: Noise Robustness, Integration, Counterfactual Analysis.