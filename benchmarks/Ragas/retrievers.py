from langchain_community.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.retrievers import BM25Retriever
from dotenv import load_dotenv

def create_retrievers(chunks):
    # Create dense retriever using Chroma vector store
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(chunks, embedding)
    dense_retriever = vectorstore.as_retriever()
    # Create sparse retriever using BM25
    bm25_retriever = BM25Retriever.from_documents(chunks)
    return dense_retriever, bm25_retriever

class CustomHybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, dense_weight: float = 0.5, k: int = 5):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = 1 - dense_weight
        self.k = k

    def get_relevant_documents(self, query: str):
        # Retrieve documents from both retrievers
        dense_docs = self.dense_retriever.get_relevant_documents(query)
        sparse_docs = self.sparse_retriever.get_relevant_documents(query)
        # Combine scores using a linear combination
        doc_scores = {}
        for i, doc in enumerate(dense_docs):
            score = 1.0 - (i / len(dense_docs))
            doc_scores[doc.page_content] = self.dense_weight * score
        for i, doc in enumerate(sparse_docs):
            score = 1.0 - (i / len(sparse_docs))
            if doc.page_content in doc_scores:
                doc_scores[doc.page_content] += self.sparse_weight * score
            else:
                doc_scores[doc.page_content] = self.sparse_weight * score
        # Select top k documents
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        result_docs = []
        seen_content = set()
        all_docs = dense_docs + sparse_docs
        docs_lookup = {doc.page_content: doc for doc in all_docs}
        for content, _ in sorted_docs:
            if content not in seen_content and len(result_docs) < self.k:
                result_docs.append(docs_lookup[content])
                seen_content.add(content)
        return result_docs

    def invoke(self, inputs):
        query = inputs.get("query", "")
        documents = self.get_relevant_documents(query)
        return {"documents": documents}

class RunnableRetriever:
    def __init__(self, retriever):
        self.retriever = retriever

    def invoke(self, inputs):
        query = inputs.get("query", "")
        documents = self.retriever.get_relevant_documents(query)
        return {"documents": documents}

    def __call__(self, inputs):
        return self.invoke(inputs)

def setup_retrievers(chunks):
    print("Setting up dense and sparse retrievers...")
    dense_retriever, bm25_retriever = create_retrievers(chunks)
    runnable_dense = RunnableRetriever(dense_retriever)
    runnable_sparse = RunnableRetriever(bm25_retriever)
    runnable_hybrid = RunnableRetriever(
        CustomHybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=bm25_retriever,
            dense_weight=0.5,
            k=5
        )
    )
    retriever_dict = {"dense": runnable_dense, "sparse": runnable_sparse, "hybrid": runnable_hybrid}
    print("Retrievers are set up: dense, sparse, hybrid.")
    return retriever_dict 