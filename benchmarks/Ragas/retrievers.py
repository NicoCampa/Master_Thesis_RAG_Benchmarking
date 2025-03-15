from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.retrievers import BM25Retriever
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
import os

# Path configuration for persistent storage
CHROMA_PERSIST_DIR = "./data/Ragas/vectordb"
COLLECTION_NAME = "companies_collection"
DEFAULT_NUM_CHUNKS = 5  # Set default number of chunks to retrieve

class CustomHybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, dense_weight: float = 0.5, k: int = DEFAULT_NUM_CHUNKS):
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
            score = 1.0 - (i / len(dense_docs)) if dense_docs else 0
            doc_scores[doc.page_content] = self.dense_weight * score
            
        for i, doc in enumerate(sparse_docs):
            score = 1.0 - (i / len(sparse_docs)) if sparse_docs else 0
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

def create_runnable_retriever(retriever):
    """Create a proper LangChain runnable from a retriever"""
    def _retrieve_docs(input_data):
        # Extract query from input
        if isinstance(input_data, str):
            query = input_data
        else:
            query = input_data.get("question", "")
            if not query and "query" in input_data:
                query = input_data.get("query", "")
        
        # Get documents from retriever
        documents = retriever.get_relevant_documents(query)
        
        # Return documents directly with the key main.py expects
        return {"documents": documents, "query": query}
    
    # Use LangChain's proper runnable transformation
    return RunnableLambda(_retrieve_docs)

def check_vectordb_status():
    """Check if the vector database exists and is up to date"""
    context_path = "./data/Ragas/testContext.txt"
    
    # If vectordb doesn't exist, it needs to be built
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print("Vector database directory doesn't exist. Creating new database.")
        return True
    
    # Check if directory is empty
    if not os.listdir(CHROMA_PERSIST_DIR):
        print("Vector database directory is empty. Creating new database.")
        return True
    
    # If context file doesn't exist, something's wrong but we'll build anyway
    if not os.path.exists(context_path):
        print("Context file not found. Creating new database.")
        return True
    
    # Check if context file has been modified since vectordb was last updated
    context_mtime = os.path.getmtime(context_path)
    vectordb_mtime = os.path.getmtime(CHROMA_PERSIST_DIR)
    
    if context_mtime > vectordb_mtime:
        print("Context file has been modified. Rebuilding database.")
        return True
    
    print("Using existing vector database.")
    return False

def setup_retrievers(chunks):
    """Setup retrievers with persistent storage for dense retriever"""
    print("Setting up retrievers...")
    
    # Check if we need to rebuild the database
    should_rebuild = check_vectordb_status()
    
    # Create directory if it doesn't exist
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    
    # Create embeddings model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Handle dense retriever with persistence
    if should_rebuild:
        print("Creating new Chroma vector database...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=COLLECTION_NAME
        )
        # Ensure the database is saved
        vectorstore.persist()
    else:
        print("Loading existing Chroma vector database...")
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
    
    # Create retrievers - all with k=5
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": DEFAULT_NUM_CHUNKS})
    
    sparse_retriever = BM25Retriever.from_documents(chunks)
    sparse_retriever.k = DEFAULT_NUM_CHUNKS
    
    hybrid_retriever = CustomHybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        dense_weight=0.5,
        k=DEFAULT_NUM_CHUNKS
    )
    
    # Wrap retrievers using proper LangChain runnables
    retriever_dict = {
        "dense": create_runnable_retriever(dense_retriever),
        "sparse": create_runnable_retriever(sparse_retriever),
        "hybrid": create_runnable_retriever(hybrid_retriever)
    }
    
    print(f"Retrievers setup complete: each retriever will fetch {DEFAULT_NUM_CHUNKS} chunks")
    return retriever_dict 