from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter
import os
import pandas as pd

def load_documents():
    # Load the document from the file and split it into chunks
    loader = TextLoader("./data/Ragas/testContext.txt")
    docs = loader.load()
    text_splitter = TokenTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        encoding_name="cl100k_base"  # This is GPT-4's encoding
    )
    chunks = text_splitter.split_documents(docs)
    # Modify each document's metadata to add a file_name key (needed for RAGAS)
    for document in chunks:
        document.metadata["file_name"] = document.metadata["source"]
    return docs, chunks

def load_qa():
    # Load QA pairs from the CSV file
    df = pd.read_csv("./data/Ragas/qa.csv", delimiter=";")
    questions = df["question"].tolist()
    ground_truth = df["ground_truth"].tolist()
    return questions, ground_truth 