from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

# Define the prompt template – exactly as in the notebook
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

class RetrieverToContext:
    def __init__(self, retriever):
        self.retriever = retriever

    def invoke(self, inputs):
        query = inputs.get("question", "")
        retriever_output = self.retriever.invoke({"query": query})
        docs = retriever_output.get("documents", [])
        joined_context = "\n".join([doc.page_content for doc in docs])
        return {"context": joined_context, "question": query}

    def __call__(self, inputs):
        return self.invoke(inputs)

def rag_chain(retriever_dict, strategy="dense", model=None):
    """
    Return a chain that uses either 'dense', 'sparse', or 'hybrid'
    depending on the argument.
    """
    if model is None:
        raise ValueError("Model must be provided to rag_chain")
        
    # Fallback to dense if unknown strategy
    selected_retriever = retriever_dict.get(strategy, retriever_dict["dense"])

    chain_instance = (
        {
            "context": selected_retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )
    return chain_instance 