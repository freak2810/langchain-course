import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore

load_dotenv()

print("Intitializing Components")

embeddings = PineconeEmbeddings(model="llama-text-embed-v2")
llm = ChatOllama(model="qwen3:8b")

vector_store = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"),
    embedding=embeddings,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions based only on the following context provided.
    If you don't know the answer, just say you don't know. Do not try to make up an answer.

    {context}

    Question: {question}
    Answer:"""
)


def format_docs(docs):
    return "\n\n ".join([doc.page_content for doc in docs])


def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LLMs. This function retrieves relevant documents from the vector store based on the query and formats them for display.
    """
    docs = retriever.invoke(query)
    context = format_docs(docs)

    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)
    return response.content

def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain



if __name__ == "__main__":
    print("Retrieving")

    query = "What is pinecone in machine learning"

    # Call without any RAG
    # print("\n" + "=" * 70)
    # print("Implementation - No RAG")
    # result_raw = llm.invoke([HumanMessage(content=query)])
    # print(f"Answer: {result_raw.content}")

    # Call with RAG without LCEL
    # print("\n" + "=" * 70)
    # print("Implementation - RAG without LCEL")
    # result_rag = retrieval_chain_without_lcel(query)
    # print(f"Answer: {result_rag}")

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
