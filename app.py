from calendar import c
import json
import os
from pyexpat import model
import sys
import boto3
import streamlit as st

## We will use Titan Embedding Model to genearate Embeddings.

import boto3
from langchain_aws import BedrockEmbeddings, Bedrock
from langchain_community.vectorstores import FAISS

## Data ingestion Libraries

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding and Vector store

from langchain_community.vectorstores import FAISS

# llm Moadels
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

## data ingestion

def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs

## Vector Embedding and vector store
def get_vectorstore(docs):
    vectorstore = FAISS.from_documents(documents=docs, embedding=bedrock_embeddings)

    vectorstore.save_local("faiss_index")
    return vectorstore

def get_claude_llm():
    bedrock_llm = Bedrock(model_id= "anthropic.claude-3-haiku-20240307-v1:0", client=bedrock, temperature=0, max_tokens=1000)
        
    return bedrock_llm


def get_llama_llm():
    bedrock_llm = Bedrock(model_id= "meta.llama3-8b-instruct-v1:0", client=bedrock, temperature=0, max_gen_len=1000)
        
    return bedrock_llm


prompt_template = f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>
Question: {question}
Question: {question}

Assistant:"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore, query):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    response = qa_chain({"query": query})
    return response['result']

def main():
    st.title("Document QA using Bedrock LLMs and Embeddings")
    st.sidebar.title("Settings")

    llm_option = st.sidebar.selectbox(
        "Select LLM Model",
        ("Claude 3 Haiku", "Llama 3 Instruct")
    )

    if llm_option == "Claude 3 Haiku":
        llm = get_claude_llm()
    else:
        llm = get_llama_llm()

    if st.sidebar.button("Ingest Data and Create Vector Store"):
        with st.spinner("Ingesting data and creating vector store..."):
            docs = data_ingestion()
            vectorstore = get_vectorstore(docs)
        st.success("Data ingested and vector store created!")

    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local("faiss_index", bedrock_embeddings)
        query = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            with st.spinner("Getting answer..."):
                answer = get_response_llm(llm, vectorstore, query)
            st.success("Answer retrieved!")
            st.write("Answer:", answer)
    else:
        st.warning("Please ingest data and create vector store first.")

if __name__ == "__main__":
    main()
