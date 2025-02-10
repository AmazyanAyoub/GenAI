import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_groq import ChatGroq

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

model = ChatGroq(model="llama3-70b-8192")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()

text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = text_spliter.split_documents(docs)

vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

retriever = vectordb.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context":retriever | format_docs, "question":RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

response = rag_chain.invoke("What is Task Decomposition?")

print(response)