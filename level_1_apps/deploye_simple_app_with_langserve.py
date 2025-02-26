import os
import warnings
from langchain._api import LangChainDeprecationWarning
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq

warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
_ = load_dotenv(find_dotenv())

from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(model="llama3-70b-8192")

parser = StrOutputParser()

system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

chain = prompt_template | llm | parser

app = FastAPI(
    title="Simple Translator",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


## pyhton filename.py and to this link:
## http://localhost:8000/chain/playground/