import os 
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

_ = load_dotenv(find_dotenv())

chat_model = ChatGroq(
    model="llama3-70b-8192"
)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

loaded_text = TextLoader("../data/state_of_the_union.txt", encoding='utf-8').load()

# text = loaded_text.load()

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)


# chat_prompt = ChatPromptTemplate(
#     [("system","Your a professional in {role} and you have to help the human and answer what he wants"),
#     ("human", "Answer this {question} here some extrat informations {text}")]
# )

# message = chat_prompt.format(
#     role = "human history",
#     question = "where was JFK born?",
#     text = text
# )

# response = chat_model.invoke(message)

# print(response.content)

splited_text = text_spliter.split_documents(loaded_text)

# embedded_text = embedding.embed_documents(splited_text)

vector_db = FAISS.from_documents(splited_text, embedding)

retriever = vector_db.as_retriever(search_kwargs={"k": 1})

question = "What did the president say about the John Lewis Voting Rights Act?"

def format_docs(doc):
    return "\n\n".join([d.page_content for d in doc])


template = """Answer the question based only on the following context:

{context}

Question: {question}
"""

chat_template = ChatPromptTemplate.from_template(template)

chain = (
    {"context":retriever | format_docs, "question":RunnablePassthrough()}
    | chat_template
    | chat_model
    | StrOutputParser()
)

response = chain.invoke(question)

print(response)
