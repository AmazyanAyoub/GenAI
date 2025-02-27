import streamlit as st
from langchain import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain

st.set_page_config(page_title="Evaluate QA from a long document")
st.header("Evaluate QA from a long document")

def generate_response(uploaded_file, Groq_api_key, query_text, response_text):

    text = [uploaded_file.read().decode("utf-8")]

    text_spliter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    docs = text_spliter.split_documents(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectrdb = FAISS.from_documents(docs, embeddings)

    retriever = vectrdb.as_retriever()

    llm = ChatGroq(model="llama3-70b-8192", api_key=Groq_api_key)

    real_qa = [
        {
            "question": query_text,
            "answer": response_text
        }
    ]

    qachain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="question"
    )

    predictions = qachain.predict(real_qa)

    eval_chain = QAEvalChain.from_llm(llm)

    grabed_output = eval_chain.evaluate(
        real_qa,
        predictions,
        question_key="question",
        prediction_key="key",
        answer_key="answer"
    )

    response = [
        {
            "prediction": predictions,
            "grabed_output": grabed_output
        }
    ]

    return response

