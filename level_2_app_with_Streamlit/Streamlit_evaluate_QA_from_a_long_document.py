import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain

st.set_page_config(
    page_title="Evaluate a RAG App"
)
st.title("Evaluate a RAG App")

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

with st.expander("Evaluate the quality of a RAG APP"):
    st.write("""
        To evaluate the quality of a RAG app, we will
        ask it questions for which we already know the
        real answers.
        
        That way we can see if the app is producing
        the right answers or if it is hallucinating.
    """)

uploaded_file = st.file_uploader(
    "Upload a .txt document",
    type="txt"
)

query_text = st.text_input(
    "Enter a question you have already fact-checked:",
    placeholder="Write your question here",
    disabled=not uploaded_file
)

response_text = st.text_input(
    "Enter the real answer to the question:",
    placeholder="Write the confirmed answer here",
    disabled=not uploaded_file
)

result = []
with st.form(
    "myform",
    clear_on_submit=True
):
    Groq_api_key = st.text_input(
        "Groq api key:",
        type="password",
        disabled=not (uploaded_file and query_text)
    )
    submitted = st.form_submit_button(
        "Submit",
        disabled=not (uploaded_file and query_text)
    )
    if submitted and Groq_api_key.startswith("sk-"):
        with st.spinner(
            "Wait, please. I am working on it..."
            ):
            response = generate_response(
                uploaded_file,
                Groq_api_key,
                query_text,
                response_text
            )
            result.append(response)
            del Groq_api_key
            
if len(result):
    st.write("Question")
    st.info(response["predictions"][0]["question"])
    st.write("Real answer")
    st.info(response["predictions"][0]["answer"])
    st.write("Answer provided by the AI App")
    st.info(response["predictions"][0]["result"])
    st.write("Therefore, the AI App answer was")
    st.info(response["graded_outputs"][0]["results"])