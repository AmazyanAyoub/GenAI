import streamlit as st
from langchain import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from io import StringIO

#Page title and header
st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer")

st.markdown("ChatGPT cannot summarize long texts. Now you can do it with this app.")

def load_llm(Groq_api_key):
    llm = ChatGroq(model="llama3-70b-8192", api_key=Groq_api_key)
    return llm


col1, col2 = st.columns(2)

with col1:
    st.markdown("ChatGPT cannot summarize long texts. Now you can do it with this app.")

st.markdown("## Enter Your OpenAI API Key")
def get_api_key():
    api_key = st.text_input(label="OpenAI API Key", key="openai_api_key_input", placeholder="Ex: sk-2twmA8tfCb8un4...", type="password")
    return api_key

Groq_api_key = get_api_key()


st.markdown("## Upload the file you want to summarize")
file = st.file_uploader("Choose a file", type="txt")

st.markdown("### Here is your Summary:")
if file is not None:
    stringio = StringIO(file.getvalue().decode("utf-8"))
    # st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    # st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    #dataframe = pd.read_csv(uploaded_file)
    #st.write(dataframe)

    # Or we can use directly this command:
    # text = uploaded_file.read().decode("utf-8")

    file_text = string_data

    if len(file_text) > 20000:
        st.write("Please enter a shorter text. The maximum length is 20000 characters.")
        st.stop()

    if file_text:
        if not Groq_api_key:
            st.warning("Please enter your Groq API Key. \
                       Instructions [here](https://console.groq.com/keys)", icon="⚠️")
            st.stop()

        llm = load_llm(Groq_api_key)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=350,
        )

        docs = text_splitter.create_documents([file_text])

        summarize_chain = load_summarize_chain(
            llm,
            chain_type="map_reduce"
        )

        summary = summarize_chain.run(docs)

        st.write(summary)