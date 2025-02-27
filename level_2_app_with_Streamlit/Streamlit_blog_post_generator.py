import streamlit as st
from langchain import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

st.title("Blog Post Generator")

Groq_api_key = st.sidebar.text_input(
    "Groq Api Key",
    type="password"
)

def generator_response(topic):
    lmm = ChatGroq(model="llama3-70b-8192", api_key=Groq_api_key)

    template = """
    As experienced startup and venture capital writer, 
    generate a 400-word blog post about {topic}
    
    Your response should be in this format:
    First, print the blog post.
    Then, sum the total number of words on it and print the result like this: This post has X words.
    """

    prompt = PromptTemplate(
        input_variables=["topic"],
        template=template
    )

    message = prompt.format(topic=topic)
    response = lmm.invoke([HumanMessage(content=message)])

    st.write(response.content)

topic = st.text_input("Enter a topic", placeholder="Enter a topic...", key="topic")
if st.button("Enter"):
    if topic:
        if not Groq_api_key:
            st.warning("Please enter your Groq API Key. \
                    Instructions [here](https://console.groq.com/keys)", icon="⚠️")
            st.stop()
            generator_response(topic)