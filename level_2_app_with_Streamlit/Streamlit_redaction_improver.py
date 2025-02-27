import streamlit as st
from langchain import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

st.title("Redaction Improver")

template = """
    Below is a draft text that may be poorly worded.
    Your goal is to:
    - Properly redact the draft text
    - Convert the draft text to a specified tone
    - Convert the draft text to a specified dialect

    Here are some examples different Tones:
    - Formal: Greetings! OpenAI has announced that Sam Altman is rejoining the company as its Chief Executive Officer. After a period of five days of conversations, discussions, and deliberations, the decision to bring back Altman, who had been previously dismissed, has been made. We are delighted to welcome Sam back to OpenAI.
    - Informal: Hey everyone, it's been a wild week! We've got some exciting news to share - Sam Altman is back at OpenAI, taking up the role of chief executive. After a bunch of intense talks, debates, and convincing, Altman is making his triumphant return to the AI startup he co-founded.  

    Here are some examples of words in different dialects:
    - American: French Fries, cotton candy, apartment, garbage, \
        cookie, green thumb, parking lot, pants, windshield
    - British: chips, candyfloss, flag, rubbish, biscuit, green fingers, \
        car park, trousers, windscreen

    Example Sentences from each dialect:
    - American: Greetings! OpenAI has announced that Sam Altman is rejoining the company as its Chief Executive Officer. After a period of five days of conversations, discussions, and deliberations, the decision to bring back Altman, who had been previously dismissed, has been made. We are delighted to welcome Sam back to OpenAI.
    - British: On Wednesday, OpenAI, the esteemed artificial intelligence start-up, announced that Sam Altman would be returning as its Chief Executive Officer. This decisive move follows five days of deliberation, discourse and persuasion, after Altman's abrupt departure from the company which he had co-established.

    Please start the redaction with a warm introduction. Add the introduction \
        if you need to.
    
    Below is the draft text, tone, and dialect:
    DRAFT: {draft}
    TONE: {tone}
    DIALECT: {dialect}

    YOUR {dialect} RESPONSE:
"""
prompt = PromptTemplate(
    input_variables=["draft", "tone", "dialect"],
    template=template,
    )

def load_llm(Groq_api_key):
    llm = ChatGroq(model="llama3-70b-8192", api_key=Groq_api_key)
    return llm

# st.set_page_config(page_title="Re-write your text")
# st.header("Re-write your text")

col1, col2 = st.columns(2)

with col1:
    st.markdown("Re-write your text in different styles.")

with col2:
    st.write("Contact with [AI Accelera](https://aiaccelera.com) to build your AI Projects")

st.markdown("## Enter Your OpenAI API Key")

def get_api_key():
    api_key = st.text_input(label="OpenAI API Key", key="openai_api_key_input", placeholder="Ex: sk-2twmA8tfCb8un4...", type="password")
    return api_key

Groq_api_key = get_api_key()

st.markdown("## Enter the text you want to re-write")

def get_text():
    draft = st.text_area(label="text", label_visibility="collapsed", key="draft")
    return draft

draft = get_text()

if len(draft.split(" ")) > 700:
    st.write("Please enter a shorter text. The maximum length is 700 words.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    option_tone = st.selectbox(
        "which tone would you like your redaction to have?",
        ("Formal", "Informal")
    )

with col2:
    option_dialect = st.selectbox(
        "which dialect would you like your redaction to have?",
        ("American", "British")
    )

st.markdown("### Your Re-written text:")

if draft:
    if not Groq_api_key:
        st.warning("Please enter your Groq API Key. \
                   Instructions [here](https://console.groq.com/keys)", icon="⚠️")
        st.stop()
    
    llm = load_llm(Groq_api_key)

    prompt_with_draft = prompt.format(
        draft=draft,
        tone=option_tone,
        dialect=option_dialect
    )

    improved_redaction = llm([HumanMessage(content=prompt_with_draft)])

    st.write(improved_redaction)