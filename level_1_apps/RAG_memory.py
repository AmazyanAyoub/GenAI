import os
import warnings
from langchain._api import LangChainDeprecationWarning
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
_ = load_dotenv(find_dotenv())


from langchain import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory


memory = ConversationBufferMemory(
    chat_memory = FileChatMessageHistory("memory.json"),
    memory_key="messages",
    return_messages=True
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)


llm = ChatGroq(model="llama3-70b-8192")

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

response = chain.invoke({"content":"hello!"})

print("\n----------\n")

print("hello!")

print("\n----------\n")
print(response)

print("\n----------\n")

response = chain.invoke("my name is Julio")

print("\n----------\n")

print("my name is Julio")

print("\n----------\n")
print(response)

print("\n----------\n")

response = chain.invoke("what is my name?")

print("\n----------\n")

print("what is my name?")

print("\n----------\n")
print(response)

print("\n----------\n")
