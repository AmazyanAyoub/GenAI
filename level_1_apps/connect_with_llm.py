import os
from dotenv import load_dotenv, find_dotenv 
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

_ = load_dotenv(find_dotenv())
llm = ChatGroq(model="llama3-70b-8192")


class Joke(BaseModel):
    setup: str = Field(description="question to setup the joke")
    punchline: str = Field(description="answer to resolve the joke")

output_parser = SimpleJsonOutputParser(pydantic_object=Joke)


Prompt_template =  PromptTemplate(
    template="Answer the user query. \n {format_instructions} \n {query} \n",
    input_variables=["query"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},

)


chain = Prompt_template | llm | output_parser

response = chain.invoke({"query", "Tell me a short joke"})

print(response)
# message = prompt_template.format(
#     role = "AI",
#     subject = "Latest informations in the market of ai"
# )

# response = llm.invoke(message)



# chat_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "you are a profetional in {profession} and you should help to create {role} depending on {user_input}"),
#         ("human", "Hello Mr. profetional in {profession} can you help me?"),
#         ("ai", "Yes i can help you"),
#         ("human", "{user_input}"),
#     ]
# )


# message = chat_template.format(
#     profession = "HR",
#     role = "resume",
#     user_input = "I am an employee with one year of experience in AI field, I can work with python and other ai languages"
# )

# examples = [
#     {"input": "hi!", "output": "¡hola!"},
#     {"input": "bye!", "output": "¡adiós!"},
# ]

# example_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("human", "{input}"),
#         ("ai", "{output}"),
#     ]
# )

# few_shot_prompt = FewShotChatMessagePromptTemplate(
#     example_prompt=example_prompt,
#     examples=examples,
# )

# final_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are an English-Spanish translator."),
#         few_shot_prompt,
#         ("human", "{input}"),
#     ]
# )

# chain = final_prompt | llm

# response = chain.invoke({"input", "Hello my name is ayoub and am 23 years old"})

# print(response.content)

