## Simple GenAI app Using LangChain and OpenAI

import os
from dotenv import load_dotenv
from langchain_classic.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT']=os.getenv('LANGCHAIN_PROJECT')

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's query accurately."),
        ("user", "Question: {question}"),
    ]
)

## Streamlit Framework
st.title("💬 LangChain with Gemma Model")
input_text=st.text_input("What question do you have in mind?")

## Ollama LLAMA2 Model
llm=Ollama(model="gemma:2b")
output_parse=StrOutputParser()
chain=prompt | llm | output_parse

if input_text:
    st.write(chain.invoke({"question":input_text}))
