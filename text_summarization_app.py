import streamlit as st
import os
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Configuración de la página, esto debe ser lo primero
st.set_page_config(page_title='🦜🔗 Text Summarization App')

def generate_response(txt, openai_api_key):
    if openai_api_key is None:
        st.error("OpenAI API key is not set. Please enter your API key to use the app.")
        return "API Key not available. Please enter your API key."

    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

st.title('🦜🔗 Text Summarization App')
txt_input = st.text_area('Enter your text', '', height=200)
openai_api_key = st.text_input('Enter your OpenAI API key', type='password')

if st.button('Summarize'):
    response = generate_response(txt_input, openai_api_key)
    if response:
        st.write(response)
