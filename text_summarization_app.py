import streamlit as st
import os
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Configuración de la página, esto debe ser lo primero
st.set_page_config(page_title='🦜🔗 Text Summarization App')

def generate_response(txt):
    openai_api_key = os.getenv(sk-proj-ESytyXnCgpz4OeAMUCifT3BlbkFJIlbk8hPFbGNRQkXPOPXI)
    if openai_api_key is None:
        st.error("OpenAI API key is not set. Please check your environment variables.")
        return "API Key not available. Check environment settings."

    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

st.title('🦜🔗 Text Summarization App')
txt_input = st.text_area('Enter your text', '', height=200)

if st.button('Summarize'):
    response = generate_response(txt_input)
    if response:
        st.write(response)
