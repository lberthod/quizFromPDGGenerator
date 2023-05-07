# Bring in deps
import os 
from apikey import apikey 

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
    
os.environ['OPENAI_API_KEY'] = apikey
st.set_page_config(page_title="Ask your PDF")
st.header("Transmettez vos de PDF du cours 💬")

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Ecris moi un quizz sur le contenu {topic}'
)


# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')



    # upload file
pdf = st.file_uploader("Upload votre PDF", type="pdf")
    
    # extract the text
if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
      
    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
      
     # show user input
    #user_question = st.text_input("Générer un questionnaire comprenant 4 question vrai ou faux et 4 question à choix multiple.")
    #if user_question:
    user_question = "Générer un questionnaire comprenant 4 question vrai ou faux et 4 question à choix multiple."
    docs = knowledge_base.similarity_search("connaisance santé")
        
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)

    #title = title_chain.run(docs)
    st.write(docs) 
    #with get_openai_callback() as cb:
    #    response = chain.run(input_documents=docs, question="Peux tu me générer un questionnaire comprenant 4 question vrai ou faux et 4 question à choix multiple et écrire ici la première question.")
    #    print(cb)
           
   # st.write(response)
