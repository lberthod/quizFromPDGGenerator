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
st.header("Transmettez vos PDF 💬")

# Prompt templates
truefalse_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Write me for  "True or false" question about this : {topic}. Explain in a paragrph the answer'
)

mcq_title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Write me four  "multiple choice" question, with one true answer and three false about this : {topic}. Explain in a paragrph the answer'
)

# Memory 
truefalse_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
mcq_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

    # upload file
pdf = st.file_uploader("Upload your PDF", type="pdf")
    
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
    user_question = st.text_input("Donnez le thème de la question")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        truefalse_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
        mcq_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
        st.write(docs) 

        mcq = mcq_chain.run(str(docs))
        truefalse = truefalse_chain.run(str(docs))

        st.write(mcq)
        st.write(truefalse) 
