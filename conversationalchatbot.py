import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from PyPDF2 import PdfMerger, PdfReader
from datetime import datetime
import os

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="shashaaa/fine_tune_embeddnew_SIH")

st.title("Conversational RAG chatbot")
st.write("Upload pdfs")

# api_key=st.text_input("enter your groq api key:",type="password")
api_key = os.getenv("GROQ_API_KEY")

if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")

    if 'store' not in st.session_state:
        st.session_state.store={}

    session_id=st.text_input("Unique case name",placeholder="An unique session identifier")
    tm = datetime.now()
    if session_id in st.session_state.store:
        fname = st.session_state.store[session_id][1]
    else:
        fname = f"Case: {session_id}  Time: {tm.strftime('%d-%m-%Y %H:%M:%S')}"
    st.write(fname)


    uploaded_files = st.file_uploader("Choose a PDF",type="pdf",accept_multiple_files=True)
    merger = PdfMerger()
    print(uploaded_files)

    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./tmp/temp-{session_id}.pdf"
            with open(temppdf,"wb") as file:
                file.write(b'')
            #     file_name=uploaded_file.name
            merger.append(PdfReader(uploaded_file))
        merger.write(temppdf)
        merger.close()


        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)
        # print(documents)

        
        text_splitter= RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits= text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits , embedding=embeddings)
        retriever = vectorstore.as_retriever()


        contextualize_q_system_prompt=(
            "given a chat history and latest user question"
            "which might refrence context in chat history"
            "formulate a standalone question which can be understood"
            "without thta chat history.DO NOt answer the question"
            "just refromulate it if needed and otherwise return it as is"
        )

        contextualize_q_system_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )

        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_system_prompt)


        system_prompt=(
            "You are a virtual lawyer, an AI-powered legal assistant designed to provide accurate and helpful legal information. "
            "You assist users with understanding legal concepts, answering legal questions, and providing guidance on legal matters. "
            "You do not provide legal advice but offer information to help users make informed decisions. "
            "Maintain a neutral tone and avoid taking sides, even when there are emotional or ethical factors involved. "
            "Always provide relevant legal context, including references to applicable acts, laws, or past cases. "
            "If you don't know the answer, say that you don't know. "
            "Keep your responses informative but not overly lengthy."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )


        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->ChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=[ChatMessageHistory(), fname]
            return st.session_state.store[session_id][0]

        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input= st.text_input("your question", value='')
        if user_input:
            session_history=get_session_history(session_id)
            response =  conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                }
            )
            st.write("Assitant:",response['answer'])
            st.write(st.session_state.store[session_id])
            st.write("chat_history:",session_history.messages)

else:
    st.warning("please enter api key")