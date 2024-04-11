import streamlit as st
from st_social_media_links import SocialMediaIcons
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import random
import time

# Function to generate streamed response
def response_generator(message):
    for word in message.split():
        yield word + " "
        time.sleep(0.05)

# Function to get vector store from URL
def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings(api_key=openai_api_key))
    return vector_store

# Function to get context retriever chain
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(api_key=openai_api_key)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain
    
# Function to get conversational RAG chain
def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(api_key=openai_api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Function to get response
def get_response(user_input, openai_api_key):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']

# App configuration
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    openai_api = st.text_input("Please enter your OpenAI API key", type="password")
    openai_api_key = openai_api
    if openai_api_key:
        website_url = st.text_input("Website URL")
    SocialMediaIcons(["https://twitter.com/DanielEftekhari"]).render(sidebar=True, justify_content="space-evenly")

# Main functionality
if openai_api_key and website_url:
    if "chat_history" not in st.session_state:
        # Initialize the chat history with a greeting message from the AI
        st.session_state.chat_history = [
            AIMessage(content="Hi, how can I assist you?")
        ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # Display the initial AI message (greeting) before user input
    for message in st.session_state.chat_history:
        role = "AI" if isinstance(message, AIMessage) else "Human"
        with st.chat_message(role):
            st.write(message.content)

    # Process user input
    user_query = st.chat_input("Type your message here...")
    if user_query:
        human_msg = HumanMessage(content=user_query)
        st.session_state.chat_history.append(human_msg)
    # AI response
        response_text = get_response(user_query, openai_api_key)
        ai_msg = AIMessage(content=response_text)
        st.session_state.chat_history.append(ai_msg)

        # Display chat history with user and streamed AI responses
        for message in st.session_state.chat_history[len(st.session_state.chat_history) - 2:]:
            role = "AI" if isinstance(message, AIMessage) else "Human"
            with st.chat_message(role):
                if isinstance(message, AIMessage) and message is ai_msg:
                    st.write_stream(response_generator(message.content))
                else:
                    st.write(message.content)
else:
    st.info("Please enter both your OpenAI API key and website URL to continue.")