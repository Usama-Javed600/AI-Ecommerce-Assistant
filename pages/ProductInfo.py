import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from styles import css, bot_template, user_template
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProductChatbot:
    def __init__(self):
        logging.info("Initializing ProductChatbot...")
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.vectorstore = None
        self.conversation_chain = None
        self.init_ui()

    def init_ui(self):
        logging.info("Initializing UI...")
        
        st.write(css, unsafe_allow_html=True)
        st.header("Product Information Assistant")
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    def get_text_chunks(self, text):
        logging.info("Splitting text into chunks...")
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vectorstore(self, text_chunks):
        logging.info("Creating vector store...")
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore

    def get_conversation_chain(self, vectorstore):
        logging.info("Creating conversation chain...")
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=OpenAI(),
            retriever=vectorstore.as_retriever()
        )
        return conversation_chain

    def handle_userinput(self, user_question):
        logging.info("Handling user input...")
        chat_history = st.session_state.chat_history

        try:
            response = st.session_state.conversation({'question': user_question, 'chat_history': chat_history})
            st.session_state.chat_history.append((user_question, response['answer']))

            bot_message = bot_template.replace("{{MSG}}", response['answer'])
            st.write(bot_message, unsafe_allow_html=True)
            logging.info(f"User question: {user_question}")
            logging.info(f"Bot response: {response['answer']}")
        except Exception as e:
            logging.error(f"Error handling user input: {e}")
            st.error("An error occurred while processing your question.")

    def run(self):
        logging.info("Running the chatbot...")
        user_question = st.text_input("Feel free to ask any questions about the product you've chosen:")
        if user_question:
            try:
                with open('artifacts/product_info.txt',  encoding='utf-8') as file:
                    raw_text = file.read()

                text_chunks = self.get_text_chunks(raw_text)
                self.vectorstore = self.get_vectorstore(text_chunks)
                st.session_state.conversation = self.get_conversation_chain(self.vectorstore)
                self.handle_userinput(user_question)
            except FileNotFoundError:
                logging.error("Product data file not found.")
                st.error("No product data file found. Please log in and choose a product for order or negotiation.")
                st.stop()
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                st.error("An unexpected error occurred. Please try again later.")

if __name__ == '__main__':
    chatbot = ProductChatbot()
    chatbot.run()
