import streamlit as st
import pandas as pd
import recom_utils

st.set_page_config(page_title="EcommerceAI Assistant", layout="centered")

class LoginApp:
    def __init__(self):
        self.customer_data = self.load_customer_data('customer.csv')
        self.initialize_session_state()

    @staticmethod
    @st.cache_data
    def load_customer_data(file_path):
        return pd.read_csv(file_path, encoding='ISO-8859-1')

    def initialize_session_state(self):
        if 'login_status' not in st.session_state:
            st.session_state.login_status = False
        if 'username' not in st.session_state:
            st.session_state.username = ""
        if 'password' not in st.session_state:
            st.session_state.password = ""
        if 'page' not in st.session_state:
            st.session_state.page = 'login'  # Set the initial page to 'login'

    def authenticate(self, username, password):
        user_record = self.customer_data[
            (self.customer_data['username'] == username) & 
            (self.customer_data['password'] == password)
        ]
        return not user_record.empty

    def show_login_page(self):
        if st.session_state.page == 'login':
            st.subheader('Login ')
            with st.form(key='login_form'):
                username = st.text_input('Username')
                password = st.text_input('Password', type='password')
                login_button = st.form_submit_button('Login')

                if login_button:
                    if self.authenticate(username, password):
                        # Save the username and password in session state
                        st.session_state.username = username
                        st.session_state.password = password

                        # Set login status to True and update the page to 'chatbot'
                        st.session_state.login_status = True
                        st.session_state.page = 'chatbot'

                        # Rerun the app to update the UI
                        st.rerun()
                    else:
                        st.error('Invalid username or password')

    def show_chatbot(self):
        chatbot = recom_utils.EcommerceChatAssistant(data_path='products.json')  # Create an instance of the Chatbot class
        chatbot.run()  # Run the chatbot

    def run(self):
        if st.session_state.page == 'chatbot':
            # Show the chatbot if the page is set to 'chatbot'
            self.show_chatbot()
        else:
            # Display the login page if the page is set to 'login'
            st.title("Welcome to the AI Ecommerce Chat Assistant")
            st.write("""
            This e-commerce platform, integrated with an external e-commerce site, offers a comprehensive product recommendation and negotiation experience:

            - **Product Recommendations:** Personalized suggestions with summarized product details.
            - **Product Info Page:** Detailed insights on selected products obtained through web scraping at the time of product selection.
            - **Negotiation Page:** Flexible pricing options with product discount predictions based on product profile and user profile, including factors such as rating, category, transaction history, and subscription status.
            """)
            
            
            st.subheader("User Testing Examples")

            st.write("""
            | **Type**                             | **Username** | **Password** |
            |--------------------------------------|--------------|--------------|
            | **Subscribed Customer**              | `username1`  | `password1`  |
            | **Non-Subscribed Customer**          | `username2`  | `password2`  |
            | **Subscribed Customer with Different History** | `username3`  | `password3`  |
            """)


            st.write("Experience tailored recommendations and flexible pricing.")

            self.show_login_page()

# Create an instance of the LoginApp class and run it
if __name__ == "__main__":
    app = LoginApp()
    app.run()
