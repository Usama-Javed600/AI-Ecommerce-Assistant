import streamlit as st

# General settings for the app
PAGE_TITLE = "Ecommerce Assistant"
PAGE_ICON = ":shopping_cart:"
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

st.title("Welcome to the Ecommerce Assistant")
st.write("Please navigate to the Login, Recommend, or Negotiate pages using the sidebar.")
