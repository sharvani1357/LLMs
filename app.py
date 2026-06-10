import streamlit as st

st.set_page_config(

    page_title="Mini GPT Assistant",

    page_icon="🤖",

    layout="wide"

)

st.sidebar.title(
    "🤖 Mini GPT Assistant"
)

st.sidebar.success(
    "Navigation"
)

st.title(
    "Mini GPT Content Assistant"
)

st.write(
    """
Use the sidebar to navigate:

• Dashboard

• Model Tuning

• Settings
"""
)