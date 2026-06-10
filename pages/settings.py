import streamlit as st

st.title(
    "Settings"
)

theme = st.selectbox(

    "Theme",

    [

        "Dark",

        "Light"

    ]

)

st.write(
    f"Selected Theme: {theme}"
)