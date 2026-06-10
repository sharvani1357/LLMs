import streamlit as st

st.title(
    "Model Tuning"
)

temperature = st.slider(

    "Temperature",

    0.1,

    2.0,

    0.7

)

top_k = st.slider(

    "Top-K",

    1,

    50,

    10

)

st.write(
    f"Temperature: {temperature}"
)

st.write(
    f"Top-K: {top_k}"
)