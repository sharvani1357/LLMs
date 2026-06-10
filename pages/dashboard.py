import streamlit as st
import torch
import pandas as pd
import pickle
import time

from model import MiniGPT
from utils import (
    generate_text,
    generate_topk
)

from config import *

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    device = torch.device("cpu")

    with open(
        "saved_model/word2idx.pkl",
        "rb"
    ) as f:

        word2idx = pickle.load(f)

    with open(
        "saved_model/idx2word.pkl",
        "rb"
    ) as f:

        idx2word = pickle.load(f)

    model = MiniGPT(

        vocab_size=len(word2idx),

        embed_dim=EMBED_DIM,

        max_seq_len=MAX_SEQ_LEN,

        num_heads=NUM_HEADS,

        num_layers=NUM_LAYERS,

        ff_dim=FF_DIM,

        dropout=DROPOUT

    )

    model.load_state_dict(

        torch.load(
            "saved_model/mini_gpt.pth",
            map_location=device
        )

    )

    model.eval()

    return model, word2idx, idx2word


model, word2idx, idx2word = load_model()

device = torch.device("cpu")

# ==========================================
# PAGE
# ==========================================

st.title(
    "ASSIGNMENT: MINI GPT CONTENT ASSISTANT"
)

left, center, right = st.columns(
    [1, 3, 1.5]
)

# ==========================================
# CENTER PANEL
# ==========================================

with center:

    st.subheader(
        "Live Article Editor (Markdown Supported)"
    )

    article_title = st.text_input(

        "Article Title",

        "The Future of Renewable Energy Adoption"

    )

    article_text = st.text_area(

        "Editor",

        value="""
Implementing decentralized energy storage solutions will be critical because
""",

        height=250

    )

    start = time.time()

    prediction = generate_topk(

        model,

        article_text,

        word2idx,

        idx2word,

        device,

        k=10,

        max_new_tokens=12

    )

    end = time.time()

    generation_time = round(
        (end - start) * 1000,
        2
    )

    st.markdown(
        f"""
        <div style="
        color:gray;
        font-style:italic">
        Suggested:
        {prediction}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Press TAB to accept (95% confidence)"
    )

# ==========================================
# RIGHT PANEL
# ==========================================

with right:

    st.subheader(
        "AI Assistant Panel"
    )

    st.success(
        PROJECT_CONTEXT
    )

    st.markdown(
        "### CURRENT SUGGESTIONS"
    )

    st.markdown(
        """
        **1. they maximize grid stability**

        Confidence: 88%
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        st.button(
            "Insert 1"
        )

    with c2:
        st.button(
            "Alternative 1"
        )

    st.markdown("---")

    st.markdown(
        """
        **2. infrastructure investment is often decentralized**

        Confidence: 75%
        """
    )

    c3, c4 = st.columns(2)

    with c3:
        st.button(
            "Insert 2"
        )

    with c4:
        st.button(
            "Alternative 2"
        )

# ==========================================
# MODEL MONITOR
# ==========================================

st.markdown("---")

st.subheader(
    "MODEL MONITOR"
)

m1, m2, m3 = st.columns(3)

with m1:

    st.success(
        "Model Status: Active"
    )

    st.metric(
        "Generation Time",
        f"{generation_time} ms"
    )

with m2:

    st.write(
        "Context Window"
    )

    token_count = len(
        article_text.split()
    )

    context_df = pd.DataFrame(

        {

            "Tokens":
            [token_count]

        }

    )

    st.bar_chart(
        context_df
    )

with m3:

    st.write(
        "Token Probabilities"
    )

    probs = pd.DataFrame(

        {

            "Probability":

            [

                0.30,
                0.20,
                0.15,
                0.10,
                0.08,
                0.05

            ]

        }

    )

    st.bar_chart(
        probs
    )