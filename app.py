from __future__ import annotations
import streamlit as st
import pandas as pd
import json

st.set_page_config(layout='wide')

class App:

    with open('strings.txt', 'r') as f:
        __STRINGS = json.load(f)

    def __init__(self):
        # Preset session states; Used to preserve states on reruns/button pushes
        if 'data' not in st.session_state:
            st.session_state.data = ""
        if 'has_data' not in st.session_state:
            st.session_state.has_data = False
        if 'raw_toggle' not in st.session_state:
            st.session_state.raw_toggle = False