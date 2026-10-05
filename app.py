from pathlib import Path

import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Pixel GPT", page_icon="◧", layout="wide")
torch.set_num_threads(2)
st.markdown(f"<style>{(ROOT / 'ui' / 'style.css').read_text()}</style>", unsafe_allow_html=True)

with st.container(key="header"):
    title, demo_choice, navigation = st.columns([1.1, 0.75, 1.4], vertical_alignment="center")
    with title:
        st.title("Pixel GPT")
        st.caption("Pixels, one at a time")
    with demo_choice:
        demo = st.selectbox("Demo", ["Sprites", "Faces"], key="demo", label_visibility="collapsed")
    with navigation:
        pages = ["Generate", "Compare", "Results"] if demo == "Sprites" else ["Complete", "Compare", "Results"]
        page = st.segmented_control("Page", pages, default=pages[0], required=True,
                                    label_visibility="collapsed", key=f"{demo.lower()}_page")

if demo == "Sprites":
    from ui.sprites import render
else:
    from ui.faces import render

render(page)
