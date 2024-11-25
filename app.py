import streamlit as st
from components.header import render_header
from components.sidebar import render_sidebar
from components.sliders import render_sliders


def main():
    render_header("Mon Application Streamlit")
    selected = render_sidebar()

    if selected == "Accueil":
        render_sliders()
  

if __name__ == "__main__":
    main() 
