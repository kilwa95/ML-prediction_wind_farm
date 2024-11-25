import streamlit as st
import numpy as np

def render_sliders():
    st.header("Exemples de Sliders")
    
    # Slider simple
    age = st.slider("Quel âge avez-vous?", 0, 120, 25)
    st.write(f"J'ai {age} ans")
    
    # Range slider
    valeurs = st.slider(
        "Sélectionnez une plage",
        0.0, 100.0, (25.0, 75.0)
    )
    st.write(f"Plage sélectionnée: {valeurs}")
    
    # Slider avec step
    nombre = st.slider(
        "Choisissez un nombre pair",
        0, 100, 50, step=2
    )
    st.write(f"Nombre sélectionné: {nombre}") 