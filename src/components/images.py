import streamlit as st
from PIL import Image
import os

def render_images():
    st.header("Visualisation d'Images")
    
    # Upload d'image
    uploaded_image = st.file_uploader("Choisissez une image", type=['png', 'jpg', 'jpeg'])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        # Affichage des informations de l'image
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Image téléchargée")
        with col2:
            st.write("Informations sur l'image:")
            st.write(f"Format: {image.format}")
            st.write(f"Taille: {image.size}")
            st.write(f"Mode: {image.mode}")
        
        # Options de transformation
        st.subheader("Transformations")
        option = st.selectbox(
            "Choisissez une transformation",
            ["Original", "Noir et Blanc", "Rotation"]
        )
        
        if option == "Noir et Blanc":
            st.image(image.convert("L"), caption="Version Noir et Blanc")
        elif option == "Rotation":
            angle = st.slider("Angle de rotation", -180, 180, 0)
            st.image(image.rotate(angle), caption=f"Rotation de {angle}°") 