import streamlit as st

def render_buttons():
    st.header("Exemples de Boutons")
    
    # Bouton simple
    if st.button("Cliquez-moi"):
        st.write("Bouton cliqué!")
    
    # Bouton de téléchargement
    st.download_button(
        label="Télécharger les données",
        data="Contenu du fichier",
        file_name="donnees.txt"
    )
    
    # Checkbox
    if st.checkbox("Afficher/Masquer"):
        st.write("Contenu visible!")
    
    # Radio buttons
    genre = st.radio(
        "Choisissez votre genre",
        ["Homme", "Femme", "Non-binaire"]
    )
    st.write(f"Vous avez choisi: {genre}") 