import streamlit as st

def render_selectors():
    st.header("Exemples de Sélecteurs")
    
    # Selectbox simple
    option = st.selectbox(
        "Choisissez votre pays",
        ["France", "Belgique", "Suisse", "Canada"]
    )
    st.write(f"Vous avez choisi: {option}")
    
    # Multiselect
    options = st.multiselect(
        "Quelles langues parlez-vous?",
        ["Français", "Anglais", "Espagnol", "Allemand", "Italien"],
        ["Français"]
    )
    st.write(f"Langues sélectionnées: {', '.join(options)}")
    
    # Select slider
    niveau = st.select_slider(
        "Sélectionnez votre niveau",
        options=["Débutant", "Intermédiaire", "Avancé", "Expert"]
    )
    st.write(f"Votre niveau: {niveau}") 