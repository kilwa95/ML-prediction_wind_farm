import streamlit as st

def render_theme_customization():
    st.header("Personnalisation du Thème")
    
    # Sélection du mode clair/sombre
    theme_mode = st.radio(
        "Mode d'affichage",
        ["Clair", "Sombre"],
        horizontal=True
    )
    
    # Sélection des couleurs principales
    col1, col2 = st.columns(2)
    with col1:
        primary_color = st.color_picker(
            "Couleur principale",
            "#FF4B4B"
        )
        background_color = st.color_picker(
            "Couleur de fond",
            "#FFFFFF" if theme_mode == "Clair" else "#0E1117"
        )
    
    with col2:
        secondary_color = st.color_picker(
            "Couleur secondaire",
            "#4B4BFF"
        )
        text_color = st.color_picker(
            "Couleur du texte",
            "#000000" if theme_mode == "Clair" else "#FFFFFF"
        )

    # Configuration du thème
    theme_config = {
        "primaryColor": primary_color,
        "backgroundColor": background_color,
        "secondaryBackgroundColor": secondary_color,
        "textColor": text_color,
        "font": "sans serif"
    }
    
    # Aperçu du thème
    st.subheader("Aperçu des composants")
    
    # Exemple de boutons
    st.button("Bouton Principal")
    
    # Exemple de slider
    st.slider("Exemple de Slider", 0, 100, 50)
    
    # Exemple de texte
    st.markdown("""
    ### Exemple de titre
    Voici un exemple de texte pour voir le rendu avec les couleurs choisies.
    
    - Point 1
    - Point 2
    - Point 3
    """)
    
    # Code pour appliquer le thème
    st.code(f"""
    [theme]
    primaryColor="{primary_color}"
    backgroundColor="{background_color}"
    secondaryBackgroundColor="{secondary_color}"
    textColor="{text_color}"
    font="sans serif"
    """)
    
    st.info("Pour appliquer ce thème, créez un fichier `.streamlit/config.toml` avec la configuration ci-dessus.") 