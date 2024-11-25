import streamlit as st
import os

def render_videos():
    st.header("Visualisation de Vidéos")
    
    # Upload de vidéo
    uploaded_video = st.file_uploader("Choisissez une vidéo", type=['mp4', 'mov', 'avi'])
    if uploaded_video is not None:
        # Sauvegarde temporaire de la vidéo
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
            
        # Affichage de la vidéo
        st.video(video_path)
        
        # Options de lecture
        st.subheader("Options de lecture")
        start_time = st.slider("Temps de début (secondes)", 0, 60, 0)
        
        # Nettoyage du fichier temporaire
        if os.path.exists(video_path):
            os.remove(video_path)
    
    # Exemple avec une vidéo depuis une URL
    st.subheader("Vidéo depuis une URL")
    video_url = st.text_input(
        "Entrez l'URL d'une vidéo (YouTube, Vimeo, etc.)",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )
    if video_url:
        st.video(video_url) 