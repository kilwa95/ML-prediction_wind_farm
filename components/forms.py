import streamlit as st
import pandas as pd

def render_contact_form():
    st.header("Formulaire de Contact")
    
    with st.form("contact_form"):
        # Informations personnelles
        col1, col2 = st.columns(2)
        with col1:
            nom = st.text_input("Nom")
            email = st.text_input("Email")
            telephone = st.text_input("Téléphone")
        with col2:
            prenom = st.text_input("Prénom")
            age = st.number_input("Âge", min_value=0, max_value=120, value=25)
            pays = st.selectbox("Pays", ["France", "Belgique", "Suisse", "Canada"])
        
        # Message
        message = st.text_area("Votre message")
        
        # Préférences
        st.subheader("Préférences")
        newsletter = st.checkbox("S'abonner à la newsletter")
        contact_method = st.radio("Méthode de contact préférée", ["Email", "Téléphone"])
        
        # Bouton de soumission
        submitted = st.form_submit_button("Envoyer")
        
        if submitted:
            st.success("Formulaire envoyé avec succès!")
            st.write("Récapitulatif:")
            st.write({
                "Nom": nom,
                "Prénom": prenom,
                "Email": email,
                "Téléphone": telephone,
                "Âge": age,
                "Pays": pays,
                "Message": message,
                "Newsletter": newsletter,
                "Méthode de contact": contact_method
            })

def render_reservation_form():
    st.header("Formulaire de Réservation")
    
    with st.form("reservation_form"):
        # Date et heure
        col1, col2 = st.columns(2)
        with col1:
            date = st.date_input("Date de réservation")
            nb_personnes = st.number_input("Nombre de personnes", 1, 10, 2)
        with col2:
            heure = st.time_input("Heure")
            type_table = st.selectbox("Type de table", ["Intérieur", "Terrasse", "Salon privé"])
        
        # Options supplémentaires
        st.subheader("Options")
        options = st.multiselect(
            "Services supplémentaires",
            ["Menu végétarien", "Gâteau d'anniversaire", "Bouteille de champagne", "Décoration spéciale"]
        )
        
        notes = st.text_area("Notes spéciales")
        
        # Soumission
        submitted = st.form_submit_button("Réserver")
        
        if submitted:
            st.success("Réservation effectuée!")
            st.write("Détails de la réservation:")
            st.write({
                "Date": date,
                "Heure": heure,
                "Nombre de personnes": nb_personnes,
                "Type de table": type_table,
                "Options": options,
                "Notes": notes
            })

def render_survey_form():
    st.header("Questionnaire de Satisfaction")
    
    with st.form("survey_form"):
        # Évaluation
        satisfaction = st.slider("Niveau de satisfaction global", 0, 10, 5)
        
        # Questions à choix multiples
        st.subheader("Évaluation détaillée")
        qualite = st.select_slider(
            "Qualité du service",
            options=["Très insatisfait", "Insatisfait", "Neutre", "Satisfait", "Très satisfait"]
        )
        
        aspects = st.multiselect(
            "Quels aspects avez-vous le plus appréciés?",
            ["Interface", "Facilité d'utilisation", "Design", "Performance", "Support client"]
        )
        
        # Commentaires
        ameliorations = st.text_area("Suggestions d'amélioration")
        
        # Recommandation
        recommandation = st.radio(
            "Recommanderiez-vous notre service?",
            ["Oui, certainement", "Peut-être", "Non"]
        )
        
        # Soumission
        submitted = st.form_submit_button("Envoyer le questionnaire")
        
        if submitted:
            st.success("Merci pour votre retour!")
            st.write("Vos réponses:")
            st.write({
                "Satisfaction": satisfaction,
                "Qualité du service": qualite,
                "Aspects appréciés": aspects,
                "Suggestions": ameliorations,
                "Recommandation": recommandation
            })

def render_forms():
    form_type = st.selectbox(
        "Choisissez le type de formulaire",
        ["Contact", "Réservation", "Questionnaire"]
    )
    
    if form_type == "Contact":
        render_contact_form()
    elif form_type == "Réservation":
        render_reservation_form()
    else:
        render_survey_form() 