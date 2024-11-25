import streamlit as st
import pandas as pd
from components.database import run_query

def render_data_viewer():
    st.header("Visualisation des données MySQL")
    
    # Exemple de requête SELECT
    try:
        # Récupération des tables de la base de données
        tables = run_query("SHOW TABLES")
        table_names = [table[0] for table in tables]
        
        selected_table = st.selectbox("Sélectionnez une table", table_names)
        
        if selected_table:
            # Récupération des données de la table sélectionnée
            rows = run_query(f"SELECT * FROM {selected_table}")
            # Récupération des noms de colonnes
            columns = run_query(f"SHOW COLUMNS FROM {selected_table}")
            column_names = [column[0] for column in columns]
            
            # Création du DataFrame
            df = pd.DataFrame(rows, columns=column_names)
            st.dataframe(df)
            
            # Affichage des statistiques
            st.subheader("Statistiques")
            st.write(f"Nombre total d'enregistrements : {len(df)}")
            
    except Exception as e:
        st.error(f"Erreur de connexion à la base de données : {str(e)}") 