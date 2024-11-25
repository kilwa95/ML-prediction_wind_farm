import streamlit as st
import pandas as pd
import numpy as np

def render_tables():
    st.header("Tableaux de Données")
    
    # Exemple 1: Tableau simple
    st.subheader("1. Tableau de données basique")
    df_simple = pd.DataFrame({
        'Nom': ['Pierre', 'Marie', 'Jean', 'Sophie'],
        'Age': [25, 30, 35, 28],
        'Ville': ['Paris', 'Lyon', 'Marseille', 'Toulouse']
    })
    st.dataframe(df_simple)
    
    # Exemple 2: Tableau avec données aléatoires
    st.subheader("2. Tableau avec données aléatoires")
    np.random.seed(42)
    df_random = pd.DataFrame(
        np.random.randn(10, 4),
        columns=['A', 'B', 'C', 'D']
    )
    st.dataframe(df_random.style.highlight_max(axis=0))
    
    # Exemple 3: Tableau avec statistiques
    st.subheader("3. Statistiques descriptives")
    st.write("Statistiques du tableau précédent:")
    st.dataframe(df_random.describe())
    
    # Exemple 4: Tableau éditable
    st.subheader("4. Tableau éditable")
    df_editable = pd.DataFrame({
        'Produit': ['A', 'B', 'C'],
        'Prix': [10, 20, 30],
        'Stock': [100, 50, 75]
    })
    edited_df = st.data_editor(df_editable)
    
    # Exemple 5: Tableau avec filtres
    st.subheader("5. Tableau avec métriques")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Moyenne Prix", f"{edited_df['Prix'].mean():.2f} €")
    with col2:
        st.metric("Total Stock", f"{edited_df['Stock'].sum()}")
    with col3:
        st.metric("Nb Produits", len(edited_df)) 