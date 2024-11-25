import streamlit as st

def render_home():
    st.title("Prédiction de Production d'Énergie Éolienne")
    
    st.markdown("""
    ## À propos du projet
    
    Cette application permet d'analyser et de prédire la production d'électricité d'un champ de 50 éoliennes en utilisant 
    des techniques de Machine Learning.
    
    ### Objectifs
    - Analyser les données historiques de production
    - Créer des modèles prédictifs précis
    - Optimiser la production d'énergie éolienne
    
    ### Fonctionnalités principales
    1. **Visualisation des données**
       - Graphiques interactifs
       - Tableaux de données détaillés
       - Analyses statistiques
    
    2. **Modèles de Machine Learning**
       - Classification
       - Régression
       - Optimisation des paramètres
    
    3. **Prédictions**
       - Prévisions de production
       - Analyses de performance
       - Recommandations d'optimisation
    
    ### Technologies utilisées
    - Streamlit pour l'interface utilisateur
    - Scikit-learn pour les modèles ML
    - Plotly pour les visualisations
    - MySQL pour le stockage des données
    """)
    
    # Affichage des métriques exemple
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Précision moyenne", value="92%", delta="↑ 2%")
    with col2:
        st.metric(label="Éoliennes analysées", value="50")
    with col3:
        st.metric(label="Production mensuelle", value="1.2 GWh", delta="↑ 5%")
    
    st.info("""
    💡 **Comment utiliser l'application ?**
    
    Utilisez la barre latérale pour naviguer entre les différentes fonctionnalités :
    - Visualisation des données
    - Modèles de prédiction
    - Analyses statistiques
    """)

if __name__ == "__main__":
    render_home()
