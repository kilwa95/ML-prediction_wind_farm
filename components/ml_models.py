import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px

def render_classification():
    st.subheader("Classification avec Scikit-learn")
    
    # Génération de données synthétiques
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                             n_informative=2, random_state=1, 
                             n_clusters_per_class=1)
    
    # Création du DataFrame
    df = pd.DataFrame(data=X, columns=['Feature 1', 'Feature 2'])
    df['Target'] = y
    
    # Affichage des données
    st.write("Aperçu des données:")
    st.dataframe(df.head())
    
    # Visualisation
    fig = px.scatter(df, x='Feature 1', y='Feature 2', color='Target',
                    title='Visualisation des données')
    st.plotly_chart(fig)
    
    # Choix du modèle
    model_type = st.selectbox(
        "Choisissez le modèle",
        ["Random Forest", "Régression Logistique"]
    )
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Entraînement
    if st.button("Entraîner le modèle"):
        if model_type == "Random Forest":
            model = RandomForestClassifier()
        else:
            model = LogisticRegression()
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Affichage des résultats
        st.write(f"Précision du modèle: {accuracy_score(y_test, y_pred):.2f}")
        st.text("Rapport de classification:")
        st.text(classification_report(y_test, y_pred))

def render_regression():
    st.subheader("Régression")
    
    # Génération de données synthétiques
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2 * X + 1 + np.random.randn(100, 1) * 0.5
    
    # Création du DataFrame
    df = pd.DataFrame(data=X, columns=['X'])
    df['y'] = y
    
    # Visualisation
    fig = px.scatter(df, x='X', y='y', title='Données de régression')
    st.plotly_chart(fig)
    
    # Interface de prédiction
    input_value = st.slider("Entrez une valeur X pour prédiction", 0.0, 10.0, 5.0)
    if st.button("Prédire"):
        prediction = 2 * input_value + 1
        st.write(f"Prédiction pour X = {input_value}: {prediction:.2f}")

def render_ml_models():
    st.header("Machine Learning")
    
    ml_type = st.selectbox(
        "Choisissez le type d'analyse",
        ["Classification", "Régression"]
    )
    
    if ml_type == "Classification":
        render_classification()
    else:
        render_regression() 