import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def render_line_chart():
    # Création de données exemple
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    df = pd.DataFrame({'x': x, 'y': y})
    
    # Création du graphique ligne
    fig = px.line(df, x='x', y='y', title='Graphique Sinusoïdal')
    st.plotly_chart(fig)

def render_scatter_plot():
    # Création de données aléatoires
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.random.normal(0, 1, 100),
        'y': np.random.normal(0, 1, 100),
        'groupe': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    fig = px.scatter(df, x='x', y='y', color='groupe',
                    title='Nuage de Points par Groupe')
    st.plotly_chart(fig)

def render_bar_chart():
    # Données exemple
    categories = ['A', 'B', 'C', 'D']
    values = [20, 14, 23, 25]
    
    fig = go.Figure(data=[
        go.Bar(name='Valeurs', x=categories, y=values)
    ])
    
    fig.update_layout(title='Graphique à Barres')
    st.plotly_chart(fig)

def render_charts():
    st.header("Visualisations Interactives")
    
    chart_type = st.selectbox(
        "Choisissez le type de graphique",
        ["Ligne", "Nuage de points", "Barres"]
    )
    
    if chart_type == "Ligne":
        render_line_chart()
    elif chart_type == "Nuage de points":
        render_scatter_plot()
    else:
        render_bar_chart() 