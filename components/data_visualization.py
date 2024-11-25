import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st


def render_data_visualization():
    # Titre de la page
    st.title("Visualisation des Données")

    # Chargement des données (exemple avec un fichier CSV fictif)
    try:
        df = pd.read_csv('data/Wind_turbine_data.csv')
        
        # Affichage des premières lignes du dataset
        st.subheader("Aperçu des données brutes")
        st.dataframe(df.head())

        # Renommage des colonnes pour plus de clarté
        st.subheader("Renommage des colonnes pour plus de clarté")
        df.drop(["Theoretical_Power_Curve (KWh)"], axis = 1, inplace = True)
        df.columns = ["Date", "Puissance (kW)", "Vitesse (m/s)", "Direction (°)"]
        st.dataframe(df.head())

        # Conversion de la colonne Date en datetime et définition comme index
        st.subheader("Conversion de la colonne Date en datetime et définition comme index")
        df["Date"] = pd.to_datetime(df["Date"], format = "%d %m %Y %H:%M")
        df.set_index("Date", drop = True, inplace = True)
        st.dataframe(df.head())

        # Création d'une figure avec une grille de 3 lignes et 2 colonnes
        fig = plt.figure()
        gs = fig.add_gridspec(3,2)

        # Création du premier subplot pour afficher la puissance en fonction du temps
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Puissance (kW)")
        ax1.plot(df.index, df["Puissance (kW)"], c = "orange")

        # Création du second subplot pour afficher la puissance en fonction de la vitesse
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.set_xlabel("Vitesse (m/s)")
        ax2.set_ylabel("Puissance (kW)")
        ax2.scatter(df["Vitesse (m/s)"], df["Puissance (kW)"])
       

        # Création du troisième subplot pour afficher la puissance en fonction de la direction
        ax3 = fig.add_subplot(gs[2, 1])
        ax3.set_xlabel("Direction (°)")
        ax3.set_ylabel("Puissance (kW)")
        ax3.scatter(df["Direction (°)"], df["Puissance (kW)"])

        # Création du quatrième subplot pour afficher la puissance en fonction de la vitesse et de la direction
        ax4 = fig.add_subplot(gs[1:, 0], projection = "3d")
        ax4.scatter(df["Vitesse (m/s)"], df["Direction (°)"], df["Puissance (kW)"], c = "green")

        # Ajustement des marges entre les sous-graphiques
        st.subheader("Ajustement des marges entre les sous-graphiques")
        plt.subplots_adjust(top=0.88,
        bottom=0.11,
        left=0.125,
        right=0.9,
        hspace=0.4,
        wspace=0.02)    
        st.pyplot(fig)

        # Statistiques descriptives
        st.subheader("Statistiques descriptives")
        st.write(df.describe())
        
        # Vérification des valeurs manquantes
        st.subheader("Valeurs manquantes")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])
        
    except FileNotFoundError:
        st.error("Le fichier de données n'a pas été trouvé. Veuillez vérifier le chemin d'accès.")
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du chargement des données: {str(e)}")
