import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def main():
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

        # ===========================================================================
        # Préparation du dataset
        # ===========================================================================
        st.title("Préparation du dataset")

        # Création d'une heatmap pour visualiser les corrélations entre les variables
        st.subheader("Création d'une heatmap pour visualiser les corrélations entre les variables")
        plt.figure()
        sns.heatmap(df.corr(), annot = True, cmap = "inferno")
        st.pyplot(plt)

        # Création d'un graphique joint pour visualiser la relation entre la vitesse et la puissance
        st.subheader("Création d'un graphique joint pour visualiser la relation entre la vitesse et la puissance")
        plt.figure()
        sns.jointplot(x = df["Vitesse (m/s)"], y = df["Puissance (kW)"])
        st.pyplot(plt)

        # Filtrage des valeurs de puissance supérieures à 0.01 kW pour éliminer les valeurs aberrantes
        df = df[df["Puissance (kW)"] > 0.01]

        # Préparation des données pour l'entraînement du modèle : conversion des colonnes en arrays numpy et reshape
        st.subheader("Préparation des données pour l'entraînement du modèle : conversion des colonnes en arrays numpy et reshape")
        x = np.array(df["Vitesse (m/s)"]).reshape(len(df["Vitesse (m/s)"]), 1)
        y = np.array(df["Puissance (kW)"]).reshape(len(df["Puissance (kW)"]), 1)
        st.write(x)
        st.write(y)

        # Normalisation des données avec RobustScaler pour réduire l'impact des valeurs aberrantes
        st.subheader("Normalisation des données avec RobustScaler pour réduire l'impact des valeurs aberrantes")
        scaler = RobustScaler()
        x = scaler.fit_transform(x)
        y = scaler.fit_transform(y)
        st.write(x)
        st.write(y)

        # Création d'un graphique joint pour visualiser la distribution des données normalisées
        st.subheader("Visualisation des données après normalisation")
        plt.figure()
        sns.jointplot(x = np.ravel(x), y = np.ravel(y))
        st.pyplot(plt) 

        # ===========================================================================
        # Mise en place du modèle
        # ===========================================================================
        st.title("Mise en place du modèle")

        # Création des caractéristiques polynomiales avec un degré de 12 pour capturer les relations non linéaires
        st.subheader("Création des caractéristiques polynomiales avec un degré de 12 pour capturer les relations non linéaires")
        poly_features = PolynomialFeatures(degree = 12, include_bias = False)
        x_poly = poly_features.fit_transform(x)
        st.write(x_poly)

        # Division du dataset en ensembles d'entraînement et de test
        st.subheader("Division du dataset en ensembles d'entraînement et de test")
        x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size = 0.2)
        st.write(x_train)
        st.write(x_test)
        st.write(y_train)
        st.write(y_test)

        # Entraînement du modèle Lasso
        st.subheader("Entraînement du modèle Lasso")
        lasso = Lasso(max_iter = 10000, alpha = 1e-3)
        lasso.fit(x_train, y_train)
        y_pred1 = lasso.predict(x_test)
        score1 = lasso.score(x_test, y_test)
        st.write(y_pred1)


        # Recherche des meilleurs hyperparamètres pour le modèle Lasso avec GridSearchCV
        st.subheader("Recherche des meilleurs hyperparamètres pour le modèle Lasso avec GridSearchCV")
        params_lasso = {
            "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        }
        lasso_grid = GridSearchCV(estimator = lasso,
        param_grid = params_lasso,
        scoring = "r2",
        cv = 5,
        n_jobs = -1)
        lasso_grid.fit(x_train, y_train)
        lasso_best = lasso_grid.best_estimator_
        y_pred1_best = lasso_best.predict(x_test)
        st.write(lasso_grid.best_params_)
        st.write(lasso_grid.best_score_)

        # Entraînement du modèle ElasticNet
        st.subheader("Entraînement du modèle ElasticNet")
        elastic = ElasticNet(max_iter = 10000, alpha = 1e-3, l1_ratio = 0.5)
        elastic.fit(x_train, y_train)
        y_pred2 = elastic.predict(x_test)
        score2 = elastic.score(x_test, y_test)
        st.write(y_pred2)

        # Recherche des meilleurs hyperparamètres pour le modèle ElasticNet avec GridSearchCV
        st.subheader("Recherche des meilleurs hyperparamètres pour le modèle ElasticNet avec GridSearchCV")
        params_elastic = {
            "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            "l1_ratio": [1e-2, 1e-1, 0.5, 1, 5]
        }
        elastic_grid = GridSearchCV(estimator = elastic,
        param_grid = params_elastic,
        scoring = "r2",
        cv = 5,
        n_jobs = -1)
        elastic_grid.fit(x_train, y_train)
        elastic_best = elastic_grid.best_estimator_
        y_pred2_best = elastic_best.predict(x_test)
        st.write(elastic_grid.best_params_)
        st.write(elastic_grid.best_score_)

        # Empiler verticalement les données de test et les prédictions des deux modèles
        st.subheader("Empiler verticalement les données de test et les prédictions des deux modèles")
        predictions = np.vstack(( x_test[:,0], y_pred2, y_pred2_best )).T
        predictions = predictions[np.argsort(predictions[:,0])]
        st.write(predictions)

        # Création d'un graphique joint pour visualiser les prédictions des deux modèles
        st.subheader("Création d'un graphique joint pour visualiser les prédictions des deux modèles")
        plt.figure()
        plt.scatter(x[:,0], y, label = "Données")
        plt.plot(predictions[:,0], predictions[:,1], c = "orange", label = f"Lasso R2 = {score1}")
        plt.plot(predictions[:,0], predictions[:,2], c = "purple", label = f"ElasticNet R2 = {score2}")
        plt.legend()
        st.pyplot(plt)


    

    except FileNotFoundError:
        st.error("Le fichier de données n'a pas été trouvé. Veuillez vérifier le chemin d'accès.")
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du chargement des données: {str(e)}")

  

if __name__ == "__main__":
    main() 
