#%%
# =========================================================================
# Import des librairies
# =========================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#%%
# =========================================================================
# Import et la visualisation des données brutes
# =========================================================================

df = pd.read_csv("Wind_turbine_data.csv")
df.drop(["Theoretical_Power_Curve (KWh)"], axis = 1, inplace = True)
df.columns = ["Date", "Puissance (kW)", "Vitesse (m/s)", "Direction (°)"]
df["Date"] = pd.to_datetime(df["Date"], format = "%d %m %Y %H:%M")
df.set_index("Date", drop = True, inplace = True)

fig = plt.figure()
gs = fig.add_gridspec(3,2)

ax1 = fig.add_subplot(gs[0, :])
ax1.set_xlabel("Date")
ax1.set_ylabel("Puissance (kW)")
ax1.plot(df.index, df["Puissance (kW)"], c = "orange")

ax2 = fig.add_subplot(gs[1, 1])
ax2.set_xlabel("Vitesse (m/s)")
ax2.set_ylabel("Puissance (kW)")
ax2.scatter(df["Vitesse (m/s)"], df["Puissance (kW)"])

ax3 = fig.add_subplot(gs[2, 1])
ax3.set_xlabel("Direction (°)")
ax3.set_ylabel("Puissance (kW)")
ax3.scatter(df["Direction (°)"], df["Puissance (kW)"])

ax4 = fig.add_subplot(gs[1:, 0], projection = "3d")
ax4.scatter(df["Vitesse (m/s)"], df["Direction (°)"], df["Puissance (kW)"], c = "green")

plt.subplots_adjust(top=0.88,
bottom=0.11,
left=0.125,
right=0.9,
hspace=0.4,
wspace=0.02)

plt.show()

#%%
# =========================================================================
# Préparation du dataset
# =========================================================================

plt.figure()
sns.heatmap(df.corr(), annot = True, cmap = "inferno")
plt.show()

plt.figure()
sns.jointplot(x = df["Vitesse (m/s)"], y = df["Puissance (kW)"])
plt.show()
#%%
df = df[df["Puissance (kW)"] > 0.01]
df = df[(df["Vitesse (m/s)"] > 0.1) & (df["Vitesse (m/s)"] < 25.5)]

x = np.array(df["Vitesse (m/s)"]).reshape(len(df["Vitesse (m/s)"]), 1)
y = np.array(df["Puissance (kW)"]).reshape(len(df["Puissance (kW)"]), 1)

plt.figure()
sns.jointplot(x = np.ravel(x), y = np.ravel(y))
plt.show()

plt.close("all")

#%%
# =========================================================================
# Mise en place du modèle
# =========================================================================

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
import time

t_start = time.time()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#%% Optimisation de la méthode Lasso
lasso_pipeline = make_pipeline(PolynomialFeatures(),
                               RobustScaler(),
                               Lasso())
lasso_params = {
    "polynomialfeatures__degree": [4, 6, 8, 10, 12, 14, 16, 18, 20],
    "polynomialfeatures__include_bias": [False],
    "lasso__max_iter": [50000],
    "lasso__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
    }

lasso_grid = GridSearchCV(lasso_pipeline,
                          param_grid = lasso_params,
                          cv = 5,
                          n_jobs = -1)

lasso_grid.fit(x_train, y_train)

best_comb_lasso = {
    "R2": lasso_grid.best_score_,
    "Params": lasso_grid.best_params_
    }

lasso_opt = lasso_grid.best_estimator_
y_pred_lasso = lasso_opt.predict(x_test).reshape(len(x_test), 1)

#%% Optimisation de la méthode ElasticNet

elastic_pipeline = make_pipeline(PolynomialFeatures(),
                               RobustScaler(),
                               ElasticNet())
elastic_params = {
    "polynomialfeatures__degree": [4, 6, 8, 10, 12, 14, 16, 18, 20],
    "polynomialfeatures__include_bias": [False],
    "elasticnet__max_iter": [50000],
    "elasticnet__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
    "elasticnet__l1_ratio": [0.01, 0.1, 0.5, 1],
    }

elastic_grid = GridSearchCV(elastic_pipeline,
                          param_grid = elastic_params,
                          cv = 5,
                          n_jobs = -1)

elastic_grid.fit(x_train, y_train)

best_comb_elastic = {
    "R2": elastic_grid.best_score_,
    "Params": elastic_grid.best_params_
    }

elastic_opt = elastic_grid.best_estimator_
y_pred_elastic = elastic_opt.predict(x_test).reshape(len(x_test), 1)

#%% Optimisation de la méthode SVR

svr_pipeline = make_pipeline(RobustScaler(),
                             SVR())
svr_params = {
    "svr__C": [0.1, 1, 2, 5, 10],
    "svr__epsilon": [0.01, 0.1, 0.2, 0.5, 1],
    }

svr_grid = GridSearchCV(svr_pipeline,
                          param_grid = svr_params,
                          cv = 5,
                          n_jobs = -1)

svr_grid.fit(x_train, y_train)

best_comb_svr = {
    "R2": svr_grid.best_score_,
    "Params": svr_grid.best_params_
    }

svr_opt = svr_grid.best_estimator_
y_pred_svr = svr_opt.predict(x_test).reshape(len(x_test), 1)

#%% Visualisation des résultats

predictions = np.hstack(( x_test, y_pred_lasso, y_pred_elastic, y_pred_svr ))

predictions = predictions[np.argsort(predictions[:,0])]

plt.figure()
plt.scatter(x, y, label = "Données")
plt.plot(predictions[:,0], predictions[:,1], c = "orange",
          label = f"Lasso optimal R2 = {lasso_opt.score(x_test,y_test)}")
plt.plot(predictions[:,0], predictions[:,2], c = "purple",
          label = f"ElasticNet optimal R2 = {elastic_opt.score(x_test,y_test)}")
plt.plot(predictions[:,0], predictions[:,3], c = "purple",
          label = f"SVR optimal R2 = {svr_opt.score(x_test,y_test)}")
plt.legend()
plt.show()

elapsed_time = time.time() - t_start
print(f"Temps de calcul total = {elapsed_time}")

#%%
# ===========================================================================
# Import et préparation des données de vitesses de vent
# ===========================================================================

df_wind = pd.read_csv("Wind_speed_data.csv", sep = ";")
df_wind.drop(["orientation"], axis = 1, inplace = True)

df_wind["Date"] = pd.to_datetime(df_wind["Date"], format = "%d/%m/%y %H:%M")
df_wind = df_wind[(df_wind["Date"].dt.month == 4) & (df_wind["Date"].dt.year == 2020)]

sns.histplot(df_wind["wind_speed"])

#%%
# ===========================================================================
# Prédiction de l'énergie produite par les éoliennes mensuellement
# ===========================================================================

wind_matrix = np.array(df_wind)
power_pred = svr_opt.predict(wind_matrix[:,1].reshape(len(wind_matrix[:,1]),1))
energy_hourly_pred = power_pred*1 # on calcule des énergies sur base des puissances
total_energy_production = energy_hourly_pred.sum()*50

daily_conso = 180 # kWh par mois par habitant
nb_personnes = np.floor(total_energy_production/daily_conso)

print(f"Le champ de 50 éoliennes pourra subvenir aux besoins de {np.round((nb_personnes/68000000)*100, decimals = 2)}% de la population en France")
print(f"Ce qui représente un nombre d'habitants = {nb_personnes}")