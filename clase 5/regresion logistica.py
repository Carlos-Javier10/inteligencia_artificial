# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:07:43 2024

@author: Carlos-Javier

regrrsion logistica 
solo utiliza cualitativas
que hace-- ya no predice las ventas sino que 

aqui entra la probabilidad 
esta entre 0 y 1 y se responde a aciones muy censillas por eso no se usa las cuantitativas sino cualitativas 
un modelo de calsificacion que se llama knn 
cuando predigo variables categoricas 
debe a´parecer la palabra clasificacion 
las variables comparables se escla 
y escalar es ponerles en el mismo rango 


"""

###importacion de librerias 
import numpy as np ## Libraría para tratamiento de números y herramientas matemáticas (np Alias)
import matplotlib.pyplot as plt  ## Librería para representación gráfica (pyplot es una sublibrearía - gráficos) (plt Alias)
import pandas as pd  ## Librería para manipular datos (pd Alias)

## Importar el Dataset
dataset  = pd.read_csv("Social_Network_Ads.csv")

##matris de caracteristca 
X = dataset.iloc[:,[2.3]].values#variable independientes en este caso la edad y el salario estimado
Y = dataset.iloc[:,-1].values# variable dependientes 

#dividir el dataset en conjunto de entrenamiento y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 

#escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

##ajuasdtar el modelo de regresion logistioca en el conjutno de entrenmamiento
## aprender a predecir las clasificaciones 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

##prediccion de los resultados con el conjutno de test
y_pred = classifier.predict(X_test)

##analisis de resultados con la matris
##saver si la prediccion es b uena 
## se clacula la matriz de confusion sobre el modelo de testing 
## cuantas son las predicciones correctas 

from sklearn.metrics import confusion_matrix
conmet = confusion_matrix(y_test, y_pred)

##65 no comprAN Y 24 SI 
## 8 Y 3 CORRECVTAS 
##   cual es el porcenmtaje 
##para la comprobacion se suma en diagonal

##visualizae el algoritmo de train graficamente con los resudltados 
from matplotlib.colors
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,stop = X_set[:, ] ))


























