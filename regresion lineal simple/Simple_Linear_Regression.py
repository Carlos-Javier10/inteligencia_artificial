# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 18:14:30 2024

@author: Javier Gómez R.

SIMPLE LINEAR REGRESSION
"""

## Análisis de sueldo de empleados basado en los años de experiencia
## Vamos a predecir el sueldo con el modelo lineal
## La fuente de datos es "Salary_Data.csv"

## Importar las librearías
import numpy as np ## Libraría para tratamiento de números y herramientas matemáticas (np Alias)
import matplotlib.pyplot as plt  ## Librería para representación gráfica (pyplot es una sublibrearía - gráficos) (plt Alias)
import pandas as pd  ## Librería para manipular datos (pd Alias)


## Importar el Dataset
dataset  = pd.read_csv("Salary_Data.csv")

## Saber con que filas y columnas se trabajará de la Var. Predictora
## Variables independientes "X" tiene 3 Col y 10 Fil
## El menos 1 es la última columna y es era para predecir

X = dataset.iloc[:,:-1].values ## iloc Localizar elementos de filas y columnas por posición (i - Index -- loc Localization)
## Los primeros : puntos desde la fila inicio hasta la fila fin
## Los dos siguientes : puntos desde  la columna inicio hasta el fin
## El negativo es excepto la últimva
## .values solo los valores del arreglo

## Obtener datos de la columna a Predecir
y = dataset.iloc[:, 1].values ## Los dos primeros puntos : filas y los dos punto siguiente columnas

'''

## Tratamiento de los NA's (qué es un NAN)
## Libraría para limipieza e imputación de datos
## from sklearn.preprocessing import Imputer línea del código original con Imputer no funciona
### Original from sklearn.preprocesssing import Imputer
from sklearn.impute import SimpleImputer

## Creación de un objeto de la clase Imputer
## Que vamos a buscar y que estratégia utilizaremos
## axis = 0 aplicando la media por columna

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean") ## Estratégia reemplazar por la media de la columna
##imputer.fit(X)  ## Con esto es la media a todas las columnas
## Seleccionar los NAN
imputer = imputer.fit(X[:, 1:3]) ## tomamos la columnas de la 1 a la 2 ya que es n-1 (3-1=2) -- ojo si se pone solo Fit de X
## Colocar los valores de la media de NAN
X[:, 1:3] = imputer.transform(X[:, 1:3]) ## Ejecuta la ejecución de la imputación Todas las filas y solo 2 columnas edad y sueldo

## Codificar datos categóricos


from sklearn import preprocessing 
## Primero la variable categógica de las variables independientes
labelencoder_X = preprocessing.LabelEncoder();
## labelencoder_X.fit_transform(X[:, 0]) -> Matriz de valores con 0, 1, 2, etc
## Estas son variables Dummy - La Idea es un One Hot Encode en 1 sola columna
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) ## Codifica y convierte a datos numéricos

from sklearn.preprocessing import OneHotEncoder, LabelEncoder  ## Llamamos a las Variables Dummy
from sklearn.compose import ColumnTransformer

## Primero a número y luego a variable Dummy --> OneHotEncoder or Vector -> Traducir una categoría que no tiene orden
## a un conjunto de tantas columnas como categorías existen
onehotencoder = ColumnTransformer([('one_hot_encoder',OneHotEncoder(categories='auto'), [0])], remainder='passthrough')

## variable Dummy para variable X
X = np.array(onehotencoder.fit_transform(X),dtype=float) ## Genera una matriz de características

## para variable Dicotómica - Solo es necesario un Label Encoder y no One Hot Encoder - BooL Value 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

################-------------------------------
####### Esto normalmente quedaría en la plantilla principal
'''


## dividir el Dataset en conjunto entrenamiento y conjunto testing
from sklearn.model_selection import train_test_split
## Esto devolverá 4 variables y hay que crearlas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) ##ramdom state es para los valores de test split y size
## Random es para que siempre tome los mismos datos
## cuando hay outlawers se usa la mediana 

## Escalar los datos -> Mostrar lámina de Standarización y Normalización
## Para que no exista outloyers y existes 2 formas
## Xstan = (x- mean(x))/standar deviation (Standarización) -- en Relación a la media
## Xnorm 0 (x - min(x))/(max(x)-min(x))   (Normalización)  -- Pequeño 0 mas Gránde 1 y el resto escalado de forma lineal

### Escalado de variables: es el que permitia que dos variables se comparen 
## estandarizacion y normalizacion: nos permite llenar los nan 
### Procesamos la estandarización
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test) ## el conjunto X de test se escalará con la misma transformación de los de entrenamiento.


## En el caso de la Regresión Lineal Simple -- normalmente no rqeuiere Escalado
## ESCALADO -> Normalización o Estandarización

''' MODELO DE REGRESION '''
### Creación del modelo de regresión lineal con los datos de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)  ## Importante tener el nmismo número de filas y aprendió lo que tenía como dato de entrenamiento

### Predecir el modelo y probarlo con test
y_pred = regression.predict(X_test)

## Visualización de resultados con el Entrenamiento
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs. Años de Experiencia (Conjunto de entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo ($)")
plt.show()

## Visualización de resultados con el test
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_test, regression.predict(X_test), color = "blue")
plt.title("Sueldo vs. Años de Experiencia (Conjunto de test)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo ($)")
plt.show()















