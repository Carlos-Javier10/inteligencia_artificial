## Analizamos empresas startups para decidir en cuál invertir según varios criterios.
## La idea es ver si hay alguna relación entre ganancias y ubicación y predecir cuánto gana la empresa.

# Primero, importamos las librerías que necesitamos
import numpy as np  # para operaciones matemáticas
import matplotlib.pyplot as plt  # para hacer gráficos bonitos
import pandas as pd  # para trabajar con datos en forma de tablas

# Cargamos el dataset
dataset = pd.read_csv("Position_Salaries.csv")

# Seleccionamos la columna independiente "X" y la variable dependiente "y"
X = dataset.iloc[:, 1:2].values  # Aquí agarramos todas las filas y la segunda columna
y = dataset.iloc[:, -1].values  # Y en 'y' ponemos la última columna, que es lo que queremos predecir

# Ahora importamos el modelo de regresión lineal
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)  # Entrenamos el modelo con los datos que tenemos en X e y

# Configuración de regresión polinómica
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)  # Elegimos que el modelo sea de grado 2
X_poly = poly_reg.fit_transform(X)  # Transformamos X a su versión polinómica

# Ajustamos los datos a la regresión polinómica
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Graficamos los resultados de la regresión lineal
plt.scatter(X, y, color="red")  # Puntos originales en rojo
plt.plot(X, lin_reg.predict(X), color="blue")  # Línea de regresión lineal en azul
plt.title("Modelo de regresión lineal")
plt.xlabel("Posición del empleador")
plt.ylabel("Sueldo en $")
plt.show()

# Graficamos los resultados de la regresión polinómica
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg2.predict(X_poly), color="blue")  # Línea de regresión polinómica en azul
plt.title("Modelo de regresión polinómica")
plt.xlabel("Posición del empleador")
plt.ylabel("Sueldo en $")
plt.show()

# Codificación de variables categóricas en el dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])  # Codificamos la columna 3 como numérica

onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [3])], remainder='passthrough')
X = np.array(onehotencoder.fit_transform(X), dtype=float)  # Convertimos a un formato entendible para el modelo

# Eliminamos la primera columna de variables ficticias para evitar duplicidad
X = X[:, 1:]

# Dividimos el dataset en entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # 80% entrenamiento, 20% test

# Escalado de variables (esto evita que algunos valores sobresalgan mucho)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)  # Escalamos test con los mismos valores de entrenamiento

# Modelo de regresión múltiple
regression = LinearRegression()
regression.fit(X_train, y_train)  # Ajustamos el modelo

# Predicción con datos de prueba
y_pred = regression.predict(X_test)

# Eliminación hacia atrás para optimizar el modelo
import statsmodels.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)  # Agregamos columna de 1s para el intercepto
SL = 0.05  # Nivel de significancia

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regression_OLS = sm.OLS(y, X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(y, X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regression_OLS = sm.OLS(y, X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regression_OLS = sm.OLS(y, X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3]]
regression_OLS = sm.OLS(y, X_opt).fit()
regression_OLS.summary()

# Visualización de resultados de entrenamiento
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regression.predict(X_train), color="blue")
plt.title("Sueldo vs. Años de Experiencia (Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo ($)")
plt.show()

# Visualización de resultados de prueba
plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, regression.predict(X_test), color="blue")
plt.title("Sueldo vs. Años de Experiencia (Test)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo ($)")
plt.show()
