import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

#Cargar conjunto de datos Iris
iris = datasets.load_iris()

#Convertir los datos a tipo DataFrame
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=["species"])

#Separar el conjunto de datos en datos de prueba y de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train, y_test = y_train["species"], y_test["species"]

#Crear el modelo
model = RandomForestClassifier(n_estimators = 100, random_state = 42)
#Ajustar el modelo a los datos
model.fit(X_train, y_train)

#Hacer predicciones con el conjunto de prueba
Y_hat = model.predict(X_test) 

#Guardar todos los datos en una tabla para mostrarla 
printData = X_test
printData["Valor real"] = y_test
printData["Valor predicho"] = Y_hat

#Calcular las métricas de desempeño del modelo
accuracy = accuracy_score(y_test, Y_hat)

#Imprimir información
print("Modelo que predice la especie de flor Iris")
print("0: Iris setosa")
print("1: Iris versicolor")
print("2: Iris virginica")
print(" ")
print("A continuación de muestra una tabla con algunos de los datos que se predijeron, su valor real y su valor predicho")
print(printData[0:5])

print("Desempeño del modelo")
print("Accuracy:", accuracy)