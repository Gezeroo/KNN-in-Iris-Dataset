import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Carregar o conjunto de dados Iris
iris = pd.read_csv("./Iris.csv")

# Seleciona as colunas de características
features = ["PetalWidthCm", "PetalLengthCm"]

# Embaralha o database
iris = iris.sample(frac=1, random_state=130)

# Encode the string labels into numerical values
y = iris.Species 
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
tamanho_treinamento = round(len(iris) * 0.8)
X_train = iris[:tamanho_treinamento][["Id", "PetalWidthCm", "PetalLengthCm"]].copy()
y_train = y[:tamanho_treinamento].copy()
X_test = iris[tamanho_treinamento:][["Id", "PetalWidthCm", "PetalLengthCm"]].copy()
y_test = y[tamanho_treinamento:].copy()

# Calcula a média e o desvio padrão com base apenas no conjunto de treinamento
media_treinamento = X_train[features].mean()
desvio_padrao_treinamento = X_train[features].std()

X_train[features] = (X_train[features] - media_treinamento) / desvio_padrao_treinamento #normaliza os dados de treino
X_test[features] = (X_test[features] - media_treinamento) / desvio_padrao_treinamento #normaliza os dados de teste

# Avaliar a acurácia para diferentes valores de k ímpares de 1 a 120
k_values = range(1, 121, 2)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train[features], y_train)
    y_pred = knn.predict(X_test[features])
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append((k, accuracy))

# Encontra o valor de k com a melhor acurácia
best_k, best_accuracy = max(accuracies, key=lambda item: item[1])

print(f"Melhor valor de k: {best_k} com acurácia de: {best_accuracy:.4f}")

# Treina o modelo com o melhor valor de k
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train[features], y_train)
y_pred_best = best_knn.predict(X_test[features])

# Exibe o relatório de classificação
print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))

print(f"Melhor valor de k: {best_k}")
print(f"Melhor acurácia: {best_accuracy:.4f}")

# Plotar a acurácia para diferentes valores de k
k_values, accuracies = zip(*accuracies)
plt.plot(k_values, accuracies)
plt.xlabel("Valor de k")
plt.ylabel("Acurácia")
plt.title("Acurácia do KNN para diferentes valores de k")
plt.show()
