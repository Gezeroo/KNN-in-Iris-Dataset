import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# iris = pd.read_csv("./Iris.csv")

# # Seleciona as colunas de características
# caracteristicas = ["PetalWidthCm", "PetalLengthCm", "SepalWidthCm", "SepalLengthCm"]

# # Embaralha o database
# iris = iris.sample(frac=1, random_state=42).reset_index(drop=True)  # Embaralha e reseta os índices

# # Separa o database em conjunto de treinamento e teste
# tamanho_treinamento = round(len(iris) * 0.8)
# treinamento = iris[:tamanho_treinamento].copy()
# teste = iris[tamanho_treinamento:].copy()

# Carregar o conjunto de dados Iris
# Lê o database
iris = pd.read_csv("./Iris.csv")

# Seleciona as colunas de características
features = ["PetalWidthCm", "PetalLengthCm", "SepalWidthCm", "SepalLengthCm"]

# Embaralha o database
iris = iris.sample(frac=1, random_state=42)#.reset_index(drop=True)  # Embaralha e reseta os índices
X = iris.copy()  # Selecionar apenas petal length e petal width


# Encode the string labels into numerical values
y = iris.Species 
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

tamanho_treinamento = round(len(iris) * 0.8)
X_train = iris[:tamanho_treinamento][["Id", "PetalWidthCm", "PetalLengthCm", "SepalWidthCm", "SepalLengthCm"]].copy()
y_train = y[:tamanho_treinamento].copy()
X_test = iris[tamanho_treinamento:][["Id", "PetalWidthCm", "PetalLengthCm", "SepalWidthCm", "SepalLengthCm"]].copy()
y_test = y[tamanho_treinamento:].copy()

# Dividir o conjunto de dados em treinamento e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Padronizar os dados
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
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
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Encontrar o melhor valor de k
best_k = k_values[np.argmax(accuracies)]
print(f"Melhor valor de k: {best_k}")

# Treinar o modelo com o melhor valor de k
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_pred_best = best_knn.predict(X_test)

# Exibir o relatório de classificação
print("Relatório de classificação para o melhor valor de k:")
print(classification_report(y_test, y_pred_best))

# # Plotar o gráfico
# plt.figure(figsize=(10, 5))
# plt.plot(k_values, accuracies, marker='o')
# plt.title('Acurácia para cada valor de k (ímpares)')
# plt.xlabel('Número de vizinhos (k)')
# plt.ylabel('Acurácia')
# plt.xticks(range(0, 121, 10))
# plt.grid(True)
# plt.axvline(x=best_k, color='r', linestyle='--', label=f'Melhor k = {best_k} ({max(accuracies)*100:.2f}%)')
# plt.legend()
# plt.show()

plt.plot(k_values, accuracies)
plt.xlabel("Valor de k")
plt.ylabel("Precisão")
plt.title("Precisão do KNN para diferentes valores de k")
plt.show()
