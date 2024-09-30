import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from collections import defaultdict


class pontoIris:  # classe de cada ponto de análise (válido para treinamento e teste)
    dists = []

    def __init__(self, x, y, type, id):
        self.id = id
        self.x = x
        self.y = y
        self.type = type


def generate_pontos(ids, xs, ys, species):
    pontos = []
    for i in range(len(ids)):
        pontos.append(pontoIris(xs[i], ys[i], species[i], ids[i]))
    return pontos


def distancias(pontos_treinamento, ponto):
    dists = []
    for i in range(len(pontos_treinamento)):
        dists.append(
            (
                pontos_treinamento[i],
                math.dist(
                    (pontos_treinamento[i].x, pontos_treinamento[i].y),
                    (ponto.x, ponto.y),
                ),
            )
        )
    dists.sort(key=lambda tup: tup[1])
    ponto.dists = dists[1:]


# Lê o database
iris = pd.read_csv("./Iris.csv")

# Seleciona as colunas de características
features = ["PetalWidthCm", "PetalLengthCm", "SepalWidthCm", "SepalLengthCm"]

# Embaralha o database
iris = iris.sample(frac=1)  # random_state para reprodutibilidade

# Separa o database em conjunto de treinamento e teste
tam_treinamento = round(len(iris) * 0.8)
treinamento = iris[:tam_treinamento]
teste = iris[tam_treinamento:]

# Calcula a média e o desvio padrão com base apenas no conjunto de treinamento
media_treinamento = treinamento[features].mean()
std_treinamento = treinamento[features].std()

# Normaliza o conjunto de treinamento usando .loc
treinamento.loc[:, features] = (treinamento.loc[:, features] - media_treinamento) / std_treinamento

# Normaliza o conjunto de teste utilizando a média e o desvio padrão do conjunto de treinamento usando .loc
teste.loc[:, features] = (teste.loc[:, features] - media_treinamento) / std_treinamento

# print(treinamento.shape)
# print(teste.shape)

# separa as linhas do database que sao de cada especie (fins de analise grafica)
# setosa = iris.loc[iris["Species"] == "Iris-setosa"]
# virginica = iris.loc[iris["Species"] == "Iris-virginica"]
# versicolor = iris.loc[iris["Species"] == "Iris-versicolor"]

# print(setosa)
# print(virginica)
# print(versicolor)

# Boxplots e Scatter Plot em um único gráfico
# fig, axs = plt.subplots(2, 3, figsize=(18, 10))
# fig.suptitle("Boxplots e Scatter Plot para cada característica")

# # Boxplots
# axs[0, 0].boxplot(
#     [setosa["PetalWidthCm"], virginica["PetalWidthCm"], versicolor["PetalWidthCm"]],
#     labels=["setosa", "virginica", "versicolor"],
# )
# axs[0, 0].set_title("Petal Width")

# axs[0, 1].boxplot(
#     [setosa["PetalLengthCm"], virginica["PetalLengthCm"], versicolor["PetalLengthCm"]],
#     labels=["setosa", "virginica", "versicolor"],
# )
# axs[0, 1].set_title("Petal Length")

# axs[1, 1].boxplot(
#     [setosa["SepalWidthCm"], virginica["SepalWidthCm"], versicolor["SepalWidthCm"]],
#     labels=["setosa", "virginica", "versicolor"],
# )
# axs[1, 1].set_title("Sepal Width")

# axs[1, 0].boxplot(
#     [setosa["SepalLengthCm"], virginica["SepalLengthCm"], versicolor["SepalLengthCm"]],
#     labels=["setosa", "virginica", "versicolor"],
# )
# axs[1, 0].set_title("Sepal Length")

# # Scatter plot
# axs[0, 2].scatter(setosa["PetalWidthCm"], setosa["PetalLengthCm"], c="green", label="Setosa")
# axs[0, 2].scatter(virginica["PetalWidthCm"], virginica["PetalLengthCm"], c="red", label="Virginica")
# axs[0, 2].scatter(versicolor["PetalWidthCm"], versicolor["PetalLengthCm"], c="blue", label="Versicolor")
# axs[0, 2].set_xlabel("PetalWidthCm")
# axs[0, 2].set_ylabel("PetalLengthCm")
# axs[0, 2].legend()
# axs[0, 2].set_title("Scatter Plot of Petal Width vs Petal Length")

# # Remove the empty subplot
# fig.delaxes(axs[1, 2])

# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()

# gera os pontos de treinamento com base no dataframe e guarda as distancias relativas de todos os pontos de treinamento com cada um desses pontos
pontos_treinamento = generate_pontos(
    treinamento["Id"].to_list(),
    treinamento["PetalWidthCm"].to_list(),
    treinamento["PetalLengthCm"].to_list(),
    treinamento["Species"].to_list(),
)

# gera os pontos de teste com base no dataframe e guarda as distancias relativas de todos os pontos de TREINAMENTO com cada um desses pontos
pontos_teste = generate_pontos(
    teste["Id"].to_list(),
    teste["PetalWidthCm"].to_list(),
    teste["PetalLengthCm"].to_list(),
    teste["Species"].to_list(),
)

# função que conta os sucessos de cada k ou de um k especifico (testando)
# k_n = [i for i in range(1, len(pontos_treinamento)) if i % 3 != 0]  # dá pra mudar os valores aqui se precisar
k_n = [i for i in range(1, len(pontos_treinamento)) if i % 2 != 0]  # dá pra mudar os valores aqui se precisar

for i in pontos_treinamento:  # gera as distancias de cada ponto de treinamento com todos os outros pontos de treinamento
    distancias(pontos_treinamento, i)

sucessos_por_k = {i: 0 for i in k_n}
for i in pontos_treinamento:  # analise desses sucessos....
    indices = {
        "Iris-setosa": 0,
        "Iris-virginica": 0,
        "Iris-versicolor": 0,
    }
    for k in k_n:
        for j in range(k):
            indices[i.dists[j][0].type] += 1

        if i.type == max(indices, key=indices.get):
            sucessos_por_k[k] += 1

for k in sucessos_por_k.keys():
    print(f"Para k = {k}, {sucessos_por_k[k]} sucessos. (Taxa: {100 * sucessos_por_k[k] / 120}%)")

k = max(sucessos_por_k, key=sucessos_por_k.get)  # pega o menor k com a maior taxa de sucesso (k otimo)
print("k otimo:", k)

plt.figure(figsize=(10, 6))
plt.plot(sucessos_por_k.keys(), [100 * sucessos_por_k[k] / len(pontos_treinamento) for k in sucessos_por_k.keys()])
plt.xlabel('k')
plt.ylabel('Success Rate (%)')
plt.title('Success Rate vs k')
plt.grid(True)
plt.axvline(x=k, color='r', linestyle='--', label=f'k ótimo: {k} (taxa: {100 * sucessos_por_k[k] / len(pontos_treinamento)}%)')
plt.legend()
plt.show()
# # gera os pontos de teste com base no dataframe e guarda as distancias relativas de todos os pontos de TREINAMENTO com cada um desses pontos

for i in pontos_teste:  # gera as distancias de cada ponto de teste com todos os pontos de treinamento
    distancias(pontos_treinamento, i)

sucessos = 0
for i in pontos_teste:  # analise da taxa de sucesso do k escolhido com os pontos de teste
    indices = {
        "Iris-setosa": 0,
        "Iris-virginica": 0,
        "Iris-versicolor": 0,
    }

    for j in range(k):
        indices[i.dists[j][0].type] += 1

    if i.type == max(indices, key=indices.get):
        sucessos += 1

print("taxa de sucesso:", 100 * sucessos / 30, "%")
