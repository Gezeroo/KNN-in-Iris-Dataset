from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import math
from IPython.display import display

class pontoIris:  # classe de cada ponto de análise (válido para treinamento e teste)

    def __init__(self, id, largura, comprimento, especie):
        self.id = id
        self.largura = largura
        self.comprimento = comprimento
        self.especie = especie
        self.distancias = []


# gera a lista de pontos de treinamento e teste
def gerar_conjunto_pontos(lista_ids, lista_larguras, lista_comprimentos, lista_especies):
    pontos = []

    numero_pontos = len(lista_ids)

    for i in range(numero_pontos):

        Id = lista_ids[i]
        comprimento = lista_comprimentos[i]
        largura = lista_larguras[i]
        especie = lista_especies[i]

        pontos.append(pontoIris(Id, largura, comprimento, especie))

    return pontos

def distancia_euclidiana(ponto1, ponto2):
    diferenca_comprimento = ponto1.comprimento - ponto2.comprimento
    diferenca_largura = ponto1.largura - ponto2.largura

    return math.sqrt(diferenca_comprimento ** 2 + diferenca_largura ** 2)

# calcula as distâncias de cada ponto de teste para todos os pontos de treinamento
def distancias_ponto_a_todos_pontos_treinamento(pontos_treinamento, ponto_teste):

    distancias = []

    for ponto_treinamento in pontos_treinamento:

        distancia = distancia_euclidiana(ponto_teste, ponto_treinamento)

        distancias.append((distancia, ponto_treinamento))

    distancias_ordenadas = sorted(distancias, key=lambda x: x[0])

    ponto_teste.distancias_para_pontos_treinamento = distancias_ordenadas

# Lê o database
iris = pd.read_csv("./Iris.csv")

# Seleciona as colunas de características
caracteristicas = ["PetalWidthCm", "PetalLengthCm", "SepalWidthCm", "SepalLengthCm"]

# Embaralha o database
iris = iris.sample(frac=1, random_state=42)#.reset_index(drop=True)  # Embaralha e reseta os índices

# Separa o database em conjunto de treinamento e teste
tamanho_treinamento = round(len(iris) * 0.8)
treinamento = iris[:tamanho_treinamento].copy()
teste = iris[tamanho_treinamento:].copy()

# Calcula a média e o desvio padrão com base apenas no conjunto de treinamento
media_treinamento = treinamento[caracteristicas].mean()
desvio_padrao_treinamento = treinamento[caracteristicas].std()

treinamento[caracteristicas] = (treinamento[caracteristicas] - media_treinamento) / desvio_padrao_treinamento #normaliza os dados de treino
teste[caracteristicas] = (teste[caracteristicas] - media_treinamento) / desvio_padrao_treinamento #normaliza os dados de teste

# gera os pontos de treinamento com base no dataframe e guarda as distancias relativas de todos os pontos de treinamento com cada um desses pontos
pontos_treinamento = gerar_conjunto_pontos(
    treinamento["Id"].to_list(),
    treinamento["PetalWidthCm"].to_list(),
    treinamento["PetalLengthCm"].to_list(),
    treinamento["Species"].to_list(),
)

# gera os pontos de teste com base no dataframe e guarda as distancias relativas de todos os pontos de TREINAMENTO com cada um desses pontos
pontos_teste = gerar_conjunto_pontos(
    teste["Id"].to_list(),
    teste["PetalWidthCm"].to_list(),
    teste["PetalLengthCm"].to_list(),
    teste["Species"].to_list(),
)

valores_k = [i for i in range(1, len(pontos_treinamento)) if i % 2 != 0]  # valores de k a serem testados (impares)
precisao_por_k = {i: 0 for i in valores_k}

for k in valores_k:
    acertos = 0

    for ponto_teste in pontos_teste:
        distancias_ponto_a_todos_pontos_treinamento(pontos_treinamento, ponto_teste)
        vizinhos_mais_proximos = ponto_teste.distancias_para_pontos_treinamento[:k]
        contagem_especies = defaultdict(int)

        for _, ponto in vizinhos_mais_proximos:
            contagem_especies[ponto.especie] += 1

        especie_mais_frequente = max(contagem_especies, key=contagem_especies.get)

        if especie_mais_frequente == ponto_teste.especie:
            acertos += 1

    precisao_por_k[k] = acertos / len(pontos_teste)


melhor_k = max(precisao_por_k, key=precisao_por_k.get)

print(f"Melhor valor de k: {melhor_k}")
print(f"Melhor precisão: {precisao_por_k[melhor_k]}")

plt.plot(precisao_por_k.keys(), precisao_por_k.values())
plt.xlabel("Valor de k")
plt.ylabel("Precisão")
plt.title("Precisão do KNN para diferentes valores de k")
plt.show()

