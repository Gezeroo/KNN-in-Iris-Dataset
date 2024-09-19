import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

class pontoIris:#classe de cada ponto de análise (válido para treinamento e teste)
    dists = []
    def __init__ (self, x, y, type, id):
        self.id = id
        self.x = x
        self.y = y
        self.type = type

def generate_pontos(ids,xs,ys,species):
    pontos = []
    for i in range(len(ids)):
        pontos.append(pontoIris(xs[i],ys[i],species[i],ids[i]))
    return pontos

def distancias(pontos_treinamento,ponto):
    dists = []
    for i in range(len(pontos_treinamento)):
        dists.append((pontos_treinamento[i],math.dist((pontos_treinamento[i].x,pontos_treinamento[i].y),(ponto.x,ponto.y))))
    dists.sort(key = lambda tup: tup[1])
    ponto.dists = dists[1:]





iris = pd.read_csv("./Iris.csv")#le o database

iris = iris.sample(frac = 1)#embaralha o database

#separa o database em campo de teste e treinamento
treinamento = iris[:round(len(iris)*0.8)]
teste = iris[round(len(iris)*0.8):]

# print(treinamento.shape)
# print(teste.shape)

#separa as linhas do database que sao de cada especie (fins de analise grafica)
setosa = iris.loc[iris['Species']=='Iris-setosa']
virginica = iris.loc[iris['Species']=='Iris-virginica']
versicolor = iris.loc[iris['Species']=='Iris-versicolor']

# print(setosa)
# print(virginica)
# print(versicolor)

#boxplots de cada especie
fig, (axs1,axs2) = plt.subplots(2,2)
fig.suptitle('boxplots para cada caracteristica')
axs1[0].boxplot([setosa['PetalWidthCm'],virginica['PetalWidthCm'],versicolor['PetalWidthCm']], tick_labels = ['setosa','virginica','versicolor'])
axs1[0].set_title('Petal Width')

axs1[1].boxplot([setosa['PetalLengthCm'],virginica['PetalLengthCm'],versicolor['PetalLengthCm']], tick_labels = ['setosa','virginica','versicolor'])
axs1[1].set_title('Petal Length')

axs2[0].boxplot([setosa['SepalWidthCm'],virginica['SepalWidthCm'],versicolor['SepalWidthCm']], tick_labels = ['setosa','virginica','versicolor'])
axs2[0].set_title('Sepal Width')

axs2[1].boxplot([setosa['SepalLengthCm'],virginica['SepalLengthCm'],versicolor['SepalLengthCm']], tick_labels = ['setosa','virginica','versicolor'])
axs2[1].set_title('Sepal Length')

plt.show()

#grafico de pontos de cada especie (escolhemos petal width e length como base para agrupamento)
plt.scatter(setosa['PetalWidthCm'], setosa['PetalLengthCm'], c = 'green',label = 'Setosa')
plt.scatter(virginica['PetalWidthCm'], virginica['PetalLengthCm'], c = 'red', label = 'Virginica')
plt.scatter(versicolor['PetalWidthCm'], versicolor['PetalLengthCm'], c = 'blue', label = 'Versicolor')
plt.xlabel('PetalWidthCm')
plt.ylabel('PetalLengthCm')
plt.legend()

plt.show()

#gera os pontos de treinamento com base no dataframe e guarda as distancias relativas de todos os pontos de treinamento com cada um desses pontos
pontos_treinamento = generate_pontos(treinamento['Id'].to_list(),treinamento['PetalWidthCm'].to_list(),treinamento['PetalLengthCm'].to_list(),treinamento['Species'].to_list())
for i in pontos_treinamento:
    distancias(pontos_treinamento,i)
    #print(i.id,' ',i.x,' ',i.y,' ',i.type)

sucesso_por_k = {}

for k in range(3,len(treinamento),2):#dicionario para guardar os sucessos para cada k, em cada ponto de treinamento
    sucesso_por_k[k] = 0

for i in pontos_treinamento:#analise desses sucessos....
    Indice_setosa = 0
    Indice_virginica = 0
    Indice_versicolor = 0

    for k in range(3,len(treinamento),2):
        
        for j in range(k):
            if i.dists[j][0].type == 'Iris-setosa':
                Indice_setosa += 1
            elif i.dists[j][0].type == 'Iris-virginica':
                Indice_virginica += 1
            elif i.dists[j][0].type == 'Iris-versicolor':
                Indice_versicolor += 1
        if max(Indice_setosa,Indice_versicolor,Indice_virginica) == Indice_setosa:
            if i.type == 'Iris-setosa':
                sucesso_por_k[k]+=1
        elif max(Indice_setosa,Indice_versicolor,Indice_virginica) == Indice_versicolor:
            if i.type == 'Iris-versicolor':
                sucesso_por_k[k]+=1
        elif max(Indice_setosa,Indice_versicolor,Indice_virginica) == Indice_virginica:
            if i.type == 'Iris-virginica':
                sucesso_por_k[k]+=1

for k in range(3,len(treinamento),2):
    print('para k =',k,',',sucesso_por_k[k],'sucessos.(',100*sucesso_por_k[k]/120,'%)')

k = max(sucesso_por_k, key=sucesso_por_k.get)#pega o menor k com a maior taxa de sucesso (k otimo)
print('k otimo:',k)

#gera os pontos de teste com base no dataframe e guarda as distancias relativas de todos os pontos de TREINAMENTO com cada um desses pontos
pontos_teste = generate_pontos(teste['Id'].to_list(),teste['PetalWidthCm'].to_list(),teste['PetalLengthCm'].to_list(),teste['Species'].to_list())
for i in pontos_teste:
    distancias(pontos_treinamento,i)

sucessos = 0

for i in pontos_teste:#analise da taxa de sucesso do k escolhido com os pontos de teste
    Indice_setosa = 0
    Indice_virginica = 0
    Indice_versicolor = 0
    for j in range(k):
        if i.dists[j][0].type == 'Iris-setosa':
            Indice_setosa += 1
        elif i.dists[j][0].type == 'Iris-virginica':
            Indice_virginica += 1
        elif i.dists[j][0].type == 'Iris-versicolor':
            Indice_versicolor += 1
    if max(Indice_setosa,Indice_versicolor,Indice_virginica) == Indice_setosa:
        if i.type == 'Iris-setosa':
            sucessos+=1
    elif max(Indice_setosa,Indice_versicolor,Indice_virginica) == Indice_versicolor:
        if i.type == 'Iris-versicolor':
            sucessos+=1
    elif max(Indice_setosa,Indice_versicolor,Indice_virginica) == Indice_virginica:
        if i.type == 'Iris-virginica':
            sucessos+=1

print('taxa de sucesso:',100*sucessos/30,'%')