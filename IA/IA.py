import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("./Iris.csv")#le o database

iris.drop(columns = ['Id'], inplace = True)#tira o campo de ID

iris = iris.sample(frac = 1)

treinamento = iris[:round(len(iris)*0.8)]
teste = iris[round(len(iris)*0.8):]

print(treinamento.shape)
print(teste.shape)

setosa = iris.loc[iris['Species']=='Iris-setosa']
virginica = iris.loc[iris['Species']=='Iris-virginica']
versicolor = iris.loc[iris['Species']=='Iris-versicolor']

print(setosa)
print(virginica)
print(versicolor)

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

plt.scatter(setosa['PetalWidthCm'], setosa['PetalLengthCm'], c = 'green')
plt.scatter(virginica['PetalWidthCm'], virginica['PetalLengthCm'], c = 'red')
plt.scatter(versicolor['PetalWidthCm'], versicolor['PetalLengthCm'], c = 'blue')

plt.show()