import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Perceptron import Perceptron
from PlotImage import plot_decision_regions
from Adaline import Adaline

df = pd.read_csv('iris.txt', header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa' , -1, 1)
X = df.iloc[0:100, [0, 2]].values


ppn = Perceptron(eta=0.5, n_iter= 5)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1) , ppn.errors_, marker ='o')
plt.title('Perceptron')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

plot_decision_regions(X, y, classifier=ppn)
plt.title('Perceptron')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()


## copy data
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
ada = Adaline(n_iter=15 , eta=0.01, random_state=1)
ada.fit(X_std, y)


plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.title('Adaline')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()