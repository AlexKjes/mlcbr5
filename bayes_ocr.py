from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from data_loader import load_shuffled
from cnnao import ConvolutionalAutoEncoder
import torch as t
import numpy as np


cnnao = ConvolutionalAutoEncoder([1, 20, 20], 30)
pca = PCA(50)

x, y, enum, w = load_shuffled()

divide = int(x.shape[0]*0.8)
x_train = x[:divide]
#x_train = cnnao(t.from_numpy(x_train), mode='code').detach().numpy()
x_train = pca.fit_transform(x_train.reshape(-1, 400))
y_train = y[:divide]

x_test = x[divide:]
#x_test = cnnao(t.from_numpy(x_test), mode='code').detach().numpy()
x_test = pca.transform(x_test.reshape(-1, 400))
y_test = y[divide:]


classifier = KNeighborsClassifier()
classifier.fit(x_train, y_train)

print(accuracy_score(y_test, classifier.predict(x_test)))
print(classifier.predict(x_train[0].reshape(1, -1)))

def classify(img):
    val = cnnao(img).detach().numpy()
    val = classifier.predict(val)
    ret = np.zeros((1, 27))
    ret[int(val)] = 1
    return ret