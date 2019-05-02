import numpy as np
from nn_classifier import classify
from data_loader import load_shuffled

from sklearn.metrics import accuracy_score

from nn_classifier import ConvolutionalAutoEncoder, NNClassifier

import torch as t

classifier = NNClassifier(50, 27)
cnnae = ConvolutionalAutoEncoder([1, 20, 20], 50)

try:
    classifier.load_state_dict(t.load('classifier.pt'))
except:
    pass


x, y, enum, w = load_shuffled()

xt = t.from_numpy(x[y==enum.index('t')])
yt = t.from_numpy(y[y==enum.index('t')])


y_ = cnnae(xt, mode='code')
y_ = classifier(y_).detach().argmax(1).numpy().astype(np.int8)


print(accuracy_score(yt, y_))
