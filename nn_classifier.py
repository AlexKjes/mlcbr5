import torch as t
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np

from data_loader import load_shuffled
from cnnao import ConvolutionalAutoEncoder
from sklearn.decomposition import PCA


class NNClassifier(nn.Module):

    def __init__(self, in_size, out_size, class_weights=None):
        super().__init__()

        self.class_weights = class_weights
        self.model = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.ReLU6(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU6(),
            nn.Dropout(0.1),
            nn.Linear(32, out_size),
            nn.Sigmoid(),
            #nn.Softmax(1)
        )

        self.optimizer = optim.Adam(self.parameters(), weight_decay=0.0001)
        self.loss = nn.CrossEntropyLoss(self.class_weights)



    def forward(self, input):
        return self.model(input)

    def trainzor(self, x, y):
        self.zero_grad()
        y_ = self.model(x)
        loss = self.loss(y_, y)
        loss.backward()
        self.optimizer.step()

        #return loss.detach()

    def fit(self, x, y, n_epochs=20, batch_size=20):
        x = x.cuda()
        y = y.cuda()
        self.model = self.model.cuda()
        for e in range(n_epochs):
            print('\r{}'.format(e), end='')

            for i in range(0, x.shape[0], batch_size):
                self.trainzor(x[i:i+batch_size], y[i:i+batch_size])



classifier = NNClassifier(50, 27)
cnnae = ConvolutionalAutoEncoder([1, 20, 20], 50)
pca = PCA(30)

x, y, em, w = load_shuffled()
w = t.from_numpy(w).cuda()
x = t.from_numpy(x).double()
y = t.from_numpy(y).long()
split = int(x.shape[0]* .8)
x_train = x[:split]
x_test = x[split:]
y_train = y[:split]
y_test = y[split:]
pca.fit(x_train.reshape(-1, 400))
try:
    classifier.load_state_dict(t.load('classifier.pt'))
except:
    pass

def classify(img):
    val = cnnae(img, mode='code')
    #val = t.from_numpy(pca.transform(img.reshape(1, -1)))
    val = classifier(val)
    return val

if __name__ == '__main__':

    x, y, em, w = load_shuffled()
    w = t.from_numpy(w).cuda()
    x = t.from_numpy(x).double()
    y = t.from_numpy(y).long()
    split = int(x.shape[0]* 1)
    x_train = x[:split]
    x_test = x[split:]
    y_train = y[:split]
    y_test = y[split:]

    feature_size = 50

    cnnae = ConvolutionalAutoEncoder([1, 20, 20], feature_size, target_weights=w)
    cnnae.fit(x_train, 50)

    pca = PCA(feature_size)
    pca.fit(x_train.reshape(x_train.shape[0], -1))

    ae_train = cnnae.forward(x_train, 'code').detach().cuda()
    #ae_test = cnnae.forward(x_test, 'code').detach().cuda()

    pca_train = x_train.reshape(x_train.shape[0], -1)
    pca_train = t.from_numpy(pca.transform(pca_train))
    #pca_test = x_test.reshape(x_test.shape[0], -1)
    #pca_test = t.from_numpy(pca.transform(pca_test))

    classifier = NNClassifier(feature_size, 27, w)

    try:
        classifier.load_state_dict(t.load('classifier.pt'))
    except:
        pass
    #classifier.fit(x_train.reshape(x_train.shape[0], -1), y_train)

    #t.save(classifier.state_dict(), 'classifier.pt')

    classifier.model = classifier.model.cuda()
    test_pred = classifier(ae_train).argmax(1).cpu()
    test_err = t.mean((test_pred == y_train).double().cpu())
    best_err = test_err
    while True:
        classifier.fit(ae_train, y_train)

        #test_pred = classifier(ae_test).argmax(1).cpu()
        #test_err = t.mean((test_pred == y_test).double().cpu())

        train_pred = classifier(ae_train).argmax(1).cpu()
        train_err = t.mean((train_pred == y_train).double().cpu())
        #print('\rTrain err: {}, Test err: {}'.format(train_err, test_err))
        print('\rTrain err: {}'.format(train_err))

        if train_err > best_err:
            best_err = train_err
            t.save(classifier.state_dict(), 'classifier.pt')

"""
    print(y_test.shape)
    pred = classifier(x_test.reshape(x_test.shape[0], -1).cuda()).argmax(1).cpu()
    print(pred.shape)
    err = t.mean((pred == y_test).double().cpu())
    print(err)
"""