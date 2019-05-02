import torch as t
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np

from matplotlib import pyplot as plt

from data_loader import load_char_images

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

t.set_default_tensor_type(t.DoubleTensor)

class ReLU1(nn.Module):

    def forward(self, input):
        input[input>1] = 1
        input[input<0] = 0
        return input


class ConvolutionalAutoEncoder(nn.Module):

    def __init__(self, input_size, bottle_neck, target_weights=None):
        super().__init__()
        self.input_size = input_size
        ips = input_size
        dense_size = int(np.floor(ips[1]/4)*np.floor(ips[2]/4)*64)
        self.cnn_in = nn.Sequential(
            nn.ReplicationPad2d(3),
            nn.Conv2d(input_size[0], 64, 7, 2),
            nn.ELU(),
            nn.ReplicationPad2d(2),
            nn.Conv2d(64, 64, 5, 2),
            nn.ELU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(64, 64, 3, 1),
            nn.ELU(),
        )

        self.dense_in = nn.Sequential(
            nn.Linear(dense_size, 124),
            nn.ELU(),
            nn.Linear(124, 124),
            nn.ELU(),
            nn.Linear(124, bottle_neck),
            nn.Tanh()
        )

        self.dense_out = nn.Sequential(
            nn.Linear(bottle_neck, dense_size),
            nn.ELU()
        )

        self.cnn_out = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ELU(),
            nn.ConvTranspose2d(32, ips[0], 4, 2, 1),
            ReLU1()
        )

        self.optimizer = optim.Adam(self.parameters())
        self.target_weights = target_weights

        try:
            self.load_state_dict(t.load('./cnnau.pt'))
        except:
            pass

    def forward(self, input, mode='encode'):
        if mode == 'encode':
            return self.encode(input)
        elif mode == 'code':
            return self.code(input)
        elif mode == 'decode':
            return self.decode(input)

    def encode(self, input):
        ret = self.code(input)
        return self.decode(ret)

    def code(self, input):
        ips = self.input_size
        xs = int(np.floor(ips[1]/4))
        ys = int(np.floor(ips[2]/4))
        ret = self.cnn_in(input)
        ret = ret.view(-1, xs*ys*64)
        ret = self.dense_in(ret)

        return ret

    def decode(self, input):
        ips = self.input_size
        xs = int(np.floor(ips[1]/4))
        ys = int(np.floor(ips[2]/4))
        ret = self.dense_out(input)

        ret = ret.view(-1, 64, ys, xs)
        ret = self.cnn_out(ret)

        return ret

    def trizzle(self, x, y=None):
        if y is None:
            y = x
        self.zero_grad()
        y_ = self.forward(x)
        loss = F.smooth_l1_loss(y_, y)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def fit(self, x, epochs=20, batch_size=20, noise=0, cache=True):
        x = x.cuda()
        self.set_cuda()
        for ep in range(epochs):
            epoch_img = x[t.randperm(x.shape[0])]
            print("\rEpoch: {}".format(ep), end='')
            for i in range(0, epoch_img.shape[0], batch_size):
                x_noise = epoch_img[i:i + 20] + noise
                loss = self.trizzle(x_noise, epoch_img[i:i + 20])
        self.unset_cuda()
        if cache:
            t.save(self.state_dict(), './cnnau.pt')


    def set_cuda(self):
        self.dense_out = self.dense_out.cuda()
        self.dense_in = self.dense_in.cuda()
        self.cnn_in = self.cnn_in.cuda()
        self.cnn_out = self.cnn_out.cuda()

    def unset_cuda(self):
        self.dense_out = self.dense_out.cpu()
        self.dense_in = self.dense_in.cpu()
        self.cnn_in = self.cnn_in.cpu()
        self.cnn_out = self.cnn_out.cpu()


if __name__ == '__main__':

    cnn = ConvolutionalAutoEncoder([1, 20, 20], 26)
    #inpt = t.rand(1, 1, 20, 20)
    #cnn(inpt)

    x, y, encoding = load_char_images()
    images = t.from_numpy(x)
    #digits = digits.type(t.FloatTensor)
    try:
        cnn.load_state_dict(t.load('./model.pkl'))
    except:
        plt.ion()
        std = images.std(0)/4
        for ep in range(20):
            epoch_img = images[t.randperm(images.shape[0])]
            for i in range(0, epoch_img.shape[0], 20):
                loss = cnn.trizzle(epoch_img[i:i+20])
                print(loss)

            """
            scatter = cnn(images[:100], 'code').detach().numpy()
            for i in range(100, images.shape[0], 100):
                scatter = np.concatenate((scatter, cnn(images[i:i+100], 'code').detach().numpy()))
    
            for i in np.random.permutation(y.max())[:10]:
                cls = scatter[y==i]
                plt.scatter(cls[:, 0], cls[:, 1], i, label=encoding[i])
            plt.legend()
            plt.pause(0.001)
            """

            rnd = np.random.randint(0, images.shape[0])
            img = cnn.forward(images[rnd:rnd+1])[0].squeeze(0).detach().numpy()
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.imshow(images[rnd].squeeze(0).detach().numpy())
            plt.pause(0.0001)

        t.save(cnn.state_dict(), './model.pkl')

    scatter = cnn(images[:100], 'code').detach().numpy()
    for i in range(100, images.shape[0], 100):

        scatter = np.concatenate((scatter, cnn(images[i:i + 100], 'code').detach().numpy()))

    rshuf = t.randperm(images.shape[0])
    scatter = scatter[rshuf]
    images = images[rshuf]
    y = y[rshuf]


    split = int(scatter.shape[0]*0.8)
    train = scatter[:split]
    test = scatter[split:]

    cknn = KNeighborsClassifier(10)
    cknn.fit(train, y[:split])

    pca_data = PCA(26)
    pd = images.squeeze(1).numpy()
    print(images.shape)
    pdata = pca_data.fit_transform(pd.reshape(pd.shape[0], pd.shape[1]*pd.shape[2]))
    pd_train = pdata[:split]
    pd_test = pdata[split:]

    pknn = KNeighborsClassifier(10)
    pknn.fit(pd_train, y[:split])

    print(accuracy_score(y[split:], cknn.predict(test)))
    print(accuracy_score(y[split:], pknn.predict(pd_test)))

