import torch as t
from torch import nn
from torch.nn import functional as F
from torch import optim



class CharDetector(nn.Module):

    def __init__(self, in_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.ReLU6(),
            nn.Linear(64, 64),
            nn.ReLU6(),
            nn.Linear(64, 64),
            nn.ReLU6(),
            nn.Linear(64, 64),
            nn.Sigmoid()
        )

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam()

    def forward(self, input):
        return self.model(input)


    def trizle(self, x, y):
        self.zero_grad()
        pred = self(x)
        loss = self.loss(pred, y)
        loss.backward()
        self.optimizer.step()

        return loss.detach()


if __name__ == '__main__':

    model = CharDetector(50)






