import numpy as np
import os
from scipy.misc import imread


char_enum = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']

def load_char_images():
    path = './dataset/chars74k-lite/'

    int_to_char = []
    x = []
    y = []

    for i, dir in enumerate(os.listdir(path)):

        int_to_char.append(dir)
        for file in os.listdir(path+dir):
            img = imread(path+dir+'/'+file)
            x.append(img)
            y.append(char_enum.index(dir))


    x = np.array(x)
    x = x/x.max()
    x = np.expand_dims(x, 1)
    y = np.array(y)
    return x, y, int_to_char


def load_shuffled():
    path = 'dataset/shuffled_data.csv'
    data = np.genfromtxt(path, delimiter=',')
    x = data[:, 0:-1].reshape(-1, 1, 20, 20)
    y = data[:, -1]
    print(np.unique(y, return_counts=True)[1])
    count = np.unique(y, return_counts=True)[1]
    ds_weightd = np.min(count)/count

    return x, y, char_enum, ds_weightd


def make_shuffle():
    x, y, enum = load_char_images()
    x = x.reshape(-1, 400)
    data = np.column_stack((x, y))
    data = np.row_stack((data, 1-data))
    blank = 1-np.random.rand(np.min(np.unique(y, return_counts=True)[1]), 401)*0.1
    blank[:, -1] = 26
    data = np.row_stack((data, blank))
    np.random.shuffle(data)
    np.savetxt('dataset/shuffled_data.csv', data, delimiter=',')

    print(data.shape)


def make_chardetector_set():

    def impose(img1, img2):
        pass


    x, y, enum,w = load_shuffled()
    count = np.unique(y, return_counts=True)[1]
    data = []
    for i in range(x.shape[0]):
        rand_y = np.random.multinomial(26, np.ones(26)/26)
        letter = x[i]
        displace = np.random.randn(2)


        sample = np.ones(20, 20)




if __name__ == '__main__':

    make_chardetector_set()
    #make_shuffle()
    #load_shuffled()


