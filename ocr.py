import numpy as np
from imageio import imwrite, imread
from scipy.misc import imshow
import torch as t

from nn_classifier import classify as nn_classifier
from bayes_ocr import classify as knn_classifier
from data_loader import char_enum

from scipy.misc import imshow
from data_loader import char_enum

img1 = t.from_numpy(imread('dataset/detection-images/detection-1.jpg')).double()
img2 = t.from_numpy(imread('dataset/detection-images/detection-2.jpg')).double()





def ocr_img(img, classifier):
    ret = t.zeros(img.shape+(27,)).double()
    for y in range(0, img.shape[0]-20, 3):
        for x in range(0, img.shape[1]-20, 3):
            #val = img[y:y+20, x:x+20].reshape(1, 1, 20, 20)
            #val = img[75:95, 25:45].reshape(1, 1, 20, 20)
            #val = img[96:116, 65:85].reshape(1, 1, 20, 20)
            val = img[68:88, 102:122].reshape(1, 1, 20, 20)

            show = val.numpy().reshape(20, 20)
            #print(show.shape)
            imshow(show)
            val = classifier(val).detach()

            #for i in range(ret.shape[-1]):
            #    ret[y:y+20, x:x+20, i] += val[0, i]
            #if val.max() > 0.1:
            ret[y:y + 20, x:x + 20] += val
            break
        break


    return ret

def ocr_classify(img):
    ocr = ocr_img(img, nn_classifier).cpu()
    print(ocr.reshape(40000, -1).max(0)[0].shape)
    print(ocr.max(2)[0].shape)
    ocr /= (ocr.max(2)[0]).unsqueeze(2)
    ocr *= 255

    for i in range(ocr.shape[-1]):

        imwrite('img_ocr/img1_{}.bmp'.format(char_enum[i]), ocr[:, :, i].numpy().astype(np.uint8))


def ntocr_img(img, classifier):
    ret = np.zeros(img.shape+(27,))
    for y in range(0, img.shape[0]-20, 2):
        for x in range(0, img.shape[1]-20, 2):
            val = img[y:y+20, x:x+20].reshape(1, 1, 20, 20)
            #val = img[75:95, 25:45].reshape(1, 1, 20, 20)
            val = classifier(val)

            #for i in range(ret.shape[-1]):
            #    ret[y:y+20, x:x+20, i] += val[0, i]
            #if val.max() > 0.1:
            ret[y:y + 20, x:x + 20, val.argmax()] += val.max()


    return ret

def ntocr_classify(img):
    ocr = ocr_img(img, knn_classifier)
    print(ocr.reshape(40000, -1).max(0)[0].shape)
    print(ocr.max(2)[0].shape)
    ocr /= (ocr.max(2)[0])
    ocr *= 255

    for i in range(ocr.shape[-1]):

        imwrite('img_ocr/img1_{}.bmp'.format(char_enum[i]), ocr[:, :, i].astype(np.uint8))


ocr_classify(img1)
#ntocr_classify(img1)
