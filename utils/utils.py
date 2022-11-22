import torch
import copy
import random
import numpy as np
from numpy import fliplr, flipud
from skimage.transform import resize, rotate

def data_augment(x,prob=0.5):
    img_batches, img_deps, img_rows, img_cols = x.shape
    imgs = copy.deepcopy(x)
    for i in range(img_batches):
        im = imgs[i,:,:,:].transpose(0, 2)
        im = im.numpy()
        if random.random() >= prob:
            im = fliplr(im)
        if random.random() >= prob:
            im = flipud(im)
        if random.random() >= prob:
            temp = random.random()
            for j in range(8):
                if temp<= (j+1)/8.0:
                    im = rotate(im, j*45)
        im = im.copy()
        im = torch.from_numpy(im)
        im = im.transpose(2, 0).unsqueeze(0)
        imgs[i,:,:,:] = im
    return imgs

def cut_out(x, nholes, length, prob=0.5):
    if random.random() >= prob:
        return x
    img_batches, img_deps, img_rows, img_cols = x.shape
    imgs = copy.deepcopy(x)
    for i in range(img_batches):
        mask = np.ones((img_rows,img_cols))
        for n in range(nholes):
            c_x = np.random.randint(img_cols)
            c_y = np.random.randint(img_rows)

            y1 = np.clip(c_y - length // 2, 0, img_rows)
            y2 = np.clip(c_y + length // 2, 0, img_rows)
            x1 = np.clip(c_x - length // 2, 0, img_cols)
            x2 = np.clip(c_x + length // 2, 0, img_cols)
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(imgs[i,:,:,:]).type(torch.FloatTensor)
        imgs[i,:,:,:] = mask * imgs[i,:,:,:].type(torch.FloatTensor)
    return imgs