import numpy as np
from scipy.io import loadmat

def lire_alpha_digit(caracteres):
    idx = list(caracteres)
    for i, a in enumerate(caracteres):
        try:
            idx[i] = int(a)
        except ValueError:
            idx[i] = ord(a) - ord('a') + 10
    bad = loadmat('../data/binaryalphadigs.mat')
    data = list()
    for j in idx:
        sample = [np.expand_dims(img.flatten(), axis=0) for img in bad['dat'][j,:]]
        data.extend(np.concatenate(sample, axis=0))
    data = np.array(data)
    return data

def init_RBM():
    return 0

def entree_sortie_RBM():
    return 0

def sortie_entree_RBM():
    return 0

def train_RBM():
    return 0

def generer_image_RBM():
    return 0