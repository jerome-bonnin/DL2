import numpy as np
from scipy.io import loadmat

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lire_alpha_digit(caracteres):
    # transforme la chaine de caracteres en liste d'indices
    idx = list(caracteres)
    for i, a in enumerate(caracteres):
        try:
            idx[i] = int(a)
        except ValueError:
            idx[i] = ord(a) - ord('a') + 10
    # charge les echantillons voulus et les concatene
    bad = loadmat('../data/binaryalphadigs.mat')
    data = list()
    for j in idx:
        sample = [np.expand_dims(img.flatten(), axis=0) for img in bad['dat'][j,:]]
        data.extend(np.concatenate(sample, axis=0))
    # matrice des donnees (n_samples,n_pixels)
    data = np.array(data)
    return data

def cross_entropy(y_hat, y):
    '''
    Attention, y est vecteur de taille n.
    y_hat est une matrice de taille (n_données x nombre_classe).
    '''
    #Binaristion de y_hat
    y_bin = np.zeros(shape = y.shape())
    for i in range(len(y_hat)):
        y_bin[i, y_hat[i]] = 1
    #Calcul de la cross entropy
    N = len(y) #Taille de la population
    return(-np.sum(y_bin, y_hat)/N)
