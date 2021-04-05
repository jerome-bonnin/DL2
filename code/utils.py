import numpy as np
from scipy.io import loadmat
import gzip

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
    return - np.sum(y * np.log(y_hat + 1e-9)) / len(y)

def training_images():
    with gzip.open('../data/train-images-idx3-ubyte.gz', 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count * column_count))
        return images

def testing_images():
    with gzip.open('../data/t10k-images-idx3-ubyte.gz', 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count * column_count))
        return images

def training_labels():
    with gzip.open('../data/train-labels-idx1-ubyte.gz', 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        #Binaristion de labels
        y_bin = np.zeros((len(labels),10))
        for i in range(len(labels)):
            y_bin[i, labels[i]] = 1
        return labels, y_bin

def testing_labels():
    with gzip.open('../data/t10k-labels-idx1-ubyte.gz', 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        #Binaristion de labels
        y_bin = np.zeros((len(labels),10))
        for i in range(len(labels)):
            y_bin[i, labels[i]] = 1
        return labels, y_bin