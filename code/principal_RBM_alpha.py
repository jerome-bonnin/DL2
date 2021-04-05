import numpy as np
import matplotlib.pyplot as plt
from utils import sigmoid

class RBM:
    def __init__(self, p, q):
        self.W = np.random.normal(loc=0.0, scale=0.01, size=(p,q))
        self.a = np.zeros((1,p))
        self.b = np.zeros((1,q))

    def entree_sortie(self, v):
        # Renvoie la valeur de la couche cachée à partir de la couche visible
        return sigmoid(np.dot(v, self.W) + self.b)

    def sortie_entree(self, h):
        # Renvoie la valeur de la couche visible à partir de la couche cachée
        return sigmoid(np.dot(h, self.W.T) + self.a)

    def train_RBM(self, data, n_epochs, learning_rate, batch_size):
        eps = learning_rate
        # Definition des batches
        n_batch = len(data) // batch_size
        np.random.shuffle(data)
        batch_list = [data[k*batch_size:(k+1)*batch_size] for k in range(n_batch)]
        if (len(data) % batch_size != 0):
            batch_list.append(data[n_batch*batch_size:])
        # Algorithme CD-1
        for i in range(n_epochs):
            loss = 0
            for x in batch_list:
                v0 = x
                p_h_v0 = self.entree_sortie(v0)
                h0 = np.random.binomial(n=1, p=p_h_v0)
                p_v_h0 = self.sortie_entree(h0)
                v1 = np.random.binomial(n=1, p=p_v_h0)
                x_prime = v1
                p_h_v1 = self.entree_sortie(v1)
                grad_W = (np.dot(v0.T, p_h_v0) - np.dot(v1.T, p_h_v1)) / len(x)
                self.W += eps * grad_W
                grad_a = np.mean(v0 - v1, axis=0, keepdims=True)
                self.a += eps * grad_a
                grad_b = np.mean(p_h_v0 - p_h_v1, axis=0, keepdims=True)
                self.b += eps * grad_b
                loss += np.sum((x_prime - x) ** 2)
            loss /= len(data)
            print("----------------------------------------")
            print("EPOCH = {}".format(i))
            print("loss = {}".format(loss))

    def generer_image_RBM(self, n_iter, n_images, height=20, width=16):
        n_rows = np.ceil(np.sqrt(n_images))
        n_cols = np.ceil(np.sqrt(n_images))
        images = []
        axes = []
        fig = plt.figure()
        for k in range(n_images):
            p_v0 = np.ones((1,height*width)) / 2
            x = np.random.binomial(n=1, p=p_v0) # On initialise l'image aléatoirement
            for _ in range (n_iter):
                p_h_v0 = self.entree_sortie(x)
                h = np.random.binomial(n=1, p=p_h_v0)
                p_v_h = self.sortie_entree(h)
                x = np.random.binomial(n=1, p=p_v_h)
            reconstructed_image = x.reshape((height, width))
            images.append(reconstructed_image)
            axes.append(fig.add_subplot(n_rows, n_cols, k+1))
            axes[-1].set_title("Subplot {}".format(k))
            axes[-1].imshow(reconstructed_image, cmap='gray')
        fig.tight_layout()
        plt.show()
        return(images)

