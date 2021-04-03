import numpy as np
import random

def sigmoid(x) :
    return(1/(1 + np.exp(-x)))

class RBM:
    def __init__(p, q):
        self.W = np.random.normal(0, 0.01, size = (q, p))
        self.a = np.zeros(shape = (p, 1))
        self.b = np.zeros(shape = (q, 1))

    def sigmoid(self, x):
        return(1/(1 + np.exp(-x)))

    def entree_sortie(self, x): #Renvoie la valeur de la couche cachée à partir de la couche visible
        return(self.sigmoid(np.dot(self.W, x) + self.b))

    def sortie_entree(self, h): #Renvoie la valeur de la couche visible à partir de la couche cachée
        W_t = np.transpose(self.W)
        return(self.sigmoid(np.dot(W_t, h) + self.a))

    def train_RBM(self, nb_iter, learning_rate, batch_size, data):
        eps = learning_rate
        nb_elm = len(x)//batch_size
        data = np.random.shuffle(data)
        batch = [data[i: i + nb_elm] for i in range(batch_size)]
        for i in range (nb_iter):
            error = 0
            for x in batch:
                v_0 = x
                p_h_v0 = self.entree_sortie(v_0)
                h_0 = np.random.binomial(n = 1, p = p_h_v0)
                p_v1_h0 = self.sortie_entree(h_0)
                v_1 = np.random.binomial(n = 1, p = p_v1_h0)
                x_prime = v_1
                p_h_v1 = self.entree_sortie(v_1)
                grad_W = np.dot(p_h_v_0, np.transpose(v_0)) - np.dot(p_h_v1, np.transpose(v_1))
                self.W = self.W + eps*grad_W
                grad_a = v_0 - v_1
                self.a = self.a + eps*grad_a
                grad_b = p_h_v0 - p_h_v1
                self.b = self.b + eps*grad
                error +=np.mean((x_prime - x)^2)
            print("----------------------------------------")
            print("EPOCH = " + i)
            print("Error = " + error/batch_size)

    def generer_image_RBM(self, nb_iter, nb_images, nb_pixels):
        images = []
        for k in range(nb_images):
            prob = [1/2 for i in range(nb_pixels)]
            x = np.random.binomial(n = 1, p = prob) #On initialise l'image aléatoirement
            for i in range (nb_iter):
                p_h_v0 = self.entree_sortie(v_0)
                h = np.random.binomial(n = 1, p = p_h_v0)
                p_v1_h = self.sortie_entree(h)
                x = np.random.binomial(n = 1, p = p_v1_h)
