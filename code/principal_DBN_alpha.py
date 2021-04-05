import numpy as np
import matplotlib.pyplot as plt
from principal_RBM_alpha import RBM

class DNN:
    def __init__(self, couches):
        self.RBM_list = []
        self.nb_couches = len(couches) - 1
        for i in range(self.nb_couches):
            self.RBM_list.append(RBM(couches[i], couches[i+1]))

    def pretrain_DNN(self, x, nb_iter, learning_rate, batch_size):
        for i, rbm in enumerate(self.RBM_list):
            print("#########################")
            print("RBM numero {}".format(i))
            rbm.train_RBM(x, nb_iter, learning_rate, batch_size)
            x = rbm.entree_sortie(x)

    def entree_sortie_DBN(self, v):
        h = v
        for k in range(self.nb_couches):
            h = self.RBM_list[k].entree_sortie(h)
        return(h)

    def sortie_entree_DBN(self, h):
        v = h
        for k in range(self.nb_couches):
            v = self.RBM_list[-k-1].sortie_entree(v)
        return(v)

    def generer_image_DBN(self, n_iter, n_images, height=20, width=16):
        n_rows = np.ceil(np.sqrt(n_images))
        n_cols = np.ceil(np.sqrt(n_images))
        images = []
        axes = []
        fig = plt.figure()
        for k in range(n_images):
            p_v0 = np.ones((1,height*width)) / 2
            x = np.random.binomial(n=1, p=1-p_v0) # On initialise l'image aléatoirement
            for _ in range (n_iter):
                p_h_v0 = self.entree_sortie_DBN(x)
                h = np.random.binomial(n=1, p=p_h_v0)
                p_v_h = self.sortie_entree_DBN(h)
                x = np.random.binomial(n=1, p=p_v_h)
            reconstructed_image = x.reshape((height, width))
            images.append(reconstructed_image)
            axes.append(fig.add_subplot(n_rows, n_cols, k+1))
            axes[-1].set_title("Subplot {}".format(k))
            axes[-1].imshow(reconstructed_image, cmap='gray')
        fig.tight_layout()
        plt.show()
        return(images)
