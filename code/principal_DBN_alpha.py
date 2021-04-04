from principal_RBM_alpha import RBM

def class DNN:
    def __init__(nb_couches, p, q):
        self.DNN_list = []
        self.nb_couches = nb_couches
        DNN_list.append(RBM(p, q))
        for i in range(nb_couches):
            DNN_list.append(RBM(q, q))
            '''
            La premiere couche a la même forme que le RBM classique.
            Les autres couches seront des couches carrées par souci de compatibilité et de simplicité.
            p (resp. q) représente la taille des données d'entrée (resp. la taille de la sortie)
            '''

    def pretrain_DNN(self, nb_iter, learning_rate, batch_size, x):
        for i in range (self.nb_couches):
            print("#########################")
            print("RBM numéro " + i)
            self.DNN_list.train(nb_iter, learning_rate, batch_size, x)
            x = self.DNN_list.entree_sortie(x)

    def entree_sortie_DBN(self, v):
        h = v
        for k in range(self.nb_layers):
            h = self.DNN_list[k].entree_sortie(h)
        return(h)

    def sortie_entree_DBN(self, h):
        v = h
        for k in range(self.nb_layers):
            v = self.DNN_list[k].sortie_entree(v)
        return(v)

    def generer_image_DBN(self, n_iter, nb_images, nb_pixels, height = 20, width = 16):
        rows = int(nb_images) +1
        cols = int(nb_images) +1
        images = []
        axes=[]
        fig=plt.figure()
        for k in range(nb_images):
            v0 = [1/2 for i in range(nb_pixels)]
            x = np.random.binomial(n = 1, p = prob) #On initialise l'image aléatoirement
            for i in range (nb_iter):
                #Génération de l'image à partir de l'echantilloneur de Gibbs
                p_h_v0 = self.entree_sortie_DBN(v_0)
                h = np.random.binomial(n = 1, p = p_h_v0)
                p_v1_h = self.sortie_entree_DBN(h)
                x = np.random.binomial(n = 1, p = p_v1_h)
            reconstruct_image = np.reshape(x, shape = (height, width))
            images.append(images)
            #Affichage des images
            axes.append( fig.add_subplot(rows, cols, a+1) )
            subplot_title=("Subplot"+str(a))
            axes[-1].set_title(subplot_title)
            plt.imshow(reconstruct_image)
        fig.tight_layout()
        plt.show()
        return(images)
