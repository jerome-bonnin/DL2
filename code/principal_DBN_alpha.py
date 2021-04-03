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
            L[i].train(nb_iter, learning_rate, batch_size, x)
            x = L[i].entree_sortie(x)

    def generer_image_DBN():
        return 0
