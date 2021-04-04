import sys

from utils import lire_alpha_digit
from principal_RBM_alpha import RBM
from principal_DBN_alpha import DNN

if __name__ == '__main__':
    args = sys.argv[1:]

    if args[0] == 'rbm_gen':
        caracteres = args[1]
        n_epochs = int(args[2])
        learning_rate = float(args[3])
        batch_size = int(args[4])
        data = lire_alpha_digit(caracteres)
        rbm = RBM(data.shape[1], len(caracteres))
        rbm.train_RBM(data, n_epochs, learning_rate, batch_size)
        rbm.generer_image_RBM(n_iter=10, n_images=16)

    if args[0] == 'dbn_gen':
        caracteres = args[1]
        n_epochs = int(args[2])
        learning_rate = float(args[3])
        batch_size = int(args[4])
        couches = [int(s) for s in args[5].rsplit('-')]
        data = lire_alpha_digit(caracteres)
        dbn = DNN([data.shape[1]] + couches + [len(caracteres)])
        dbn.pretrain_DNN(data, n_epochs, learning_rate, batch_size)
        dbn.generer_image_DBN(n_iter=10, n_images=16)