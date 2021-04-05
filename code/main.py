import sys
import numpy as np
from utils import lire_alpha_digit, training_images, testing_images, training_labels, testing_labels, binariser
from principal_RBM_alpha import RBM
from principal_DBN_alpha import DNN
from principal_DNN_MNIST import retropropagation, test_DNN

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


    if args[0] == 'dnn_MNIST':
        n_epochs = int(args[1])
        learning_rate = float(args[2])
        batch_size = int(args[3])
        couches = [int(s) for s in args[4].rsplit('-')]
        data = training_images()
        data_label, data_label_bin = training_labels()
        data_test = testing_images()
        data_test_label, data_test_label_bin= testing_labels()
        data = binariser(data)
        data_test = binariser(data_test)
        dnn = DNN([data.shape[1]] + couches + [10])
        dnn = retropropagation(dnn, n_epochs, learning_rate, batch_size, data, data_label_bin)
        print(test_DNN(dnn, data_test, data_test_label_bin, data_test_label))