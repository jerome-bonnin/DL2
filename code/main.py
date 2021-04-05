import sys
import numpy as np
import matplotlib.pyplot as plt
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

    if args[0] == 'fig3':
        data = training_images()
        data_label, data_label_bin = training_labels()
        data_test = testing_images()
        data_test_label, data_test_label_bin= testing_labels()
        data = binariser(data)
        data_test = binariser(data_test)
        n_donnees_list = [1000, 3000, 7000, 10000, 30000, 60000]

        #Images avec n_donnees test
        n_donnees_list = [1000, 3000, 7000, 10000, 30000, 60000]
        train_err = []
        pretrain_err = []
        for n_donnees in n_donnees_list:
            print('#################################################')
            print("Nombre données train = {}".format(n_donnees))
            sig = np.random.permutation(len(data))
            data = data[sig]
            data_label = data_label[sig]
            data_label_bin = data_label_bin[sig]
            data_trunc = data[:n_donnees]
            data_label_trunc = data_label[:n_donnees]
            data_label_bin_trunc = data_label_bin[:n_donnees]
            couches = [50, 50]
            network_pretrain = DNN([data.shape[1]] + couches + [10])
            network_train = DNN([data.shape[1]] + couches + [10])
            network_pretrain.pretrain_DNN(data, 10, 0.1, 64)
            network_train = retropropagation(network_train, 20, 0.1, 64, data_trunc, data_label_bin_trunc)
            network_pretrain = retropropagation(network_pretrain, 20, 0.1, 64, data_trunc, data_label_bin_trunc)
            train_err.append(test_DNN(network_train, data_test, data_test_label_bin, data_test_label))
            pretrain_err.append(test_DNN(network_pretrain, data_test, data_test_label_bin, data_test_label))
        plt.plot(n_donnees_list, train_err, label = 'Train')
        plt.plot(n_donnees_list, pretrain_err, label = 'Pretrain')
        plt.legend()
        plt.show()

    if args[0] == 'fig1':
        data = training_images()
        data_label, data_label_bin = training_labels()
        data_test = testing_images()
        data_test_label, data_test_label_bin= testing_labels()
        data = binariser(data)
        data_test = binariser(data_test)
        n_couches_list = [1, 3, 5, 7]
        train_err = []
        pretrain_err = []
        for n_couches in n_couches_list:
            print('#################################################')
            print("Nombre couches = {}".format(n_couches))
            couches = [50 for i in range(n_couches)]
            network_pretrain = DNN([data.shape[1]] + couches + [10])
            network_train = DNN([data.shape[1]] + couches + [10])
            network_pretrain.pretrain_DNN(data, 10, 0.1, 64)
            network_train = retropropagation(network_train, 20, 0.1, 64, data, data_label_bin)
            network_pretrain = retropropagation(network_pretrain, 20, 0.1, 64, data, data_label_bin)
            train_err.append(test_DNN(network_train, data_test, data_test_label_bin, data_test_label))
            pretrain_err.append(test_DNN(network_pretrain, data_test, data_test_label_bin, data_test_label))
        plt.plot(n_couches_list, train_err, label = 'Train')
        plt.plot(n_couches_list, pretrain_err, label = 'Pretrain')
        plt.legend()
        plt.show()

    if args[0] == 'fig2':
        data = training_images()
        data_label, data_label_bin = training_labels()
        data_test = testing_images()
        data_test_label, data_test_label_bin= testing_labels()
        data = binariser(data)
        data_test = binariser(data_test)
        taille_couches_list = [50, 100, 200, 300, 500, 700]
        train_err = []
        pretrain_err = []
        for taille_couches in taille_couches_list:
            print('#################################################')
            print("Taille couches = {}".format(taille_couches))
            couches = [taille_couches, taille_couches]
            network_pretrain = DNN([data.shape[1]] + couches + [10])
            network_train = DNN([data.shape[1]] + couches + [10])
            network_pretrain.pretrain_DNN(data, 10, 0.1, 64)
            network_train = retropropagation(network_train, 20, 0.1, 64, data, data_label_bin)
            network_pretrain = retropropagation(network_pretrain, 20, 0.1, 64, data, data_label_bin)
            train_err.append(test_DNN(network_train, data_test, data_test_label_bin, data_test_label))
            pretrain_err.append(test_DNN(network_pretrain, data_test, data_test_label_bin, data_test_label))
        plt.plot(taille_couches_list, train_err, label = 'Train')
        plt.plot(taille_couches_list, pretrain_err, label = 'Pretrain')
        plt.legend()
        plt.show()































































