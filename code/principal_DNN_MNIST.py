import numpy as np
from principal_RBM_alpha import RBM
from principal_DBN_alpha import DNN
from utils import cross_entropy

def calcul_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)

def entree_sortie_reseau(network, x):
    list_RBM = network.RBM_list
    list_res = [x]
    for rbm in list_RBM[:-1]:
        x = rbm.entree_sortie(x)
        list_res.append(x)
    x = np.dot(x, list_RBM[-1].W) + list_RBM[-1].b
    z = calcul_softmax(x)
    list_res.append(z)
    return list_res

def retropropagation(network, num_iter, learning_rate, batch_size, data, data_label): #Attention aux dimensions
    eps = learning_rate
    n_batch = len(data) // batch_size
    sig = np.random.permutation(len(data))
    data = data[sig]
    data_label = data_label[sig]
    batch_list = [(data[k*batch_size:(k+1)*batch_size], data_label[k*batch_size:(k+1)*batch_size]) for k in range(n_batch)]
    if (len(data) % batch_size != 0):
        batch_list.append((data[n_batch*batch_size:], data_label[n_batch*batch_size:]))
    for i in range(num_iter):
        print("----------------------------------------")
        print("EPOCH = {}".format(i))
        loss = 0
        for data, label in batch_list:
            list_RBM = network.RBM_list
            #L'ensemble des x(l)
            list_x = entree_sortie_reseau(network, data)
            #Cas de fin de réseau
            x = list_x[-1] #représente x^(p)
            x_moins = list_x[-2] #Représente x^(p-1)
            c = x - label
            grad_W = np.dot(x_moins.T, c) / len(x)
            grad_b = np.mean(c, axis=0, keepdims=True)
            c_plus = c #Représente c^(p+1)
            list_RBM[-1].W -= eps*grad_W #Modification des poids du réseau
            list_RBM[-1].b -= eps*grad_b #Modification des biais du réseau
            #Cas général
            for k in range(len(list_RBM) - 2, -1, -1):
                x = list_x[k+1]
                x_moins = list_x[k]
                c = np.dot(c_plus, list_RBM[k+1].W.T) * x * (1 - x)
                c_plus = c
                grad_W = np.dot(x_moins.T, c) / len(x)
                grad_b = np.mean(c, axis=0, keepdims=True)
                (list_RBM[k]).W -= eps*grad_W #Modification des poids du réseau
                (list_RBM[k]).b -= eps*grad_b #Modification des biais du réseau
            network.RBM_list = list_RBM #Modification de l'objet réseau
            y_hat = entree_sortie_reseau(network, data)[-1]
            loss += cross_entropy(y_hat, label)
        loss /= len(data)
        print("Loss = {}".format(loss))
    return network

def test_DNN(network, data_test, data_label_bin, data_label):
    sortie = entree_sortie_reseau(network, data_test)
    y_hat = sortie[-1] #ici, y_hat est une matrice de proba pour chaque "individus"
    error = cross_entropy(y_hat, data_label_bin)
    classif = np.argmax(y_hat, axis=1)
    erreur_class = (classif - data_label !=0)
    sum_error_class = sum(erreur_class)
    return sum_error_class/len(y_hat)