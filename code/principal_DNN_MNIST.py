import numpy as np
from principal_RBM_alpha import RBM
from principal_DBN_alpha import DNN
from utils import cross_entropy

def calcul_softmax(layer, x):
    z = layer.entree_sortie(x)
    return np.exp(z) / np.sum(np.exp(z))

def entree_sortie_reseau(network, x):
    list_RBM = network.RBM_list
    list_res = []
    for rbm in list_RBM:
        x = rbm.entree_sortie(x)
        list_res.append(x)
    z = calcul_softmax(list_RBM[-1], x)
    list_res.append(z)
    return(list_res)

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
        loss = 0
        for data, label in batch_list:
            list_RBM = network.list_RBM
            #L'ensemble des x(l)
            list_x = entree_sortie_reseau(network, data)
            #Cas de fin de réseau
            x = list_x[len(list_x) -1] #représente x^(p)
            x_moins = list_x[len(list_x) -2] #Représente x^(p-1)
            c = [x - data_label]
            grad_W = np.dot(c,x_moins)
            grad_b = c
            c_plus = c #Représente c^(p+1)
            list_RBM[-1].W += eps*grad_W #Modification des poids du réseau
            list_RBM[-1].b += eps*grad_b #Modification des biais du réseau
            #Cas général
            for k in range(len(list_x) - 2, 0, -1):
                x = list_x[k]
                x_moins = list_x[k-1]
                c = np.dot(c_plus, list_RBM[k+1]) * x * (1 - x)
                c_plus = c
                grad_W = np.dot(c,x_moins) / len(x)
                grad_b = np.mean(c, axis=0, keepdims=True)
                list_RBM[k-1].W += eps*grad_W #Modification des poids du réseau
                list_RBM[k-1].b += eps*grad_b #Modification des biais du réseau
            network.list_RBM = list_RBM #Modification de l'objet réseau
            y_hat = entree_sortie_reseau(network, data)
            loss += cross_entropy(y_hat, label)
        loss /= len(data)
        print("----------------------------------------")
        print("EPOCH = {}".format(i))
        print("Error = {}".format(loss))
    return network

def test_DNN(network, data_test, data_label):
    sortie = network.entree_sortie_reseau(data_test)
    y_hat = sortie[-1] #ici, y_hat est une matrice de proba pour chaque "individus"
    error = cross_entropy(y_hat, data_label)
    return error