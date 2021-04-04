from principal_RBM_alpha import RBM
from principal_DBN_alpha import DNN

def calcul_softmax(layer, x):
    z = layer.entree_sortie(x)
    return(np.exp(z)/sum(np.exp(z)))

def entree_sortie_reseau(network, x):
    list_RBN = network.DNN_list
    nb_couches = network.nb_couches
    list_res = []
    for i in range(nb_couches - 1):
        x = entree_sortie(list_RBN[i])
        list_res.append(x)
    z = calcul_softmax(list_RBN[nb_couches - 1], x)
    list_res.append(z)
    return(list_res)

def retropropagation(network, num_iter, learning_rate, batch_size, data, data_label):
    eps = learning_rate
    nb_elm = len(data)//batch_size
    data = np.random.shuffle(data)
    batch = [data[i: i + nb_elm] for i in range(batch_size)]
    for i in range (nb_iter):
        error = 0
        for data_batch in batch:
            c_list = np.zeros(1, network.nb_couches)
            list_RBM = network.list_RBM
            #L'ensemble des x(l)
            list_x = entree_sortie_reseau(network, data_batch)
            #Cas de fin de réseau
            c = [list_x[len(list_x) -1] - data_label]
            grad_W = np.dot(c,list_x[len(list_x) -2])
            grad_b = c
            c_list[len(list_x) -1] = c
            list_RBM[len(list_x) -1].W = W + eps*grad_W
            list_RBM[len(list_x) -1].b = b + eps*grad_b
            #Cas général
            for k in range(len(list_x) - 2, -1, -1):
                x = list_x[k]
                c = np.dot(c_list[k+1], list_RBM[k+1])*.x*.(1 -. x)
                c_list[k] = c
                grad_W = np.dot(c,list_x[k])
                grad_b = c
                list_RBM[k].W = W + eps*grad_W
                list_RBM[k].b = b + eps*grad_b
            network.list_RBM = list_RBM
            y_hat = entree_sortie_reseau(network, data_batch)
            loss = np.dot(data_label_batch, np.transpose(y_hat))
            '''
            Faux car, les labels n'ont pas été ajoutés au batch
            Faux aussi car ceci est la formule de la binary cross entropy, il faut binariser les labels.
            Il faudrait binariser les valeurs de data_label.
            Ici, y_hat est une proba, il ne changerait pas.
            '''
        print("----------------------------------------")
        print("EPOCH = " + i)
        print("Error = " + loss)

def test_DNN():
    return 0