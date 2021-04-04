from principal_RBM_alpha import RBM
from principal_DBN_alpha import DNN
from utils import cross_entropy

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

def retropropagation(network, num_iter, learning_rate, batch_size, data, data_label): #Attention aux dimensions
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
            x = list_x[len(list_x) -1] #représente x^(p)
            x_moins = list_x[len(list_x) -2] #Représente x^(p-1)
            c = [x - data_label]
            grad_W = np.dot(c,x_moins)
            grad_b = c
            c_plus= c #Représente c^(p+1)
            list_RBM[-1].W = W + eps*grad_W #Modification des poids du réseau
            list_RBM[-1].b = b + eps*grad_b #Modification des biais du réseau
            #Cas général
            for k in range(len(list_x) - 2, 0, -1): #Pas sûr de l'indexation
                x = list_x[k]
                x_moins = list_x[k-1]
                c = np.dot(c_plus, list_RBM[k+1])*.x*.(1 -. x)
                c_plus = c
                grad_W = np.dot(c,x_moins) / len(x)
                grad_b = np.mean(c, axis=0, keepdims=True)
                list_RBM[k].W = W + eps*grad_W #Modification des poids du réseau
                list_RBM[k].b = b + eps*grad_b #Modification des biais du réseau
            network.list_RBM = list_RBM #Modification de l'objet réseau
            y_hat = entree_sortie_reseau(network, data_batch)
            loss = cross_entropy(y_hat, data_batch_label)
            '''
            Faux car, les labels n'ont pas été ajoutés au batch
            '''
        print("----------------------------------------")
        print("EPOCH = " + i)
        print("Error = " + loss)

def test_DNN(network, data_test, data_label):
    sortie = network.entree_sortie_reseau(data_test)
    y_hat = sortie[-1] #ici, y_hat est une matrice de proba pour chaque "individus"
    error = cross_entropy(y_hat, data_label)
    return (error)