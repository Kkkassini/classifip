import classifip
import numpy as np
import pandas as pd
from scipy import sparse, random
from cvxopt import solvers, matrix
import math

from classifip.evaluation.crossValidationSVM import cross_validation

model = classifip.models.ncclr.NCCLR()
dataArff= classifip.dataset.arff.ArffFile()
# dataArff.load("C:/Users/Alphantares/Downloads/iris_dense.xarff")
# dataArff.load("C:/Users/Alphantares/Desktop/GI04/TX/datasets_rang/datasets_rang/vehicle_dense.xarff")
dataArff.load("C:/Users/Alphantares/Desktop/GI04/TX/datasets_rang/datasets_rang/wine_dense.xarff")


def create_q_couples_list(labels_list):
    """
    build list Q which contains the different arcs of labels
    e.g. L1>L2>L3 ====> [[L1,L2],[L2,L3],[L1,L3]]
    :param labels_list: list
    :return: list of list
    """
    res = []
    nb_labels = len(labels_list)
    for step in range(1, nb_labels):
        for start in range(0, nb_labels):
            if start + step < nb_labels and start < nb_labels - 1:
                res.append([labels_list[start], labels_list[start + step]])

    return res

def stockage_Q(data, nb_var):
    """
    build list Q for all the instances
    :param data: dataArff.data
    :param nb_label: number of labels
    :return: list Q
    """
    Q = []
    length_data = len(data)
    for i in range(length_data):
        label = data[i][nb_var]
        labels = [i for i in label.split('>')]
        q_list_for_1_instance = create_q_couples_list(labels)
        Q.append(q_list_for_1_instance)
    return Q

def calculate_H(nb_label, q, data, nb_var):
    """
    :param data: dataArff.data
    :param nb_label: number of labels
    :param q: list Q
    :return: Matrix H
    """
    row_a = []
    col_a = []
    data_a = []
    row_b = []
    col_b = []
    data_b = []
    row_c = []
    col_c = []
    data_c = []
    row_d = []
    col_d = []
    data_d = []

    length_data = len(data)

    for r in range(1, int(nb_label*(nb_label-1)*0.5)+1):
        for l in range(1, int(nb_label*(nb_label-1)*0.5)+1):
            for j in range(0,length_data):
                for i in range(0,length_data):
                    list_pq = q[i][r-1]
                    list_ab = q[j][l-1]

                    if list_pq[0]==list_ab[0]:
                        row_a.append(length_data*(r-1)+i)
                        col_a.append(length_data*(l-1)+j)
                        x_i = np.mat(data[i][:nb_var])
                        x_j = np.mat(data[j][:nb_var])
                        data_a.append((x_i*x_j.T).item())

                    elif list_pq[0]==list_ab[1]:
                        row_b.append(length_data * (r-1) + i)
                        col_b.append(length_data * (l-1) + j)
                        x_i = np.mat(data[i][:nb_var])
                        x_j = np.mat(data[j][:nb_var])
                        data_b.append((x_i*x_j.T).item())

                    elif list_pq[1]==list_ab[0]:
                        row_c.append(length_data * (r-1) + i)
                        col_c.append(length_data * (l-1) + j)
                        x_i = np.mat(data[i][:nb_var])
                        x_j = np.mat(data[j][:nb_var])
                        data_c.append((x_i*x_j.T).item())

                    elif list_pq[1]==list_ab[1]:
                        row_d.append(length_data * (r-1) + i)
                        col_d.append(length_data * (l-1) + j)
                        x_i = np.mat(data[i][:nb_var])
                        x_j = np.mat(data[j][:nb_var])
                        data_d.append((x_i*x_j.T).item())

    taille = int(0.5*nb_label*(nb_label-1)*length_data)
    mat_a = sparse.coo_matrix((data_a, (row_a, col_a)), shape=(taille,taille))
    mat_b = sparse.coo_matrix((data_b, (row_b, col_b)), shape=(taille,taille))
    mat_c = sparse.coo_matrix((data_c, (row_c, col_c)), shape=(taille,taille))
    mat_d = sparse.coo_matrix((data_d, (row_d, col_d)), shape=(taille,taille))

    mat_h = mat_a - mat_b - mat_c + mat_d

    return mat_h


def __min_convex_qp(A, q, lower, upper, d):
    ell_lower = matrix(lower, (d, 1))
    ell_upper = matrix(upper, (d, 1))
    P = matrix(A)
    # P = A
    q = matrix(q, (d, 1))
    I = matrix(0.0, (d, d))
    I[::d + 1] = 1
    G = matrix([I, -I])
    h = matrix([ell_upper, -ell_lower])
    solvers.options['refinement'] = 2
    return solvers.qp(P=P, q=q, G=G, h=h, kktsolver='ldl', options={'kktreg': 1e-9})


def get_alpha(nb_label, data, nb_var, v):
    """

    :return: list of alpha, size k(k-1)/2
    """
    length_data = len(data)
    #1. Create the list Q which contains the arcs of labels
    q = stockage_Q(nb_label, data, nb_var=nb_var)

    #2. Calculate matrix H
    h = calculate_H(nb_label, q, data, nb_var)
    h_numpy = h.todense()
    h_numpy = h_numpy.astype(np.double)

    #3.Set the contraints for the dual problem
    e_i = int(0.5*nb_label*(nb_label-1))
    max_limit = float(v/e_i)
    taille = int(0.5*nb_label*(nb_label-1)*length_data)
    res = __min_convex_qp(h_numpy, np.repeat(-1.0, taille) , np.repeat(0.0, taille), np.repeat(max_limit, taille), taille)
    solution = np.array([v for v in res['x']])

    return solution


def get_label_preference_by_weights(weights_list: list):
    """
    Get label preferences by weights
    :param weights_list: list
    :return: String
    """
    _pref = np.argsort(weights_list)[::-1]
    pref = ['L'+str(x+1) for x in _pref]

    return pref


def visualize_matrix(h: sparse.coo_matrix):
    """

    :param h: Matrice creuse sous forme de coo_matrix
    :return: pd.Dataframe
    """
    normal_matrix = h.todense()
    return pd.DataFrame(normal_matrix)


def calculate_w (q, nb_labels, nb_var, data, v):
    """
    :param data: dataArff.data
    :param nb_var: number of variables
    :param nb_labels: number of elements in labels (here 3)
    :param : v=0.5 for this moment (implicit)
    :param q: list Q
    :return: list of vector : W
    """
    length_data = len(data)
    W = []
    alpha_list = get_alpha(nb_label=nb_labels, data=data, nb_var=nb_var, v=v)
    # Calculate W_i, i = L1, L2, ...
    for i in range(1, nb_labels+1):
        # Calculate the coefficient of X1, X2, ... (alpha_j)
        wl = 0
        for j in range (1, length_data+1):

            part_sum = 0
            part_reduce = 0
            # Check each couple in alpha_i, if there's a couple that begins with L_i
            for couple_index in range(1, len(q[0])+1):
                if 'L'+str(i) == q[j-1][couple_index-1][0]:
                    # Search for the value corresponded in get_alpha()
                    alpha = alpha_list[(couple_index - 1)*length_data+j-1]
                    part_sum = part_sum + alpha
                if 'L'+str(i) == q[j-1][couple_index-1][1]:
                    alpha = alpha_list[(couple_index - 1)*length_data+j-1]
                    part_reduce = part_reduce + alpha
            product = np.dot(part_sum - part_reduce, data[j-1][:nb_var])
            wl = wl + product
        W.append(wl)
    return W


class SVMML(object):
    """
    SVMML implements the Multi label ranking method using the SVM for
    Label ranking problem with LDL decomposition.
    """
    def __init__(self, nb_var, nb_labels):
        self.nb_var = nb_var
        self.nb_labels = nb_labels
        self.W = list()
        self.res_for_each_v = list()
        self.v_list = [0.1, 0.4, 0.7, 0.9, 8, 24, 32, 128]

    def learn(self, trainset_data):
        """
        For each hyper-parameter v in v_list, calculate the label weight of each label.
        :param v_list: list of possible v values
        :param trainset_data: dataset for model training
        :param nb_labels: number of elements in labels (here 3)
        :param nb_var: number of variables
        :return: list of list: for each v, a list of vector W, weights for each label
        """
        # 1. Create list Q: in list Q, for each instance, we stock the arcs possibles of the labels.
        q = stockage_Q(nb_label=self.nb_labels, data=trainset_data, nb_var=self.nb_var)
        length_data = len(trainset_data)
        W = []
        for i in range(len(self.v_list)):
            W.append([])

        # 2. For each v, we train the model and get the label weights corresponded.
        for v in self.v_list:

            # Get alpha values for the arcs in the Q list, by resolving the dual problem:
            # argmin 0.5*t(alpha)*H*alpha - t(1)*alpha with contraints.
            alpha_list = get_alpha(nb_label=self.nb_labels, data=trainset_data, nb_var=self.nb_var, v=v)
            v_index = self.v_list.index(v)
            for i in range(1, self.nb_labels+1):

                # Calculate the coefficient of X1, X2, ... (alpha_j)
                wl = 0
                for j in range (1, length_data+1):

                    part_sum = 0
                    part_reduce = 0

                    # Check each couple in alpha_i, if there's a couple that begins with L_i
                    for couple_index in range(1, len(q[0])+1):
                        if 'L'+str(i) == q[j-1][couple_index-1][0]:
                            # Search for the value corresponded in get_alpha()
                            alpha = alpha_list[(couple_index - 1)*length_data+j-1]
                            part_sum = part_sum + alpha
                        if 'L'+str(i) == q[j-1][couple_index-1][1]:
                            alpha = alpha_list[(couple_index - 1)*length_data+j-1]
                            part_reduce = part_reduce + alpha
                    product = np.dot(part_sum - part_reduce, trainset_data[j-1][:self.nb_var])
                    wl = wl + product
                W[v_index].append(wl)
        self.W = W
        return W

    def predict(self, data_to_predict, res_format_string=True):
        """
        For each value of v, predicts the result of preference, and stock the result into a list.
        :param data_to_predict: Dataset to be predicted
        :param res_format_string: format of the prediction result
        :return: Preference prediction for every instance (list of list)
        """
        pred_list = []
        for w_v in self.W:
            pred = []
            wl_list = []
            for instance in data_to_predict:
                for w_label in range(0, len(w_v)):
                    index_instance = data_to_predict.index(instance)
                    wl_list.append(np.dot(0.5 * w_v[w_label], data_to_predict[index_instance - 1][:self.nb_var]))
                if res_format_string:
                    pred.append(get_label_preference_by_weights(wl_list))
                else:
                    list_temp = list(np.argsort(wl_list)[::-1])
                    rank = []
                    for label in range(0, len(list_temp)):
                        point = list_temp.index(label)
                        rank.append(point)
                    labels = ['L'+str(i+1) for i in range(0, len(rank))]
                    dict1 = {}
                    for j in range(len(labels)):
                        dict1[labels[j]] = rank[j]
                        pred.append([dict1])
                wl_list = []
            pred_list.append(pred)
        return pred_list

    def get_predict_result(self, data, test_percent, data_to_predict, res_format_string=True):
        """
        Predict result for the data_to_predict, with the best value of hyper-parameter v which is
        obtained by a cross validation.
        :param data: whole dataset which is gonna take a cross validation to pick the best v
        :param test_percent: ratio of train/test during the cross validation
        :param data_to_predict: data to predict
        :param res_format_string: format of the result
        :return: predicted result, e.g: [['L1', 'L3', 'L2'], ['L1', 'L2', 'L3']]

        """
        # 1. for each value of v, calculate the result corresponded
        res_for_each_v = self.predict(data_to_predict, res_format_string=res_format_string)

        # 2. use cross validation to calculate the accuracy using each possible v, then choose the best v,
        # get its accuracy corresponded, as well as the accuracies for all the possible v
        v_optimum_value, correctness_point, avg_correctness_measure_point_for_all_folds_for_each_v = \
            cross_validation(data, test_percent, self.nb_var, self.nb_labels, self.v_list)

        # 3. from the list of prediction for each v, pick the one which corresponded to the best v
        return res_for_each_v[self.v_list.index(v_optimum_value)], correctness_point, \
               avg_correctness_measure_point_for_all_folds_for_each_v







