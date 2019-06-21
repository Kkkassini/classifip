import math
import random
import numpy as np
import classifip.models.svmMultiLabel


def correctness_measure(y_true, y_predicts):
    # print("Inference [true,predict]:", y_true, y_predicts)
    y_true = y_true.split(">")
    if y_predicts is None: return 0.0;
    k = len(y_true)
    sum_dist = 0
    for idx, label in enumerate(y_true):
        min_dist = np.array([], dtype=int)
        for y_predict in y_predicts:
            min_dist = np.append(min_dist, abs(idx - y_predict[label]))
        sum_dist += np.min(min_dist)
    return 1 - sum_dist / (0.5 * k * k)


# correctness_measure("L1>L2>L3", "L1>L3>L2")
def completeness_measure(y_true, y_predicts):
    y_true = y_true.split(">")
    k = len(y_true)
    R = 0
    if y_predicts is not None:
        learn_ranks = dict()
        for y_predict in y_predicts:
            for idx, label in enumerate(y_true):
                if label not in learn_ranks:
                    learn_ranks[label] = np.array([], dtype=int)
                learn_ranks[label] = np.append(learn_ranks[label], y_predict[label])
        # learning ranking models: learn_ranks
        for key, _ in learn_ranks.items():
            R += len(np.unique(learn_ranks[key]))
    return (k * k - R) / (k * k - k)


def cross_validation(data, test_percent, nb_var, nb_labels, v_list: list):
    model = classifip.models.svmMultiLabel.SVMML(nb_var, nb_labels)
    # 1 - Shuffle the data
    random.shuffle(data)

    nb_elem_fold = math.ceil(len(data) * test_percent)
    nb_fold = int(len(data) // nb_elem_fold)

    # sum_completeness_measure_point_for_all_folds = 0

    avg_correctness_measure_point_for_all_folds_for_each_v = []
    for v in v_list:
        sum_correctness_measure_point_for_all_folds = 0
        start_index = 0
        v_index = v_list.index(v)
        for fold in range(0, nb_fold):

            # 2 - Set trainset and testset
            testset = data[start_index:start_index + nb_elem_fold]
            trainset = [i for i in data if i not in testset]
            start_index = start_index + nb_elem_fold

            # 3 - In the fold execute learn() to get the label weights
            model.learn(trainset_data=trainset)

            # 4 - Predict the preference for testset using label weights from step 3
            res_list = model.predict(testset, res_format_string=False)
            res = res_list[v_index]

            # 5 - Calculate the 2 accuracies for each fold
            sum_correctness_measure_point = 0
            # sum_completeness_measure_point = 0

            for instance in testset:
                correctness_measure_point = correctness_measure(instance[nb_var], res[testset.index(instance)])
                # completeness_measure_point = completeness_measure(instance[nb_var], res[testset.index(instance)])

                sum_correctness_measure_point = sum_correctness_measure_point + correctness_measure_point
                # sum_completeness_measure_point = sum_completeness_measure_point + completeness_measure_point

            avg_correctness_measure_point_for_1_fold = sum_correctness_measure_point / len(testset)
            # avg_completeness_measure_point_for_1_fold = sum_completeness_measure_point / len(testset)

            sum_correctness_measure_point_for_all_folds = sum_correctness_measure_point_for_all_folds \
                                                          + avg_correctness_measure_point_for_1_fold
            # sum_completeness_measure_point_for_all_folds = sum_completeness_measure_point_for_all_folds \
            #                                                + avg_completeness_measure_point_for_1_fold
        # 6 - Calculate 2 accuracies for all folds

        avg_correctness_measure_point_for_all_folds = sum_correctness_measure_point_for_all_folds / nb_fold
        # avg_completeness_measure_point_for_all_folds = sum_completeness_measure_point_for_all_folds / nb_fold
        avg_correctness_measure_point_for_all_folds_for_each_v.append(avg_correctness_measure_point_for_all_folds)
    correctness_point = max(avg_correctness_measure_point_for_all_folds_for_each_v)
    v_optimum = v_list[avg_correctness_measure_point_for_all_folds_for_each_v.index(correctness_point)]
    return v_optimum, correctness_point, avg_correctness_measure_point_for_all_folds_for_each_v
