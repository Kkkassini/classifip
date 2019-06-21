import classifip.models.svmMultiLabel

# Example script of executing the svm multi label model.
# By default, we set the possible value of hyper-parameter v=[0.1, 0.4, 0.7, 0.9, 8, 24, 32, 128]. It could be
# assigned to other values by modifying the __init__ of the class.

# We start by creating an instance of the base classifier we want to use
print("Example of SVM multi-label ranking - Data set IRIS \n")
nb_var = 4
nb_labels = 3

model = classifip.models.svmMultiLabel.SVMML(nb_var, nb_labels)
dataArff= classifip.dataset.arff.ArffFile()
dataArff.load("C:/Users/Alphantares/Downloads/iris_dense.xarff")
data = dataArff.data

data_test = [[1, 2, 'L1>L2>L3'], [2, 1, 'L1>L3>L2']]

# Learning
model.learn(data)
print("Process learning finished")

# Prediction
res, correctness_point, avg_correctness_measure_point_for_all_folds_for_each_v = \
    model.get_predict_result(data=data, test_percent=0.1, data_to_predict=data, res_format_string=True)

print("Prediction is \n")
print(res)
print("\n")
print("Accuracy for the best v is \n")
print(correctness_point)
print("\n")
print("Accuracy for each v is \n")
print(avg_correctness_measure_point_for_all_folds_for_each_v)
