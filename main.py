import methods as m
import pandas
from sklearn.preprocessing import OneHotEncoder
import numpy
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
file_train = pandas.read_csv("train2.csv", header=None)
file_test = pandas.read_csv("test2.csv", header=None)
file_output = "lab3.csv"
data_x = numpy.asarray(file_train.drop(14, axis = 1))
data_y = numpy.asarray(file_train[14])
tmp = [1, 3, 5, 6, 7, 8, 9, 13]
simple_imputer = SimpleImputer(missing_values='?', strategy='most_frequent')
data_x = simple_imputer.fit_transform(data_x)
file_test = simple_imputer.fit_transform(file_test)
onehotencoder = OneHotEncoder(handle_unknown='ignore')
ctransformer = ColumnTransformer([("onehot", onehotencoder, tmp)], remainder='passthrough')
data_x = ctransformer.fit_transform(data_x)
file_test = ctransformer.transform(file_test)
data_y = numpy.array(data_y, dtype=int)
c_v = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
grid_search_cv = m.my_method(data_x, data_y, c_v)
print("The best estimator: ")
print(grid_search_cv.best_estimator_)
print("The best score: ")
print(grid_search_cv.best_score_)
print("The best system parameters: ")
print(grid_search_cv.best_params_)

y = grid_search_cv.predict(file_test)
y = numpy.array(y, dtype=int)
csv_output = pandas.DataFrame(y)
csv_output.to_csv(file_output, header=False, index=False)

