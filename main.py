import methods as m
import warnings
warnings.filterwarnings('ignore')
import pandas
import numpy
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
file_train = pandas.read_csv("train2.csv", header=None)
file_test = pandas.read_csv("test2.csv", header=None)
file_output = "lab3.csv"
file_train = file_train.values
file_test = file_test.values
data_x = file_train[:, :len(file_train[0]) - 1]
data_y = file_train[:, len(file_train[0]) - 1]
simple_imputer = SimpleImputer(missing_values=data_x[27, 1], strategy='most_frequent')
my_fit = simple_imputer.fit(data_x)
data_x = my_fit.transform(data_x, )
file_test = simple_imputer.transform(file_test)
data_x, file_test = m.replacement(data_x, file_test)
data_y = numpy.array(data_y, dtype=int)
c_v = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
grid_search_cv = m.grid_cv(data_x, data_y, c_v)
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

