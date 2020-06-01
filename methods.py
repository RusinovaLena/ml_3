import xgboost as xgb
import numpy
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

svm = False


def grid_cv(data_x, data_y, c_v):
    if svm:
        svm_classifier = SVC(max_iter=10000)
        grid_prm = {'gamma' : ['scale', 'auto'],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
        grid_search_cv = GridSearchCV(svm_classifier, grid_prm, scoring = "accuracy", cv = c_v)
        grid_search_cv.fit(data_x, data_y)

    else:
        svm_classifier = xgb.XGBClassifier(max_iter=10000)
        grid_prm = {'booster' : ['gbtree', 'gblinear', 'dart'],
                    'eta' : numpy.linspace(0.1, 0.5, num=5),
                    'sampling_method' : ['uniform', 'gradient_based'],
                    'tree_method'  : ['approx', 'hist', 'auto', 'exact']}
        grid_search_cv = RandomizedSearchCV(svm_classifier, grid_prm, scoring = "accuracy", cv = c_v, n_iter = 20, random_state = 0)
        grid_search_cv.fit(data_x, data_y)
    return grid_search_cv


def replacement(array_one, array_two):
    for i in range(0, len(array_one[0])):
        if type(array_one[0, i]) == str:
            array = []
            for j in range(0, len(array_one)):
                n = array_one[j, i]
                if n not in array:
                    array.append(n)
                array_one[j, i] = array.index(n)
            for j in range(0, len(array_two)):
                n = array_two[j, i]
                array_two[j, i] = array.index(n)
    array_one = numpy.array(array_one, dtype=int)
    array_two = numpy.array(array_two, dtype=int)
    return array_one, array_two