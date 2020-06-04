import xgboost as xgb
import numpy
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

v = 2


def my_method(data_x, data_y, c_v):
    if v == 1:
        svm_classifier = SVC(max_iter=100000)
        grid_prm = {'gamma' : ['scale', 'auto'],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
        grid_search_cv = GridSearchCV(svm_classifier, grid_prm, scoring = "accuracy", cv = c_v)
        grid_search_cv.fit(data_x, data_y)
    else:
        svm_classifier = xgb.XGBClassifier(max_iter=100000)
        grid_prm = {'booster' : ['gbtree', 'gblinear', 'dart'],
                    'eta' : numpy.linspace(0.1, 0.5, num=5),
                    'sampling_method' : ['uniform', 'gradient_based'],
                    'tree_method'  : ['approx', 'hist', 'auto', 'exact']}
        grid_search_cv = RandomizedSearchCV(svm_classifier, grid_prm, scoring = "accuracy", cv = c_v, n_iter = 20, random_state = 0)
        grid_search_cv.fit(data_x, data_y)
    return grid_search_cv


