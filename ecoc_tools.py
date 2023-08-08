
"""
Simple ECOC Classifier
edit by Tycho Zhong
"""

import numpy as np
import warnings
from Decoding.Decoder import get_decoder
from sklearn.metrics import euclidean_distances
import copy
import random

def _check_estimator(estimator):
    """Make sure that an estimator implements the necessary methods."""
    if (not hasattr(estimator, "decision_function") and
            not hasattr(estimator, "predict_proba")):
        raise ValueError("The base estimator should implement "
                         "decision_function or predict_proba!")


def check_is_fitted(estimator, attributes):
    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]


def _fit_ternary(estimator, X, y):
    """Fit a single ternary estimator. not offical editing.
        delete item from X and y when y = 0
        edit by elfen.
    """
    X, y = X[y != 0], y[y != 0]
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        warnings.warn('only one class')
    else:
        estimator = copy.deepcopy(estimator)
        estimator.fit(X, y)
    return estimator


def is_regressor(estimator):
    """Returns True if the given estimator is (probably) a regressor."""
    return getattr(estimator, "_estimator_type", None) == "regressor"


def _predict_binary(estimator, X):
    """Make predictions using a single binary estimator."""
    if is_regressor(estimator):
        return estimator.predict(X)
    try:
        score = np.ravel(estimator.decision_function(X))
    except (AttributeError, NotImplementedError):
        # probabilities of the positive class
        score = estimator.predict_proba(X)[:, 1]
    # print("score", score)
    return score


def _sigmoid_normalize(X):
    return 1 / (1 + np.exp(-X))


def _min_max_normalize(X):
    """Min max normalization
    warning: 0 value turns not 0 in most cases.
    """
    res = []
    for x in X:
        x_min, x_max = min(x), max(x)
        x_range = x_max - x_min
        res.append([float(i-x_min)/x_range for i in x])
    return np.array(res)


def _min_max_normalize_one(X):
    """Min max normalization
    warning: 0 value turns not 0 in most cases.
    """
    res = []
    x_min, x_max = min(X), max(X)
    x_range = x_max - x_min
    for i in X:
        res.append(float(i-x_min)/x_range)
    return np.array(res)


class SimpleECOCClassifier:
    def __init__(self, estimator, code_matrix, decoder='ED', soft=True):  # 解码方法使用弱欧式解码
        self.estimator = estimator  # classifier
        self.code_matrix = code_matrix  # code matrix
        self.decoder = get_decoder(decoder)  # decoder 弱欧式解码
        self.soft = soft  # if using soft distance.

    def fit(self, X, y):
        _check_estimator(self.estimator)
        if hasattr(self.estimator, "decision_function"):
            self.estimator_type = 'decision_function'
        else:
            self.estimator_type = 'predict_proba'

        self.classes_ = np.unique(y)  # [1,2,3,4]
        classes_index = dict((c, i) for i, c in enumerate(self.classes_))  # {1: 0, 2: 1, 3: 2, 4: 3}
        # Y是所有训练数据中学生最终成绩以编码矩阵的形式表现出来
        Y = np.array([self.code_matrix[classes_index[y[i]]] for i in range(X.shape[0])], dtype=np.int)
        self.estimators_ = [_fit_ternary(self.estimator, X, Y[:, i]) for i in range(Y.shape[1])]
        return self

    def predict(self, X):
        check_is_fitted(self, 'estimators_')
        Y = np.array([_predict_binary(self.estimators_[i], X) for i in range(len(self.estimators_))]).T

        # Y_min, Y_max = Y.min(), Y.max()
        # print('%s: (%f , %f)' % (self.estimator_type, Y_min, Y_max))

        if self.estimator_type == 'decision_function':
            Y = _min_max_normalize(Y)  # Use a normalization because scale of Y is [-1,1]
        Y = Y * 2 - 1  # mapping scale [0, +1] to [-1, +1]
        pred = self.decoder.decode(Y, self.code_matrix).argmin(axis=1)
        return self.classes_[pred]

    def fit_predict(self, X, y, test_X):
        self.fit(X, y)
        return self.predict(test_X)


# for 特征选择
class SimpleECOCClassifier2:
    def __init__(self, estimator, code_matrix, decoder='ED', soft=True):  # 解码方法使用弱欧式解码
        self.estimator = estimator  # classifier
        self.code_matrix = code_matrix  # code matrix
        self.fs_matrix = self.code_matrix.tolist()[0]  # 特征选择方法位
        self.fs_num_matrix = self.code_matrix.tolist()[1]  # 特征选择数量位
        self.code_matrix = np.array(self.code_matrix.tolist()[2:])  #编码矩阵
        # print("code_matrix", self.code_matrix)
        # print("fs_matrix", self.fs_matrix)
        self.decoder = get_decoder(decoder)  # decoder 弱欧式解码
        self.soft = soft  # if using soft distance.

    def fit(self, train_x_svmrfe30, train_y_svmrfe30, train_x_rf30, train_y_rf30, train_x_bsswss30, train_y_bsswss30, train_x_svmrfe55, train_y_svmrfe55, train_x_rf55, train_y_rf55, train_x_bsswss55, train_y_bsswss55, train_x_svmrfe80, train_y_svmrfe80, train_x_rf80, train_y_rf80, train_x_bsswss80, train_y_bsswss80):
        _check_estimator(self.estimator)
        if hasattr(self.estimator, "decision_function"):
            self.estimator_type = 'decision_function'
        else:
            self.estimator_type = 'predict_proba'
        self.estimators_ = []
        for i in range(self.code_matrix.shape[1]):
            if self.fs_matrix[i] == -1:
                if self.fs_num_matrix[i] == -1:
                    X = train_x_svmrfe30
                    y = train_y_svmrfe30
                elif self.fs_num_matrix[i] == 0:
                    X = train_x_svmrfe55
                    y = train_y_svmrfe55
                else:
                    X = train_x_svmrfe80
                    y = train_y_svmrfe80
            elif self.fs_matrix[i] == 0:
                if self.fs_num_matrix[i] == -1:
                    X = train_x_rf30
                    y = train_y_rf30
                elif self.fs_num_matrix[i] == 0:
                    X = train_x_rf55
                    y = train_y_rf55
                else:
                    X = train_x_rf80
                    y = train_y_rf80
            else:
                if self.fs_num_matrix[i] == -1:
                    X = train_x_bsswss30
                    y = train_y_bsswss30
                elif self.fs_num_matrix[i] == 0:
                    X = train_x_bsswss55
                    y = train_y_bsswss55
                else:
                    X = train_x_bsswss80
                    y = train_y_bsswss80
            self.classes_ = np.unique(y)  # [1,2,3,4]
            classes_index = dict((c, i) for i, c in enumerate(self.classes_))  # {1: 0, 2: 1, 3: 2, 4: 3}
            # Y是所有训练数据中学生最终成绩以编码矩阵的形式表现出来
            Y = np.array([self.code_matrix[classes_index[y[i]]] for i in range(X.shape[0])], dtype=np.int)
            self.estimators_.append(_fit_ternary(self.estimator, X, Y[:, i]))
        return self

    def predict(self, validate_x_svmrfe30, validate_x_rf30, validate_x_bsswss30, validate_x_svmrfe55, validate_x_rf55, validate_x_bsswss55, validate_x_svmrfe80, validate_x_rf80, validate_x_bsswss80):
        check_is_fitted(self, 'estimators_')
        Y_list = []
        for i in range(len(self.estimators_)):
            if self.fs_matrix[i] == -1:
                if self.fs_num_matrix[i] == -1:
                    X = validate_x_svmrfe30
                elif self.fs_num_matrix[i] == 0:
                    X = validate_x_svmrfe55
                else:
                    X = validate_x_svmrfe80
            elif self.fs_matrix[i] == 0:
                if self.fs_num_matrix[i] == -1:
                    X = validate_x_rf30
                elif self.fs_num_matrix[i] == 0:
                    X = validate_x_rf55
                else:
                    X = validate_x_rf80
            else:
                if self.fs_num_matrix[i] == -1:
                    X = validate_x_bsswss30
                elif self.fs_num_matrix[i] == 0:
                    X = validate_x_bsswss55
                else:
                    X = validate_x_bsswss80
            Y_list.append(_predict_binary(self.estimators_[i], X))
        Y = np.array(Y_list).T

        if self.estimator_type == 'decision_function':
            Y = _min_max_normalize(Y)  # Use a normalization because scale of Y is [-1,1]
        Y = Y * 2 - 1  # mapping scale [0, +1] to [-1, +1]
        pred = self.decoder.decode(Y, self.code_matrix).argmin(axis=1)
        return self.classes_[pred]
