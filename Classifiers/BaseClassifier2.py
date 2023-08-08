""" Base Classifiers """
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


class GBC:
    def __init__(self, train_x, train_y):
        self.classifiers_type = ["SVM", "Bayes", "Logi", "KNN", "DTree"]
        parameters_list = [
            # {},
            # {},
            # {},
            # {},
            # {}
            {"C": [0.1, 1, 10], "gamma": [1, 0.1, 0.01]},  # SVM
            {},  # Bayes
            {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},  # LR
            {'n_neighbors': range(1, 10, 1), 'weights': ['uniform', 'distance'],
             'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},  # KNN
            {'max_depth': range(1, 10, 1), 'min_samples_leaf': range(1, 10, 2)}  # DTree
        ]
        self.estimators_list = [SVC(), GaussianNB(), LogisticRegression(), KNeighborsClassifier(), tree.DecisionTreeClassifier()]
        self.estimators_best_list = []
        for classifier_type in self.classifiers_type:
            self.classifier_index = self.classifiers_type.index(classifier_type)
            clf = GridSearchCV(self.estimators_list[self.classifier_index], parameters_list[self.classifier_index], cv=5,
                               scoring='f1_macro')  # 进行网格搜索得到最优参数组合
            clf.fit(train_x, train_y)
            self.estimators_best_list.append(clf.best_estimator_)

    def get_base_clf(self, base, adaboost=False):
        """ Get classifiers from scikit-learn.

        Parameters:
            base: str
                indicates classifier, alternative str list below.
                'KNN' - K Nearest Neighbors (sklearn.neighbors.KNeighborsClassifier)
                'DTree' - Decision Tree (sklearn.tree.DecisionTreeClassifier)
                'SVM' - Support Vector Machine (sklearn.svm.SVC)
                'Bayes' - Naive Bayes (sklearn.naive_bayes.GaussianNB)
                'Logi' - Logistic Regression (sklearn.linear_model.LogisticRegression)
                'NN' - Neural Network (sklearn.neural_network.MLPClassifier)
            adaboost: bool, default False.
                Whether to use adaboost to promote the classifier.

        Return:
            model: object, A classifier object.
        """
        model = None
        if base is 'KNN':
            model = self.estimators_best_list[self.classifiers_type.index('KNN')]
            adaboost = False
        elif base is 'DTree':
            model = self.estimators_best_list[self.classifiers_type.index('DTree')]
        elif base is 'SVM':
            model = self.estimators_best_list[self.classifiers_type.index('SVM')]
            adaboost = False
        elif base is 'Bayes':
            model = self.estimators_best_list[self.classifiers_type.index('Bayes')]
            adaboost = False
        elif base is 'Logi':
            model = self.estimators_best_list[self.classifiers_type.index('Logi')]
        elif base is 'NN':
            from sklearn.neural_network import MLPClassifier
            model = MLPClassifier(max_iter=1000000)
            adaboost = False
        else:
            raise ValueError('Classify: Unknown value for base: %s' % base)

        # if use an adabost to strengthen model.
        if adaboost is True:
            from sklearn.ensemble import AdaBoostClassifier
            model = AdaBoostClassifier(model, algorithm="SAMME")

        return model

# coding:utf-8
