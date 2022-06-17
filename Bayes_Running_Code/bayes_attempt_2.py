# -*- coding: utf-8 -*-
# Bethany Jackson, Scott Stewart, Ivan Echevarria, Nadia Aly, Brook Lautenslager
# TUFF Classifier
import os
import time

import lightgbm as lgb
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import time

import numpy as np
import sklearn.metrics
from hyperopt.pyll import scope
from imblearn.under_sampling import OneSidedSelection, InstanceHardnessThreshold, RandomUnderSampler
from optuna import trial
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, matthews_corrcoef, f1_score, balanced_accuracy_score, make_scorer, \
    log_loss
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile
from imblearn.metrics import classification_report_imbalanced
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from skopt import BayesSearchCV, Space
import re
from skopt.callbacks import EarlyStopper
from skopt.space import Real, Integer, Categorical


def create_data(train_folder):
    print("Loading data")
    train = load_files(train_folder, description=None, categories=None, load_content=True, shuffle=True, encoding=None,
                       decode_error='strict', random_state=0)
    return train.data, train.target

def score_cl(clf, scoring, X, y):
    return cross_val_score(clf, X, y, cv=5, scoring=scoring, n_jobs=2)

def create_splits(train_folder):
    print("Creating data splits")
    # load the training set
    train = load_files(train_folder, description=None, categories=None, load_content=True, shuffle=True, encoding=None,
                       decode_error='strict', random_state=0)

    # train,test
    X_train, X_test, y_train, y_test = train_test_split(train.data, train.target, random_state=11)
    return X_train, X_test, y_train, y_test

def create_lgbm_cl():
    classifier = LGBMClassifier(boosting_type='gbdt', num_leaves=42, max_depth=14)


    #       ('vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')),

    text_clf = Pipeline([
        ('vect',HashingVectorizer(dtype=np.int64, ngram_range=(1, 2), stop_words='english', alternate_sign=False, norm=None)),
        ('tfidf', TfidfTransformer(sublinear_tf=True, use_idf=True)),
        ('over', RandomOverSampler(sampling_strategy=.7)),
        ('under', RandomUnderSampler(sampling_strategy=.7)),
        ('clf', classifier)
    ])

    return text_clf


dir_path = os.path.abspath('')
full_train_folder = 'china_clean_80_swap/china_clean/'

X_data, y_data = create_data(full_train_folder)
X_train, X_test, y_train, y_test = create_splits(full_train_folder)




class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        regex_num_punctuation = '(\d+)|([^\w\s])'
        regex_little_words = r'(\b\w{1,2}\b)'
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)
                if not re.search(regex_num_punctuation, t) and not re.search(regex_little_words, t)]

class CountVectorizerCustom(CountVectorizer):
    def fit(self, X, y=None):
        if isinstance(self.ngram_range, str):
            self.ngram_range = eval(self.ngram_range)
        return super().fit(X, y)

    def fit_transform(self, X, y=None):
        if isinstance(self.ngram_range, str):
            self.ngram_range = eval(self.ngram_range)
        return super().fit_transform(X, y)



def create_old_cl():
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='perceptron', penalty='l1', alpha=1e-4, max_iter=35, tol=float("-inf"), random_state=42))
    ])

    return text_clf




def cl_report(clf, X_test, y_test):
    target_names = ['No', 'Yes']

    all_preds = clf.predict(X_test)
    print(classification_report(y_test, all_preds, target_names=target_names))

def cl_report_unbalanced(clf, X_test, y_test):

    target_names = ['No', 'Yes']

    all_preds = clf.predict(X_test)
    print(classification_report_imbalanced(y_test, all_preds, target_names=target_names))

def optimize_pl(clf, X, y):
    print("Optimizing pipeline")

    params = {
        'tfidf__use_idf': (True, False),
        'tfidf__sublinear_tf': (True, False)
    }
    grid = GridSearchCV(clf, params, cv=3, scoring='f1', n_jobs=2)

    grid.fit(X, y)
    print(grid.best_params_)
    print("F1 Score:" + str(grid.best_score_))







print("\n")
print("------------------------------------------------")






num_eval = 40


class IterStopper(EarlyStopper):
    def __init__(self, iters=20):

        self.best_index = 0
        self.iters= iters
        self.best_score = 0.0

        #set the window size to 10% of the iterations requested.
        self.current_window = int(self.iters * 0.20)

        super().__init__()

    def _criterion(self, result):

        y0 = result.func_vals
        iter_count = len(y0)
        current_best = abs(min(y0))

        # set the window size to 10% of the remaining iterations
        window = int((self.iters - iter_count) * 0.20)

        print("The current iteration is ", iter_count)

        #check to see if we have a new score
        delta = current_best - self.best_score
        if (delta > 0):
            print("New Best Score Found :", current_best, "at iteration: ", iter_count)
            self.best_index = np.where(y0 == current_best)
            self.best_score = current_best
            self.current_window = iter_count + window
            print("Will now search until iteration: ", self.current_window)
            return False
        #check to check if we have no more iterations left in our window and stop if so
        if (iter_count > self.current_window):
            print("Stopping Early, no improvement detected")
            return True

clf = create_lgbm_cl()
start = time.time()
early_stopper = EarlyStopper
iter_stopper = IterStopper(1000)


search_space = {'clf__learning_rate' : Real(.001, 1),
                'clf__max_depth': Integer(5,25),
                'clf__num_leaves': Integer(5,100),
                'clf__boosting_type': Categorical(['gbdt', 'dart', 'goss']),
                'clf__class_weight': Categorical([None, 'balanced']),
                'clf__n_estimators': Integer(100, 1000),
                'tfidf__sublinear_tf': Categorical([True, False]),
                'tfidf__use_idf': Categorical([True, False]),
                'over__sampling_strategy' : Categorical(['minority', 'not minority', 'all']),
                'under__sampling_strategy' : Categorical(['majority', 'not majority', 'all'])
                }


bayessearch = BayesSearchCV(clf,
                            search_spaces = search_space
                            , n_iter=500, scoring="f1",
                            refit=False, n_jobs=-1,
                            n_points=5,
                            cv=TimeSeriesSplit(n_splits=3))


bayessearch.fit(X_data, y_data, callback=[iter_stopper])
end = time.time()
delta = end - start
print("Took %.2f seconds to process" % delta)
print('Best score of Bayes Search over ' + str(20) + ' iterations:', bayessearch.best_score_, bayessearch.best_params_)