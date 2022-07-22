# -*- coding: utf-8 -*-
# Bethany Jackson, Scott Stewart, Ivan Echevarria, Nadia Aly, Brook Lautenslager
# TUFF Classifier
from datetime import datetime
import glob
import os

import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, matthews_corrcoef, f1_score, balanced_accuracy_score, make_scorer
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import joblib
import pandas as pd


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


def create_data(train_folder):
    print("Loading data")
    train = load_files(train_folder, description=None, categories=None, load_content=True, shuffle=True, encoding=None,
                       decode_error='strict', random_state=0)
    return train.data, train.target


def create_splits(train_folder):
    print("Creating data splits")
    # load the training set
    train = load_files(train_folder, description=None, categories=None, load_content=True, shuffle=True, encoding=None,
                       decode_error='strict', random_state=0)

    # train,test
    X_train, X_test, y_train, y_test = train_test_split(train.data, train.target)
    return X_train, X_test, y_train, y_test


def create_old_cl():
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf',
         SGDClassifier(loss='perceptron', penalty='l1', alpha=1e-4, max_iter=35, tol=float("-inf"), random_state=42))
    ])

    return text_clf


def create_lgbm_cl():
    classifier = LGBMClassifier(boosting_type='gbdt', num_leaves=20)

    text_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1), stop_words='english', lowercase=False)),
        ('tfidf', TfidfTransformer(sublinear_tf=True, use_idf=True)),
        ('clf', classifier)
    ])

    return text_clf


def score_cl(clf, scoring, X, y):
    return cross_val_score(clf, X, y, cv=5, scoring=scoring, n_jobs=4)


def cl_report(clf, X_test, y_test):
    target_names = ['No', 'Yes']
    all_preds = clf.predict(X_test)
    print(classification_report(y_test, all_preds, target_names=target_names))


def optimize_pl(clf, X, y):
    print("Optimizing pipeline")

    params = {
        'tfidf__use_idf': (True, False),
        'tfidf__sublinear_tf': (True, False)
    }
    grid = GridSearchCV(clf, params, cv=3, scoring='f1', n_jobs=3)

    grid.fit(X, y)
    print(grid.best_params_)
    print("F1 Score:" + str(grid.best_score_))




results = pd.DataFrame(columns=['Title', 'True Class', 'Model Class', 'Model Probability'])
full_train_folder = 'aug_collection/aug_addition'

clf_lgbm = create_lgbm_cl()

all_files = glob.glob('aug_collection/aug_addition/**/*.txt', recursive=True)
print(len(all_files))
try:
    for i in (range(20, len(all_files)+1,20)):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        print("Iteration of ", i , "to", i-20, ".", i/len(all_files), current_time)

        train = load_files(full_train_folder, description=None, categories=None, load_content=True, shuffle=True,
                               encoding=None,
                               decode_error='strict', random_state=0)

        # Loop initialization
        clf_lgbm = create_lgbm_cl()

        # Retaining the first object's properties then removing it.

        cur_X = train.data[i-20:i]
        cur_y = train.target[i-20:i]
        cur_filename = train.filenames[i-20:i]


        for k in range(i-20, i):
            train.data.pop(i)
            train.target = np.delete(train.target, i)

        clf_lgbm.fit(train.data, train.target)

        # Obtaining results and putting them in the dataframe.

        # result_array = []
        # value_array = []
        counter = 0
        for sample in cur_X:
            sample = [str(sample)]

            proba_value = clf_lgbm.predict_proba(sample)
            proba_result = clf_lgbm.predict(sample)


            print(cur_y[counter], proba_result)
            # result_array.append(proba_result)
            # value_array.append(proba_value)
            # Putting information in DF.
            results.loc[len(results.index)] = [cur_filename[counter], cur_y[counter], proba_result[0], proba_value[0][proba_result]]

            counter = counter + 1

except:
    results.to_csv('failure_catch.csv')
finally:
    results.to_csv('test3.csv')