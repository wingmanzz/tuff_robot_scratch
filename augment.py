import pandas as pd
import random
import glob
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
#import sacremoses
from nlpaug.util import Action
import torch
import warnings
warnings.filterwarnings("ignore")

def run_class(dataframe):
    # -*- coding: utf-8 -*-
    # Bethany Jackson, Scott Stewart, Ivan Echevarria, Nadia Aly, Brook Lautenslager
    # TUFF Classifier
    import os

    import random

    import sklearn.metrics
    from imblearn.metrics import classification_report_imbalanced
    from sklearn.feature_selection import SelectFpr, chi2, SelectFromModel
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_files
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, matthews_corrcoef, f1_score, balanced_accuracy_score, make_scorer
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.pipeline import Pipeline
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.feature_selection import SelectPercentile
    from sklearn.ensemble import ExtraTreesClassifier
    import numpy as np
    from nltk import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import re
    from datetime import datetime
    from sklearn.svm import LinearSVC
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
        train = load_files(train_folder, description=None, categories=None, load_content=True, shuffle=True,
                           encoding=None,
                           decode_error='strict', random_state=0)
        return train.data, train.target

    def create_splits(train_folder):
        print("Creating data splits")
        # load the training set
        train = load_files(train_folder, description=None, categories=None, load_content=True, shuffle=True,
                           encoding=None,
                           decode_error='strict', random_state=0)

        # train,test
        X_train, X_test, y_train, y_test = train_test_split(train.data, train.target, test_size=.25, random_state=27)
        return X_train, X_test, y_train, y_test

    def create_old_cl():
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf',
             SGDClassifier(loss='perceptron', penalty='l1', alpha=1e-4, max_iter=35, tol=float("-inf"),
                           random_state=42))
        ])

        return text_clf

    def create_lgbm_cl():
        classifier = LGBMClassifier(boosting_type='gbdt', num_leaves=50)

       # print("running.")
        text_clf = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 2), stop_words='english', lowercase=False)),
            ('tfidf', TfidfTransformer(sublinear_tf=True, use_idf=True)),
            ('clf', classifier)
        ])

        return text_clf

    def score_cl(clf, scoring, X, y):
        return cross_val_score(clf, X, y, cv=5, scoring=scoring, n_jobs=2, error_score='raise')

    def cl_report(clf, X_test, y_test):
        target_names = ['No', 'Yes']

        all_preds = clf.predict(X_test)
        print(classification_report(y_test, all_preds, target_names=target_names))
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

    dir_path = os.path.abspath('')
    # full_train_folder = 'inter_swap'

    X_data = dataframe["text"].values
    y_data = dataframe["class"].values

    random.Random(4).shuffle(X_data)
    random.Random(4).shuffle(y_data)

    # X_data, y_data = create_data(full_train_folder)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=.25, random_state=27)

   # print("\n")
   # print("------------------------------------------------")
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    clf_lgbm = create_lgbm_cl()

    score_ret = score_cl(clf=clf_lgbm, scoring=make_scorer(f1_score), X=X_data, y=y_data).mean()
    return score_ret
    # print("Balanced Accuracy with 5 fold cv: ", score_cl(clf=clf_lgbm, scoring=make_scorer(balanced_accuracy_score), X=X_data, y=y_data).mean())
    # print("MCC with 5 fold cv: ", score_cl(clf=clf_lgbm, scoring=make_scorer(matthews_corrcoef), X=X_data, y=y_data).mean())



def get_acc(dataframe_train):
    # -*- coding: utf-8 -*-
    # Bethany Jackson, Scott Stewart, Ivan Echevarria, Nadia Aly, Brook Lautenslager
    # TUFF Classifier
    import os

    import random

    import sklearn.metrics
    from imblearn.metrics import classification_report_imbalanced
    from sklearn.feature_selection import SelectFpr, chi2, SelectFromModel
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_files
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, matthews_corrcoef, f1_score, balanced_accuracy_score, make_scorer
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.pipeline import Pipeline
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.feature_selection import SelectPercentile
    from sklearn.ensemble import ExtraTreesClassifier
    import numpy as np
    from nltk import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import re
    from datetime import datetime
    from sklearn.svm import LinearSVC
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
        train = load_files(train_folder, description=None, categories=None, load_content=True, shuffle=True,
                           encoding=None,
                           decode_error='strict', random_state=0)
        return train.data, train.target

    def create_splits(train_folder):
        print("Creating data splits")
        # load the training set
        train = load_files(train_folder, description=None, categories=None, load_content=True, shuffle=True,
                           encoding=None,
                           decode_error='strict', random_state=0)

        # train,test
        X_train, X_test, y_train, y_test = train_test_split(train.data, train.target, test_size=.25, random_state=27)
        return X_train, X_test, y_train, y_test

    def create_old_cl():
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf',
             SGDClassifier(loss='perceptron', penalty='l1', alpha=1e-4, max_iter=35, tol=float("-inf"),
                           random_state=42))
        ])

        return text_clf

    def create_lgbm_cl():
        classifier = LGBMClassifier(boosting_type='gbdt', num_leaves=50)

       # print("running.")
        text_clf = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 2), stop_words='english', lowercase=False)),
            ('tfidf', TfidfTransformer(sublinear_tf=True, use_idf=True)),
            ('clf', classifier)
        ])

        return text_clf

    def score_cl(clf, scoring, X, y):
        return cross_val_score(clf, X, y, cv=5, scoring=scoring, n_jobs=2, error_score='raise')

    def cl_report(clf, X_test, y_test):
        target_names = ['No', 'Yes']

        all_preds = clf.predict(X_test)
        print(classification_report(y_test, all_preds, target_names=target_names))
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

    dir_path = os.path.abspath('')
    # full_train_folder = 'inter_swap'

    X_train = dataframe_train["text"].values
    y_train = dataframe_train["class"].values

    random.Random(4).shuffle(X_train)
    random.Random(4).shuffle(y_train)

    clf_lgbm = create_lgbm_cl()
    clf_lgbm.fit(X_train, y_train)

    dataframe_test = pd.read_csv("test_df.csv")

    print("Getting Accuracy")

    X_test = dataframe_test["text"].values
    y_test = dataframe_test["class"].values

    random.Random(10).shuffle(X_test)
    random.Random(10).shuffle(y_test)

    score_ret = clf_lgbm.score(X_test, y_test)


    return score_ret

    # print("Balanced Accuracy with 5 fold cv: ", score_cl(clf=clf_lgbm, scoring=make_scorer(balanced_accuracy_score), X=X_data, y=y_data).mean())
    # print("MCC with 5 fold cv: ", score_cl(clf=clf_lgbm, scoring=make_scorer(matthews_corrcoef), X=X_data, y=y_data).mean())


# numbers_to_add = [500,1000,1500,2000,2500]
numbers_to_add = [5944]
# percentage_changed = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.10]
percentage_changed = [0.04,.05,.06,.08,.09]
print("process starting...")
stop_words = ["Source", "Publisher", "Byline", "Copyright", "Publication Date", "China"]
for percentage in percentage_changed:
    aug_text = []
    aug_train = []
    aug = naw.SynonymAug(aug_src='wordnet', model_path=None, name='Synonym_Aug', lang='eng', aug_p=percentage, stopwords=stop_words,
                         tokenizer=None, reverse_tokenizer=None, force_reload=False,
                         verbose=0)

    df = pd.read_csv('yes_col.csv')
    df_train = pd.read_csv("yes_train.csv")

    for item in df['text'].values:
        item = item.split(".")
        sentence_aug = aug.augment(item)
        full = ''.join(sentence_aug)
        aug_text.append(full)
    for item in df_train['text'].values:
        item = item.split(".")
        sentence_aug = aug.augment(item)
        full = ''.join(sentence_aug)
        aug_train.append(full)


    no = pd.read_csv("no.csv")
    no_train = pd.read_csv("no_train.csv")

    random.Random(15).shuffle(aug_text)
    random.Random(10).shuffle(aug_train)

    for number in numbers_to_add:


        ## For full F1 Score Stuff
        small_text = aug_text[0:number + 1]
        all_yes = pd.DataFrame(columns=['text', 'class'])
        place_holder = pd.DataFrame(small_text, columns=["text"])
        all_yes["text"] = df["text"].append(place_holder['text'])
        all_yes['class'] = 1

        final = all_yes.append(no)

        ## For train/test accuracy



        print(("New size of set is" + str(len(final["text"].values)) + "versus original 18800"))
        cur_f1 = run_class(final)

        print("With " + str(number) + " artifical yes files added and a aug percentage value of " + str(
            percentage) + " the f1 score is " + str(cur_f1))

        if (number == 5944):
            number = 5744 ## Accounting for 200 removed.

        small_train = aug_train[0:number + 1]
        all_yes_train = pd.DataFrame(columns=['text', 'class'])
        place_holder_train = pd.DataFrame(small_train, columns=["text"])
        all_yes_train["text"] = df_train["text"].append(place_holder_train['text'])
        all_yes_train['class'] = 1

        final_train = all_yes_train.append(no_train)

        ## Accuracy stuff

        cur_acc = get_acc(final_train)
        print("The accuracy is " + str(cur_acc))









