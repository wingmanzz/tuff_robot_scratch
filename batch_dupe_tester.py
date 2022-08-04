import os
import json
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim import corpora
from gensim.similarities import Similarity
import sys
import shutil
import pandas as pd
import glob
import numpy as np



# path = input("Please enter the directory of the folder that should be checked for duplicates: ")
#path = input("Please enter the directory of the folder that should be checked for duplicates: ")
path = "correct_swap/inter_swap/yes"


if not os.path.exists(path):
    print("Article folder " + path + " does not exist")
    sys.exit()

files = os.listdir(path)
documents = []
file_names = []


for file in files:
    file_name = file
    file_open = open(path+"/"+file, encoding='cp437')

    whole = file_open.read()
    file_open.close()
    documents.append(whole)
    file_names.append(file)

#Convert documents to collection of words
texts = [[text for text in simple_preprocess(doc, deacc=True)] for doc in documents]

#Build a bigram model to capture every pair of words in the texts
bigram = Phrases(texts, min_count=1)
bigram_phraser = Phraser(bigram)

texts_bigrams = [[text for text in bigram_phraser[ simple_preprocess(doc, deacc=True)]] for doc in documents]

dictionary = corpora.Dictionary(texts_bigrams)

#Create corpus
corpus = [dictionary.doc2bow(docString) for docString in texts_bigrams]

#Build similarity index
index = Similarity(corpus=corpus, num_features=len(dictionary), output_prefix='on_disk_output')
#Parse similarities from index
doc_id = 0

batch_size = 3000

start_point = 0
end_point = batch_size


threshold = .98 ## change this if you want a stricter/looser definition of similarity

cur_index = start_point
dupe_set = set()
root_set = set()

while end_point <= len(corpus):
    corpus_splice = corpus[start_point:end_point]
    cur_index = start_point
    print(start_point)
    print(end_point)

    for article_index, percentage_list in enumerate(index[corpus_splice]):
        file_col = []


        cur_filename = file_names[cur_index]
        cur_index += 1

        if cur_filename in dupe_set:
            continue
        else:
            root_set.add(cur_filename)

        for internal_index, percentile in enumerate(percentage_list):

            if (article_index == internal_index):
                pass
            elif (percentile >= threshold ):
                file_col.append(file_names[internal_index])


        for dupe in file_col:
            if dupe not in root_set:
                dupe_set.add(dupe)

    start_point = end_point

    if (end_point + batch_size > len(corpus) and len(corpus) + batch_size > end_point + batch_size):
        end_point = len(corpus)
    else:
        end_point += batch_size











