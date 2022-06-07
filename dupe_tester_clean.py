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


# Responsible for picking out similar documents using the gensim library over a specific threshold of similarity (sim_threshold),
# Splitting files into root_path and all_path folder.

# similar_docs : Result from parsing results from Similarity index - dictionary containing info about similar documents.
# file_names : array of all files
# root_path : where each non-duplicate file should go.
# all_path : where all duplicate files of those in the root_path should go.
# Returns total_relations, which is an array of arrays of similar documents for each file, used for the meta data dataframe.
def identify_similar(similar_docs, file_names, root_path, all_path):
    # adjust this threshold to finetune the results
    sim_threshold = 0.95

    number_of_dupes = 0
    root_dupes = set() # Holds all original root files
    all_others = set() # Holds all duplicates of root files.

    total_relations = [] # For meta-data purposes in main,
    for doc_id, sim_doc_tuples in similar_docs.items():
        this_sim_docs = []
        for sim_doc_tuple in sim_doc_tuples:
            sim_doc_id = sim_doc_tuple[0]
            sim_score = sim_doc_tuple[1]
            if sim_score >= sim_threshold and doc_id != sim_doc_id:
                this_sim_docs.append(file_names[sim_doc_id])

        add_new = True # will determine whether should be added to root or not

        for file_name in this_sim_docs: # for each designated file, check if its already in the root dupe set or not. if it is, don't add current doc, if not, add current doc to root set.
            if file_name in root_dupes:
                add_new = False
                break
        if add_new == True:
            root_dupes.add(file_names[doc_id])
        else:
            all_others.add(file_names[doc_id])
        total_relations.append(this_sim_docs)
    # Sending root , all else items in set to location in dir
    for val in root_dupes:
        file_path = path + "/" + val
        shutil.copy(file_path, root_path)
    for val in all_others:
        file_path = path + "/" + val
        shutil.copy(file_path, all_path)


    # Debugging stuff to verify correct length,

  #  all_others = len(glob.glob('china_no_dupes/all_others/*.txt'))
 #   roots = len(glob.glob('china_no_dupes/root_dupes/*.txt'))

    #print((all_others))
    #print((roots))
   # print((all_others + roots))

    return total_relations

def build_model_and_index(documents):
    # Convert documents to collection of words
    texts = [[text for text in simple_preprocess(doc, deacc=True)] for doc in documents]

    # Build a bigram model to capture every pair of words in the texts
    bigram = Phrases(texts, min_count=1)
    bigram_phraser = Phraser(bigram)

    texts_bigrams = [[text for text in bigram_phraser[simple_preprocess(doc, deacc=True)]] for doc in documents]

    dictionary = corpora.Dictionary(texts_bigrams)

    # Create corpus
    corpus = [dictionary.doc2bow(docString) for docString in texts_bigrams]

    # Build similarity index
    index = Similarity(corpus=corpus, num_features=len(dictionary), output_prefix='on_disk_output')
    # Parse similarities from index
    doc_id = 0
    similar_docs = {}
    for similarities in index:
        similar_docs[doc_id] = list(enumerate(similarities))
        doc_id += 1
    return similar_docs

# Used to scrape the metadata for .txt files for metadata purposes, obtain txt content, file_names for later use.
# Path is where info is coming from in dir.
def obtain_info_from_files(path):
    files = os.listdir(path)
    documents = []
    file_names = []
    meta_data = pd.DataFrame(
        columns=['Filename', 'Source', 'Publisher', 'Byline', 'Copyright', 'Publication Date', 'File Link'])
    for file in files:
        file_name = file
        file_open = open(path + "/" + file, encoding='cp437')
        lines = file_open.readlines()

        file_source = ""
        file_publisher = ""
        file_byline = ""
        file_copyright = ""
        file_pub = ""
        file_link = ""
        file_content = ""
        for line in lines:
            file_content = file_content + line
            if line.startswith("Source: "):
                file_source = line[7:]
            elif line.startswith("Publisher: "):
                file_publisher = line[10:]
            elif line.startswith("Byline: "):
                file_byline = line[7:]
            elif line.startswith("Copyright: (c) "):
                file_copyright = line[14:]
            elif line.startswith("Publication Date: "):
                file_pub = line[17:]
            elif line.startswith("https:"):
                file_link = line

        meta_data.loc[len(meta_data.index)] = [file_name, file_source, file_publisher, file_byline, file_copyright,
                                               file_pub, file_link]
        whole = file_open.read()
        documents.append(file_content)
        file_names.append(file)

    return file_names, meta_data, documents

if __name__ == "__main__":
    path = input("Please enter the directory of the folder that should be checked for duplicates: ")

    if not os.path.exists(path):
        print("Article folder " + path + " does not exist.")
        sys.exit()

    file_names, meta_data, documents = obtain_info_from_files(path)
    similar_docs = build_model_and_index(documents)

    root_path = input("Please enter the directory path for where all root documents should go. ")
    all_path = input("Please enter the directory path for where all dupes of root documents should go.  ")

    total_docs = identify_similar(similar_docs, file_names, root_path, all_path)
    meta_data["Possible Duplicates"] = total_docs
    meta_path = input("Please enter the directory path for the meta data should go.  ")

    meta_data = meta_data[meta_data['Possible Duplicates'].map(lambda d: len(d)) > 0]
    meta_data.to_csv(meta_path+"/meta.csv")




