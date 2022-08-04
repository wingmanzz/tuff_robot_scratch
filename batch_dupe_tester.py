import os
import sys
import tarfile
import shutil
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim import corpora
from gensim.similarities import Similarity
import glob

# Fix for Python 3
try:
    input = raw_input
except NameError:
    pass


def dedupe_dir(yes_dir_path):
    yes_dir_path = yes_dir_path.replace("\\", "/")
    new_path = yes_dir_path + "/**/**/*.txt"
    files = glob.glob(new_path) # grab all txt files.
    documents = []
    file_names = []

    for file in files:
        file_open = open(file, encoding='cp437')
        file_contents = file_open.read()
        documents.append(file_contents)
        file_names.append(file)

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

    batch_size = 3000

    start_point = 0
    end_point = batch_size

    threshold = .98  ## change this if you want a stricter/looser definition of similarity


    dupe_set = set() # Will hold duplicate file names
    root_set = set() # Will hold root file names

    while end_point <= len(corpus): # Corpus = all files + their similarities

        corpus_splice = corpus[start_point:end_point] # Current batch grab

        cur_index = start_point # Where to start grabbing file_names from.

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
                elif (percentile >= threshold):
                    file_col.append(file_names[internal_index])

            for dupe in file_col:
                if dupe not in root_set:
                    dupe_set.add(dupe)

        start_point = end_point

        if (end_point + batch_size > len(corpus) and len(corpus) + batch_size > end_point + batch_size):
            end_point = len(corpus)
        else:
            end_point += batch_size

    print("Number of dupes found: " + str(len(dupe_set)))
    return list(dupe_set) # Returning a list of filenames of dupes to be removed.


def check_for_missing_directories(full_article_path, base_path):
    found_unclassified_dir = False

    if not os.path.exists(full_article_path):
        print("Article directory does not exist: " + full_article_path)
        sys.exit()

    article_group_directories = os.listdir(full_article_path)
    for article_group_directory in article_group_directories:
        full_article_group_dir = os.path.join(full_article_path, article_group_directory)
        display_article_group_dir = os.path.join(base_path, article_group_directory)
        if os.path.isdir(full_article_group_dir):
            year_directories = os.listdir(full_article_group_dir)
            for year_directory in year_directories:
                full_year_dir = os.path.join(full_article_group_dir, year_directory)
                display_year_dir = os.path.join(display_article_group_dir, year_directory)
                if os.path.isdir(full_year_dir):
                    missing_dirs = False
                    full_json_dir = os.path.join(full_year_dir, "json")
                    full_text_dir = os.path.join(full_year_dir, "text")

                    if not os.path.exists(full_json_dir) or not os.path.isdir(full_json_dir):
                        missing_dirs = True
                    else:
                        full_json_yes_dir = os.path.join(full_json_dir, "yes")
                        if not os.path.exists(full_json_yes_dir) or not os.path.isdir(full_json_yes_dir):
                            missing_dirs = True

                    if not os.path.exists(full_text_dir) or not os.path.isdir(full_text_dir):
                        missing_dirs = True
                    else:
                        full_text_yes_dir = os.path.join(full_text_dir, "yes")
                        if not os.path.exists(full_text_yes_dir) or not os.path.isdir(full_text_yes_dir):
                            missing_dirs = True

                    if missing_dirs:
                        print("Missing classifier directories in " + display_year_dir)
                        found_unclassified_dir = True

    return found_unclassified_dir


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))

    print("What is the directory that contains the classified article files?")
    article_directory = input("> ")
    full_article_path = os.path.join(dir_path, article_directory)

    print("What directory do you want all the yes articles written to?")
    yes_article_directory = input("> ")
    full_yes_article_path = os.path.join(dir_path, yes_article_directory)

    if check_for_missing_directories(full_article_path, article_directory):
        print("Missing directories were found in the article directory.")
        print("Run the classifier on these directories before running this script")
        sys.exit()

    try:
        if not os.path.exists(full_yes_article_path):
            os.mkdir(full_yes_article_path)
            print("Created directory " + full_yes_article_path)
    except:
        print("Error creating directory " + full_yes_article_path)
        sys.exit()

    article_group_directories = os.listdir(full_article_path)
    for article_group_directory in article_group_directories:
        full_article_group_dir = os.path.join(full_article_path, article_group_directory)
        if os.path.isdir(full_article_group_dir):
            print("Searching article group directory " + article_group_directory)
            year_directories = os.listdir(full_article_group_dir)
            for year_directory in year_directories:
                full_year_dir = os.path.join(full_article_group_dir, year_directory)
                if os.path.isdir(full_year_dir):
                    # make sure the destination folders exist for this year
                    full_dest_year_dir = os.path.join(full_yes_article_path, year_directory)
                    full_dest_year_json_dir = os.path.join(full_dest_year_dir, "json")
                    full_dest_year_text_dir = os.path.join(full_dest_year_dir, "text")

                    try:
                        if not os.path.exists(full_dest_year_dir):
                            os.mkdir(full_dest_year_dir)
                        if not os.path.exists(full_dest_year_json_dir):
                            os.mkdir(full_dest_year_json_dir)
                        if not os.path.exists(full_dest_year_text_dir):
                            os.mkdir(full_dest_year_text_dir)
                    except:
                        print("Error creating result directories in " + full_yes_article_path)
                        sys.exit()

                    full_json_dir = os.path.join(full_year_dir, "json")
                    full_text_dir = os.path.join(full_year_dir, "text")
                    full_json_yes_dir = os.path.join(full_json_dir, "yes")
                    full_text_yes_dir = os.path.join(full_text_dir, "yes")

                    print("Copying " + year_directory + " json yes files to " + full_dest_year_json_dir)
                    json_files = os.listdir(full_json_yes_dir)
                    for json_file in json_files:
                        if json_file.endswith(".json"):
                            full_source_json_file = os.path.join(full_json_yes_dir, json_file)
                            full_dest_json_file = os.path.join(full_dest_year_json_dir, json_file)
                            shutil.copyfile(full_source_json_file, full_dest_json_file)

                    print("Copying " + year_directory + " text yes files to " + full_dest_year_text_dir)
                    txt_files = os.listdir(full_text_yes_dir)
                    for txt_file in txt_files:
                        if txt_file.endswith(".txt"):
                            full_source_text_file = os.path.join(full_text_yes_dir, txt_file)
                            full_dest_text_file = os.path.join(full_dest_year_text_dir, txt_file)
                            shutil.copyfile(full_source_text_file, full_dest_text_file)

    print("Completed creation of initial yes directory")

    yes_dir_path = full_yes_article_path.replace("\\", "/")
    new_path = yes_dir_path + "/**/**/*.txt"
    files = glob.glob(new_path) # grab all txt files.
    print("Article count before deduping: " + str(len(files)))

    print("Deduping now....Please wait.")

    to_remove = dedupe_dir(full_yes_article_path)

    # file path(s) modification
    to_remove = [e.replace("\\", "/") for e in to_remove]
    json_remove = [e.replace("/text/", "/json/") for e in to_remove]
    json_remove = [e.replace(".txt", ".json") for e in json_remove]
    files = [e.replace("\\", "/") for e in files]

    # removing .txt files
    for dupe_txt in to_remove:
        os.remove(dupe_txt)
    # Removing .json files.
    for dupe_json in json_remove:
        os.remove(dupe_json)

    # Check for correct # of removed text files.
    files = glob.glob(new_path) # grab all txt files.
    print("Article count after deduping: " + str(len(files)))

    print("Duplicates removed from yes directory completed.")



