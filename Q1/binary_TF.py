import os
from preprocess import preprocess
from collections import defaultdict
import numpy as np

if __name__ == "__main__":
    # preprocess() # RUN THIS ONCE TO PREPROCESS THE DATA
    base_dir = os.path.dirname(os.path.abspath(__file__))
    IDF=defaultdict(int)
    TF=defaultdict()
    TF_IDF=defaultdict()
    vocab=defaultdict(list)

    all_file_names = os.listdir(f"{base_dir}/data")
    # TF
    for file_name in all_file_names:
        TF[file_name]=defaultdict(int)
        with open(f'{base_dir}/data/{file_name}', 'r') as file:
            data = file.read()
        tokens = data.split(" ") # Tokenization was done last time in preprocess.py
        for token in tokens:
            TF[file_name][token]=1
            vocab[token].append(file_name)
    # IDF
    for token in vocab.keys():
        IDF[token] = np.log(len(all_file_names)/len(set(vocab[token]))+1)
    del vocab

    # TF-IDF
    for file_name in all_file_names:
        TF_IDF[file_name]=defaultdict(int)
        for token in TF[file_name]:
            TF_IDF[file_name][token]=TF[file_name][token]*IDF[token]

    print(f"TF-IDF[cranfield0005][onedimensional] = {TF_IDF['cranfield0005']['onedimensional']}")