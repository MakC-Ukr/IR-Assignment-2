import os
from preprocess import preprocess
import numpy as np
from nltk.tokenize import word_tokenize
from collections import defaultdict

OKGREEN = '\033[92m'
ENDC = '\033[0m'

if __name__ == "__main__":
    queries = ["flow plate", "digital process"]

    # preprocess() # RUN THIS ONCE TO PREPROCESS THE DATA
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_file_names = os.listdir(f"{base_dir}/data")

    for query in queries: 
        query_tokens = set(word_tokenize(query))
        jaccard_scores = {}
        for file_name in all_file_names:
            with open(f'{base_dir}/data/{file_name}', 'r') as file:
                data = file.read()
            txt_tokens = set(word_tokenize(data))
            
            intersection = query_tokens.intersection(txt_tokens)
            union = query_tokens.union(txt_tokens)
            jaccard_val = len(intersection) / len(union)
            jaccard_scores[file_name] = jaccard_val
        
        top_10_keys = sorted(jaccard_scores, key=jaccard_scores.get, reverse=True)[:10]
        print("Query: ", OKGREEN, query, ENDC)
        print("Top 10 results: ", top_10_keys)
        print()
        print()