import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess():
    nltk.download('stopwords')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_file_names = os.listdir(f"{base_dir}/data")

    for file_name in all_file_names:
        with open(f'{base_dir}/data/{file_name}', 'r') as file:
            data = file.read()
        title= data.split("<TITLE>")[1].split("</TITLE>")[0]
        txt = data.split("<TEXT>")[1].split("</TEXT>")[0]
        concat = title + txt
        concat = concat.replace("\n", " ").replace("\t", " ")
        concat = concat.translate(str.maketrans('', '', string.punctuation))

        tokens = word_tokenize(concat)
        sw = set(stopwords.words('english'))
        tokens = [word.lower() for word in tokens if word.lower() not in sw]    
        data = " ".join(tokens)

        with open(f'{base_dir}/data/{file_name}', 'w') as file:
            file.write(data)

if __name__ == "__main__":
    preprocess()