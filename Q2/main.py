import pandas as pd
import string
import os
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

porter_stemmer = PorterStemmer()
all_stopwords = set(stopwords.words("english"))

def preprocess_text(txt):
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(txt.lower())
    words = [word for word in words if not word in all_stopwords]
    return " ".join([porter_stemmer.stem(word) for word in words])


def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df_path = os.path.join(base_dir, "data", "BBC News Train.csv")
    whole_df = pd.read_csv(df_path)
    whole_df["Text"] = whole_df["Text"].apply(preprocess_text)
    whole_df = whole_df.sample(frac=1).reset_index(drop=True)
    train_df = whole_df[:int(0.7 * len(whole_df))]
    test_df = whole_df[int(0.7 * len(whole_df)):]
    X_train = train_df["Text"].values
    y_train = train_df["Category"].values
    X_test = test_df["Text"].values
    y_test = test_df["Category"].values
    return X_train, y_train, X_test, y_test


def get_df_icf_matrix(X_t, Y_c):
    # Calculating Term frequency
    tf={}
    for i in range(len(X_t)):
        category = Y_c[i]
        for word in X_t[i].split():
            if word not in tf:
                tf[word] = {}
                
            if category not in tf[word]:
                tf[word][category] = 1
            else:
                tf[word][category] += 1

    return tf

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    tf, df = get_df_icf_matrix(X_train, y_train)
    json.dump(tf, open("d.json", "w"))