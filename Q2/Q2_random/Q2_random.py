import pandas as pd
import string
import os
import math
import json
import nltk
import numpy
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
  
# adding cateogry score only if numpy.random gives > 0.1 

porter_stemmer = PorterStemmer()
all_stopwords = set(stopwords.words("english"))

def preprocess_text(txt):
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(txt.lower())
    words = [word for word in words if not word in all_stopwords]
    return " ".join([porter_stemmer.stem(word) for word in words])


def load_data(splitfrac):
    # base_dir = os.path.dirname(os.path.abspath())
    # df_path = os.path.join(base_dir, "data", "BBC News Train.csv")
    whole_df = pd.read_csv("D:\\IR\\BBC News Train.csv")
    whole_df["Text"] = whole_df["Text"].apply(preprocess_text)
    whole_df = whole_df.sample(frac=1).reset_index(drop=True)
    train_df = whole_df[:int(splitfrac * len(whole_df))]
    test_df = whole_df[int(splitfrac * len(whole_df)):]
    X_train = train_df["Text"].values
    y_train = train_df["Category"].values
    X_test = test_df["Text"].values
    y_test = test_df["Category"].values
    return X_train, y_train, X_test, y_test

classes=set()
words = set()

def get_tf_icf_matrix(X_t, Y_c):
    tf={}
    # x_t = [doc1, doc2, doc3]
    # y_t = [category1, catego2, catego3]
    for i in range(len(X_t)):
        category = Y_c[i]
        classes.add(category)
        for word in X_t[i].split():
            words.add(word)
            if word not in tf:
                tf[word] = {}
                
            if category not in tf[word]:
                tf[word][category] = 1
            else:
                tf[word][category] += 1

    cf = {}
    for word in tf:
      num_of_classes = len(tf[word])
      cf[word] = num_of_classes
    
    icf = {}
    for word in tf:
    #   icf_score = math.log10(len(classes) / cf[word])
      icf_score = math.log10(len(Y_c) / cf[word])
      icf[word] = icf_score

    return tf, cf, icf
      
def train(X_t, Y_c):
    docs_in_category = {}
    for category in Y_c:
        if category not in docs_in_category: 
            docs_in_category[category] = 1
        else: 
            docs_in_category[category] += 1

    words_in_category = {}
    # also the probabilty of category
    category_wordcount = {}
    for i in range(len(X_t)):
        category = Y_c[i]

        if category not in category_wordcount:
            category_wordcount[category] = 1 / len(Y_c)
        else:
            category_wordcount[category] += len(X_t[i].split()) / len(Y_c)

        for word in X_t[i].split():
            if category not in words_in_category:
                words_in_category[category] = {}
            
            if word not in words_in_category[category]:
                words_in_category[category][word] = 1
            else:
                words_in_category[category][word] += 1

    return docs_in_category, words_in_category, category_wordcount

def test(X_train, y_train, X_test, y_test, tf, icf, category_wordcount):

    category_predict = []
    for doc in X_test:
        json.dump(doc.split(), open("D:\\IR\\Q2_random\\testdata.json", "a"))

        scores_for_category = {}
        for category in classes:
            scores_for_category[category] = 0
            scores_for_category[category] = docs_in_category[category] / len(y_train)
            for word in doc.split():
                if word in tf:
                    if category in tf[word]:
                        tf_word = tf[word][category] 
                        # probability of category
                        tf_word *= category_wordcount[category]
                        icf_word = icf[word]
                        rand = numpy.random.random()
                        if rand <= 0.1:
                            scores_for_category[category] = 0
                        else:
                            scores_for_category[category] += math.log10(tf_word*icf_word)
                                     
                    else:
                        tf_word = 0
                        icf_word = 0
                        scores_for_category[category] += 0
                else:
                    continue
        scores_for_category_sort = sorted(scores_for_category.items(), key=lambda x:x[1], reverse=True)
        # print(scores_for_category_sort)
        category_predict.append(scores_for_category_sort[0][0])

    # to verify predictions
    ct = 0
    for i in range(len(y_test)):
        if y_test[i] == category_predict[i]:
            ct +=1
        # print(y_test[i], category_predict[i])
    print("Number of correct category predictions: ", ct, " out of ", len(y_test))
    outputstr = "Number of correct category predictions: " + str(ct) +  " out of "+ str(len(y_test))
    return category_predict, outputstr


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(splitfrac = 0.7)

    tf, cf, icf = get_tf_icf_matrix(X_train, y_train)
    json.dump(tf, open("D:\\IR\\Q2_random\\tf.json", "w"))
    json.dump(icf, open("D:\\IR\\Q2_random\\icf.json", "w"))

    docs_in_category, feature_probabilities, class_probabilities = train(X_train, y_train)

    category_predict, outputstr = test(X_train, y_train, X_test, y_test, tf, icf, class_probabilities)
    # json.dump(category_predict, open("final.json", "w"))
    
    output = []
    for i in range(len(category_predict)):
        output.append(f"Actual: {y_test[i]}, Predicted: {category_predict[i]}")
        
    f = open("D:\\IR\\Q2_random\\final.txt", 'a')
    f.write(outputstr)
    f.write("\n")
    for t in output:
        f.write(t)
        f.write("\n")

    # print(category_predict)
    # print(y_test.tolist())
    # labels = ['business', 'politics','tech', 'entertainment', 'sport']
    labels = ['business', 'entertainment','politics', 'sport','tech']
    precision, recall, f1score, support = score(y_test, category_predict,labels = labels)
    accuracy = accuracy_score(y_test, category_predict)
    print(f'accuracy: {accuracy*100}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'fscore: {f1score}')
        

