# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 10:47:15 2022

@author: suhai
"""

import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tokenization
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import tensorflow as tf
import ast
from collections import Counter

import pickle as pkl
# import nltk
# nltk.download('stopwords')

foundations = {"binding": ["individual", "binding"], "moral": ["moral"],
               "full": ["care", "fairness", "loyalty", "authority", "purity"]}

foundations_dict = {"binding": {"harm": "individual", "care": "individual", "degradation": "binding", 
                    "purity": "binding", "betrayal": "binding", "loyalty": "binding", 
                    "subversion": "binding", "authority": "binding",
                    "cheating": "individual", "fairness": "individual", 
                    "non-moral": "non-moral", "nm": "non-moral"},
                    
                    "moral": {"harm": "moral", "care": "moral", "degradation": "moral", 
                                    "purity": "moral", "betrayal": "moral", "loyalty": "moral", 
                                    "subversion": "moral", "authority": "moral",
                                    "cheating": "moral", "fairness": "moral", 
                                    "non-moral": "non-moral", "nm": "non-moral"},
                    "full": {"harm": "care", "care": "care", "degradation": "purity", 
                                        "purity": "purity", "betrayal": "loyalty", "loyalty": "loyalty", 
                                        "subversion": "authority", "authority": "authority",
                                        "cheating": "fairness", "fairness": "fairness", 
                                        "non-moral": "non-moral", "nm": "non-moral"}
                    }

def construct_dataset(data_file, bert_layer, mode):
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
    df = pd.read_csv(data_file)
    X, y = get_needed_fields(df, foundations[mode])
    X, y = pre_process_text(X, y, tokenizer)
    y=np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=0) #maybe take out random state
    X_train = bert_encode(X_train, tokenizer)
    X_test=bert_encode(X_test,tokenizer)
    y_train = tf.convert_to_tensor(y_train, dtype=np.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=np.float32)
    print(len(X_train[0]), len(y_train))

    return X_train, y_train, X_test, y_test

def transformData(mode = "moral"):
    
    df = pd.read_csv("data/mftc_cleaned.csv", index_col = 0).drop_duplicates("tweet_id") #drop duplicates!

    # find majority vote of annotators:  
    # create raw dataframe
    df2 = df.iloc[:, :3].copy().reset_index(drop = True)
    # transform anntotations to list of dicts
    df2.annotations = df2.annotations.apply(lambda x: ast.literal_eval(x))
    # transform to list of moral foundations (combine vices/virtues), count each category only once per annotator (dont count duplicates!)
    df2["cleaned_annotations"] = df2.annotations.apply(lambda x: 
                               list([a for l in x for a in set(map(foundations_dict[mode].get, l["annotation"].split(",")))]))
    # get number of  annotators
    df2["nr_annotators"] = df2.annotations.apply(lambda x: len([l["annotator"] for l in x]))

    #count a tweet for foundation if at least half the annotators choose it and if "non-moral" 
    for foundation in foundations[mode]:
        df2[foundation] = df2.apply(lambda x: 1 if (x["cleaned_annotations"].count(foundation)/x["nr_annotators"] >= 0.5) and 
                                    not all([Counter(x["cleaned_annotations"])["non-moral"] > value 
                                        for key, value in Counter(x["cleaned_annotations"]).items() 
                                        if key != "non-moral"]) else 0, axis = 1)
    
    #Save file
    df2.to_csv("data/mftc_cleaned_combined_" + mode + ".csv")
    
    return 0


def get_needed_fields(df, cols = ["individual", "binding", "non-moral"]):
    
    # cols = ["loyalty", "non-moral", "authority", "purity", "care", "fairness"]
    X = list(df["tweet_text"])
    y = df[cols].values.tolist()
    return X, y


def bert_encode(texts, tokenizer, max_len=256):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for i, text in enumerate(texts):
        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def pre_process_text(X, y, tokenizer):
    tokenized = [tokenizer.tokenize(x) for x in X]
    results = []
    labels = []
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"

    ## remove stop words and make lower case
    # maybe adjust this
    en_stopwords = stopwords.words('english')
    for i, text in enumerate(tokenized):
        out = [token for token in text if (token not in en_stopwords) and (token not in symbols)
               and (not token.startswith("@")
                    and (not token.startswith("http")))]
        if len(out) >= 5:               # remove tweets that are too short after preprocessing
            results.append(out)
            labels.append(y[i])
    return results, labels

if __name__ == '__main__':
    module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
    bert_layer = hub.KerasLayer(module_url, trainable=True)

    # mode = "binding"
    for mode in ["all", "binding", "moral"]:
        transformData(mode)
        
        data_file="data/mftc_cleaned_combined_" + mode + ".csv"
        X_train, y_train, X_test, y_test = construct_dataset(data_file, bert_layer=bert_layer, mode=mode)
        with open("data/train_" + mode + ".pkl","wb") as f:
            pkl.dump([X_train, y_train], f)
        with open("data/test_" + mode + ".pkl","wb") as f:
            pkl.dump([X_test, y_test], f)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
