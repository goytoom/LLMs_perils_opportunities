# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 19:52:39 2022

@author: suhai
"""

import tensorflow_hub as hub
from tensorflow.keras.models import load_model 
import tokenization

import pandas as pd
import numpy as np

import sys
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
bert_layer = hub.KerasLayer(module_url, trainable=True)    
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

foundations = {"mfrc":  {
                    "binding": ["individual", "binding", "proportionality", "thin morality", "non-moral"], 
                    "moral": ["moral", "thin morality", "non-moral"],
                    "full": ["care", "proportionality", "loyalty", "authority", "purity", "equality", "thin morality", "non-moral"],
                    "complete": ["care", "harm", "equality", "proportionality", "loyalty", "betrayal", "authority", "subversion", "purity", "degradation", "thin morality", "non-moral"]
               }
              }

#################### Functions
def pre_process_text(X, tokenizer):
    tokenized = [tokenizer.tokenize(x) for x in X]
    results = []
    drops = []
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"

    en_stopwords = stopwords.words('english')
    for i, text in enumerate(tokenized):
        out = [token for token in text if (token not in en_stopwords) and (token not in symbols)
               and (not token.startswith("@")
                    and (not token.startswith("http")))]
        if len(out) >= 5:               # remove tweets that are too short after preprocessing
            results.append(out)
        else:
            drops.append(i)
    return results, drops

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

def get_binary(_y, threshold):
    y = _y.copy()
    y[y >= threshold] = 1
    y[y < threshold] = 0
    return y


def predict(df, mode, threshold):
   
    model_file = '../models/' + corp + "_" + training + "_" + mode + '.h5'
    model = load_model(model_file, compile=True, custom_objects={"KerasLayer": bert_layer})

    cols = foundations[corp][mode]
    X_raw = list(df["text"])
    X, idx_drop = pre_process_text(X_raw, tokenizer)
    print(len(idx_drop))
    X = bert_encode(X, tokenizer)
    y_pred_proba = model.predict(X)
    y_pred = get_binary(y_pred_proba, threshold)
    df_dropped = df.drop(idx_drop, axis = 0)
    df_dropped[cols] = pd.DataFrame(y_pred, index=df_dropped.index)
    
    # save predictions
    df_dropped.to_csv("../results/predictions/" + corp + "_labels_" + training + "_" + mode + ".csv", index=False)
    return df_dropped

################ Get parameters and run predictions
corp = sys.argv[1]
mode = sys.argv[2]
training = sys.argv[3]
threshold = float(sys.argv[4])

if training == "normal":
    file_path = "../data/preprocessed/" + corp + "_sample_" + mode + ".csv"
else: # potentially add different ways of training
    pass

#get annoatations of texts
file = pd.read_csv(file_path)
results = predict(file, mode, threshold)




































