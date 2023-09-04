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
from nltk.corpus import stopwords
import glob
# import nltk
# nltk.download('stopwords')

foundations = {"binding": ["individual", "binding"], "moral": ["moral"],
               "full": ["care", "fairness", "loyalty", "authority", "purity"]}

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


def predict(df_list, modes, thresholds):
    module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
    
    bert_layer = hub.KerasLayer(module_url, trainable=True)    
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
    
    for mode, threshold in zip(modes, thresholds):
        cols = foundations[mode]
        model_file = "models/model_" + mode + ".h5"
        model = load_model(model_file, compile=True, custom_objects={"KerasLayer": bert_layer})
        
        result_list = []
        for data_file in df_list:
            df = pd.read_csv(data_file).reset_index(drop=True)
            X_raw = list(df["full_text"])
            X, idx_drop = pre_process_text(X_raw, tokenizer)
            X = bert_encode(X, tokenizer)
            y_pred_proba = model.predict(X)
            y_pred = get_binary(y_pred_proba, threshold)
            df_dropped = df.drop(idx_drop, axis = 0)
            df_dropped[cols] = pd.DataFrame(y_pred, index=df_dropped.index)
            
            # save files! maybe change naming
            if "tweets" in data_file or "stance" in data_file:
                df_dropped.to_csv("results/" + data_file.split("/")[-1].split(".")[0] + "_" + mode + ".csv", index=False)
            else:
                df_dropped.to_csv("results/" + data_file.split("_")[0].split("/")[-1] + "_" + mode + ".csv", index=False)
            result_list.append([df_dropped, y_pred_proba])
    return result_list

modes = ["moral", "binding", "full"]
thresholds = [0.5, 0.5, 0.6] #test this again!

#get annoatations of tweets
files = glob.glob("data/tweets/*.csv")
results = predict(files, modes, thresholds)

files = glob.glob("data/stance/*.csv")
results = predict(files, modes, thresholds)




































