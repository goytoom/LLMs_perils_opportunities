import pandas as pd
import numpy as np
import pickle as pkl
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model 
from keras.metrics import Precision, Recall

from nltk.corpus import stopwords
import tokenization

from sklearn.metrics import f1_score as fscore
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


foundations = {"mfrc":  {
                    "binding": ["individual", "binding", "proportionality", "thin morality"], 
                    "moral": ["moral", "thin morality"],
                    "full": ["care", "proportionality", "loyalty", "authority", "purity", "equality", "thin morality"],
                    "complete": ["care", "harm", "equality", "proportionality", "loyalty", "betrayal", "authority", "subversion", "purity", "degradation", "thin morality"]
               }
              }

classes = {"mfrc": {"full": 8, "moral": 3, "binding": 5, "complete": 12}}
activation = {"full": "sigmoid", "moral": "sigmoid", "binding": "sigmoid"}

################## Functions
def build_model(bert_layer, max_len=512, classes = 5, activation = "sigmoid"):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    outputs= bert_layer(dict(input_word_ids=input_word_ids,
    input_mask=input_mask,
    input_type_ids=segment_ids))

    # pooled_output=outputs["pooled_output"]
    sequence_output=outputs["sequence_output"]

    clf_output = sequence_output[:, 0, :]
    out = tf.keras.layers.Dense(classes, activation=activation)(clf_output)

    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss='binary_crossentropy', metrics=[Precision(), Recall()])
    return model


def get_binary(_y, threshold):
    y = _y.copy()
    y[y >= threshold] = 1
    y[y < threshold] = 0
    return y

def F1Measure(y_true, y_pred, threshold=0.5):
    y_binary = get_binary(y_pred, threshold)

    return fscore(y_true, y_binary, average = "macro")   

def train(mode):
    bert_layer = hub.KerasLayer(module_url, trainable=True)
    model = build_model(bert_layer, max_len=256, classes = classes[corp][mode], activation = activation[mode])

    with open("../data/train_test/" + corp + "_train_" + mode + ".pkl", "rb") as f:
        X_train, y_train = pkl.load(f)

    checkpoint = tf.keras.callbacks.ModelCheckpoint('../models/' + corp + "_" + training + "_" + mode + '.h5', monitor='val_loss', save_best_only=True, verbose=1)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    print("start training")
    t = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=200,
        callbacks=[checkpoint, earlystopping],
        batch_size=32, #32 works best so far
        verbose=1)
    print("Saving the model")

def crossVal(mode, threshold):
       
    with open("../data/train_test/" + corp + "_train_" + mode + ".pkl", "rb") as f:
        X, y = pkl.load(f)

    model_file = '../models/' + corp + '_' + training + "_" + mode + '_cv.h5'

    print("Start Cross-Validation")
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    cvscores = []
    for train, test in kfold.split(X[0], reverse_onehot(y)): 
        tf.keras.backend.clear_session() # remove any past model from session
        if os.path.isfile(model_file): # remove saved models from checkpoint
            os.remove(model_file)
        else:
            pass

        bert_layer = hub.KerasLayer(module_url, trainable=True)
        model = build_model(bert_layer, max_len=256, classes = classes[corp][mode], activation = activation[mode])
        checkpoint = tf.keras.callbacks.ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True, verbose=1)
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        
        X_train_cv = (X[0][train], X[1][train], X[2][train])
        y_train_cv = tf.gather(y, train)
        X_test_cv = (X[0][test], X[1][test], X[2][test])
        y_test_cv = tf.gather(y, test)
        t = model.fit(
            X_train_cv, y_train_cv,
            validation_data = (X_test_cv, y_test_cv),
            epochs=200,
            callbacks=[checkpoint, earlystopping],
            batch_size=32, #32 works best so far
            verbose=1)

        #load best model from training
        tf.keras.backend.clear_session() 
        model = load_model(model_file, compile=True, custom_objects={"KerasLayer": bert_layer})
        y_pred_val = model.predict(X_test_cv)
        score = F1Measure(y_test_cv, y_pred_val, threshold)
        cvscores.append(score * 100)
        print("%s: %.2f%%" % ("F1-Score (macro average)", score*100))
        
        score2 = fscore(y_test_cv, get_binary(y_pred_val, threshold), average=None)
        print(score2.round(3)*100)        
        
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

def reverse_onehot(onehot_data):
    # onehot_data assumed to be channel last
    data_copy = np.zeros(onehot_data.shape[:-1])
    for c in range(onehot_data.shape[-1]):
        img_c = onehot_data[..., c]
        data_copy[img_c == 1] = c
    return data_copy

############## Load Parameters and Run Training
module_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/2"

corp = sys.argv[1]
mode = sys.argv[2]
training = sys.argv[3]

if training == "eval": # determine best model using CV
    threshold = float(sys.argv[4])
    crossVal(mode, threshold)
elif training == "normal": # regular training (trains/tunes only on training set from MFRC, does not use any test data!!!)
    train(mode)
else: #potentially add different ways of training (e.g., train on completely different corpus and test on MFRC)
    pass
