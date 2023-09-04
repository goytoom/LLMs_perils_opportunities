import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model 
from keras.metrics import Precision, Recall

import pandas as pd
import numpy as np
import pickle as pkl
import sys

from nltk.corpus import stopwords
import tokenization

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


foundations = {"mftc": {
                    "binding": ["individual", "binding"], 
                    "moral": ["moral"],
                    "full": ["care", "fairness", "loyalty", "authority", "purity"],
                    "complete": ["care", "harm", "fairness", "cheating", "loyalty", "betrayal", "authority", "subversion", "purity", "degradation"]
                },
               "mfrc":  {
                    "binding": ["individual", "binding", "proportionality", "thin morality"], 
                    "moral": ["moral", "thin morality"],
                    "full": ["care", "proportionality", "loyalty", "authority", "purity", "equality", "thin morality"],
                    "complete": ["care", "harm", "equality", "proportionality", "loyalty", "betrayal", "authority", "subversion", "purity", "degradation", "thin morality"]
               }
              }

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
    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    # net = tf.keras.layers.Dense(32, activation='relu')(net)
    # net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(classes, activation=activation)(net)

    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='binary_crossentropy', metrics=[Precision(), Recall()])
    model.summary()
    return model


def get_binary(_y, threshold):
    y = _y.copy()
    y[y >= threshold] = 1
    y[y < threshold] = 0
    return y

def F1Measure(y_true, y_pred, threshold=0.5):
    y_binary = get_binary(y_pred, threshold)

    return f1_Score(y_true, y_binary, average = "macro")   

def train(mode, bert_layer, corp):
    
    classes = {"mfrc": {"full": 8, "moral": 3, "binding": 5, "complete": 12}, "mftc": {"full": 6, "moral": 2, "binding": 3, "complete": 11}}
    activation = {"full": "sigmoid", "moral": "softmax", "binding": "softmax"}
    model = build_model(bert_layer, max_len=256, classes = classes[corp][mode], activation = activation[mode])

    with open("../data/train_test/" + corp + "_train_" + mode + ".pkl", "rb") as f:
        X_train, y_train = pkl.load(f)

    checkpoint = tf.keras.callbacks.ModelCheckpoint('../models/' + corp + '_normalmodel_' + mode + '.h5', monitor='val_loss', save_best_only=True, verbose=1)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    print("start training")
    t = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=200,
        callbacks=[checkpoint, earlystopping],
        batch_size=32, #32 works best so far
        verbose=1)
    # print(t)
    print("Saving the model")
    # t.save

def crossVal(mode, bert_layer, corp):
    
    classes = {"mfrc": {"full": 8, "moral": 3, "binding": 5, "complete": 12}, "mftc": {"full": 6, "moral": 2, "binding": 3, "complete": 11}}
    activation = {"full": "sigmoid", "moral": "sigmoid", "binding": "sigmoid"}
    model = build_model(bert_layer, max_len=256, classes = classes[corp][mode], activation = activation[mode])

    # print(model.summary())
    with open("../data/train_test/" + corp + "_train_" + mode + ".pkl", "rb") as f:
        X_train, y_train = pkl.load(f)

    # add checkpoint back
        # load model before making prediction (so that the best and not last is used)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint('../models/' + corp + '_model_' + mode + '.h5', monitor='val_loss', save_best_only=True, verbose=1)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    print("Start Cross-Validation")
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    cvscores = []
    for train, test in kfold.split(X_train[0], reverse_onehot(y_train)): #potentially use CV folds as predictions to evaluate against chatGPT
        X_train_cv = (X_train[0][train], X_train[0][train], X_train[0][train])
        y_train_cv = tf.gather(y_train,train)
        X_test_cv = (X_train[0][test], X_train[0][test], X_train[0][test])
        y_test_cv = tf.gather(y_train,test)
        t = model.fit(
            X_train_cv, y_train_cv,
            validation_data = (X_test_cv, y_test_cv),
            # validation_split=0.2,
            epochs=200,
            callbacks=[earlystopping], #[checkpoint, earlystopping]
            batch_size=32, #32 works best so far
            verbose=1)
        
        y_val = model.predict(X_test_cv, y_test_cv)
        scores = F1Measure(y_test_cv, y_val, 0.5)
        print("%s: %.2f%%" % ("f1_score", scores[1]*100))
        cvscores.append(scores[1] * 100)
        
        # print(t)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    # print("Saving the model")
    # t.save

def full_train(mode, bert_layer, corp):
    
    classes = {"mfrc": {"full": 8, "moral": 3, "binding": 5, "complete": 13}, "mftc": {"full": 6, "moral": 3, "binding": 3, "complete": 11}}
    activation = {"full": "sigmoid", "moral": "sigmoid", "binding": "sigmoid"}
    model = build_model(bert_layer, max_len=256, classes = classes[corp][mode], activation = activation[mode])

    with open("../data/train_test/" + corp + "_fulltrain_" + mode + ".pkl", "rb") as f:
        X, y = pkl.load(f)

    checkpoint = tf.keras.callbacks.ModelCheckpoint('../models/' + corp + '_crossmodel_' + mode + '.h5', monitor='val_loss', save_best_only=True, verbose=1)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    print("start training")
    t = model.fit(
        X, y,
        validation_split = 0.2,
        epochs=200,
        callbacks=[checkpoint, earlystopping],
        batch_size=32, #32 works best so far
        verbose=1)
    # print(t)
    print("Saving the model")
    # t.save

def reverse_onehot(onehot_data):
    # onehot_data assumed to be channel last
    data_copy = np.zeros(onehot_data.shape[:-1])
    for c in range(onehot_data.shape[-1]):
        img_c = onehot_data[..., c]
        data_copy[img_c == 1] = c
    return data_copy

############## Load Parameters and Run Training
module_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/2"
bert_layer = hub.KerasLayer(module_url, trainable=True)

corp = sys.argv[1]
mode = sys.argv[2]
training = sys.argv[3]

data_file = "../data/" + corp + "_test_" + mode + ".pkl"
model_file = "../models/" + corp + "_model_" + mode + ".h5"

if training == "eval": # determine best model using CV
    crossVal(mode, bert_layer, corp)
elif training == "cross": # train on full corpus for cross corpus predictions
    full_train(mode, bert_layer, corp)
elif training == "normal": # regular training for test sample (against chatGPT)
    train(mode, bert_layer, corp)
else:
    pass
    