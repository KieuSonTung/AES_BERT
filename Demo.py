# import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, r2_score
import torch
import tensorflow as tf
from tensorflow.keras.utils import plot_model
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense, Dropout, Lambda, Bidirectional
from tensorflow.keras.models import Sequential, Model, load_model, model_from_config
from tensorflow.keras.optimizers import Adam
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')


# method to split data into sets
def split_in_sets(data):
    essay_sets = []

    for s in range(1, 9):
        essay_set = data[data["essay_set"] == s]
        essay_set.dropna(axis=1, inplace=True)
        n, d = essay_set.shape
        print("Set", s, ": Essays = ", n, "\t Attributes = ", d)
        essay_sets.append(essay_set)

    return essay_sets


def get_model(Hidden_dim1=400, Hidden_dim2=128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3,
              input_size_x=3, input_size_y=768, activation='relu', bidirectional=False):
    """Define the model."""
    model = Sequential()
    if bidirectional:
        model.add(Bidirectional(
            LSTM(Hidden_dim1, return_sequences=return_sequences, dropout=0.4, recurrent_dropout=recurrent_dropout),
            input_shape=[input_size_x, input_size_y]))
        model.add(
            Bidirectional(LSTM(Hidden_dim2, recurrent_dropout=recurrent_dropout, return_sequences=return_sequences)))
        model.add(Bidirectional(LSTM(Hidden_dim2, recurrent_dropout=recurrent_dropout)))
    else:
        model.add(LSTM(Hidden_dim1, dropout=0.4, recurrent_dropout=recurrent_dropout,
                       input_shape=[input_size_x, input_size_y], return_sequences=return_sequences))
        model.add(LSTM(Hidden_dim2, recurrent_dropout=recurrent_dropout, return_sequences=return_sequences))
        model.add(LSTM(Hidden_dim2, recurrent_dropout=recurrent_dropout))

    model.add(Dropout(dropout))
    model.add(Dense(128, activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['mse', 'mae', tf.keras.losses.MeanAbsolutePercentageError()])
    model.summary()

    return model


def convert_to_tensors(input_ids, chunksize=512):
    input_id_chunks = list(input_ids)

    if len(input_id_chunks) > 1:
        for i, j in enumerate(input_id_chunks):
            input_id_chunks[i] = torch.cat([torch.Tensor([101]), j, torch.Tensor([102])])

            pad_len = chunksize - input_id_chunks[i].shape[0]

            if pad_len > 0:
                input_id_chunks[i] = torch.cat([input_id_chunks[i], torch.Tensor([0] * pad_len)])

            # print(input_id_chunks[i])

        input_ids_tensor = torch.stack(input_id_chunks)

    else:
        input_id_chunks = torch.cat([torch.Tensor([101]), input_id_chunks[0], torch.Tensor([102])])

        pad_len = chunksize - input_id_chunks.shape[0]

        if pad_len > 0:
            input_id_chunks = torch.cat([input_id_chunks, torch.Tensor([0] * pad_len)])

        input_ids_tensor = input_id_chunks.view(1, chunksize)

    return input_ids_tensor


def get_longest_tensor(X_train):
    max_length = 0

    for tensor in X_train:
        if len(tensor) > max_length:
            max_length = len(tensor)

    return max_length


def add_padding_tensors(x, longest_tensor, chunksize=512):
    if len(x) < longest_tensor:
        no_padding_left = longest_tensor - len(x)

        zero_padding = torch.Tensor([0] * chunksize)
        zero_paddings = zero_padding.repeat(no_padding_left, 1)
        final_tensor = torch.cat((x, zero_paddings), 0)

    else:
        final_tensor = x

    return final_tensor


def preprocess_essays(X):
    model_class, tokenizer_class, pretrained_weights = (
    ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    essays = X['essay']

    tokenized_essay = essays.apply((lambda x: tokenizer.encode(x, add_special_tokens=False, return_tensors='pt')))
    tokenized_essay = tokenized_essay.apply(lambda x: x[0].split(510) if len(x[0]) > 510 else x)
    tokenized_essay = tokenized_essay.reset_index()['essay']
    tokenized_essay = tokenized_essay.apply(lambda x: convert_to_tensors(x))

    longest_tensor = get_longest_tensor(tokenized_essay)
    tokenized_essay = tokenized_essay.apply(lambda x: add_padding_tensors(x, longest_tensor))

    tokenized_essay = tokenized_essay.to_numpy()
    tokenized_essay = np.array([np.array(val) for val in tokenized_essay])

    return tokenized_essay


def predict_score(set, tokenized_essays):
    model = get_model(bidirectional=True, input_size_x=tokenized_essays.shape[1],
                      input_size_y=tokenized_essays.shape[2])

    saved_model_path = 'saved_model/set{}/cp1.h5'.format(set)

    model.load_weights(saved_model_path)

    y_pred = model.predict(tokenized_essays)
    y_pred = np.around(y_pred)
    y_pred = y_pred.reshape(1, -1)[0]

    return y_pred


def predict_score_df(df):
    essay_sets = split_in_sets(df)

    results_ls = []

    for sets, essays in enumerate(tuple(essay_sets)):
        tokenized_essays = preprocess_essays(essays)
        y_pred = predict_score(sets + 1, tokenized_essays)
        print('Scores of set', sets + 1, ':', y_pred)
        essays['predicted score'] = y_pred
        # print(essays)
        results_ls.append(essays)

    results_df = pd.concat(results_ls, ignore_index=True, axis=0)

    return results_df


def main():
    df = pd.read_excel('data/test_set.xlsx')

    predicted_df = predict_score_df(df)

    print(predicted_df)

    # Export to excel file
    # predicted_df.to_excel('./results/test_set_predicted_score.xlsx')

    print('Done!')


if __name__ == "__main__":
    main()