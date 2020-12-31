import pandas as pd
import numpy as np
import re
import random
from collections import Counter
import json
import os

import gensim
import sklearn
import imblearn

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class preprocessing(object):
    def __init__(self, param_json_nm):
        f = open(param_json_nm)
        param_str = f.read()
        f.close()
        self.params = json.loads(param_str)

        # hyperparameter for vectorization
        self.num_features = self.params['num_features']
        self.min_word_count = self.params['min_word_count']
        self.num_workers = self.params['num_workers']
        self.context = self.params['context']
        self.sampling = self.params['sampling']

    def load_data(self):
        data = pd.read_csv("./DGA_dataset_labeling.csv")
        #print(data['0(legit) / 1(dga)'].value_counts())
        return data

    def clean_data(self, word):
        word = re.sub(r'[^A-Za-z0-9\s.]', r'', str(word).lower())
        word = re.sub(r'-', r'', word)
        return word

    # tokenize for word2vector
    def delimiter(self, data, method):
        split_data_list = list()

        # true data에 대해서만 word dict
        if method == "true_data":
            for i in range(len(data['Domain Name'])):
                if data['0(legit) / 1(dga)'][i] == 0:
                    split_data = data['Domain Name'][i].split('.')
                    split_data_list.append(split_data)

        # 전체 데이터에 대해 word dict
        elif method == "all_data":
            for i in range(len(data['Domain Name'])):
                split_data = data['Domain Name'][i].split('.')
                split_data_list.append(split_data)
        return split_data_list


    # tokenize for charcter embedding
    def char_delimiter(self, data, method):
        split_data_list = list()

        # true data만 word dict
        if method == "true_data":
            for i in range(len(data['Domain Name'])):
                if data['0(legit) / 1(dga)'][i] == 0:
                    split_data = data['Domain Name'][i]
                    tmp_data_list = list()
                    for k in range(len(split_data)):
                        tmp_list = list(split_data[k])
                        tmp_data_list += tmp_list

                    split_data_list.append(tmp_data_list)
        # 전체 data로 word dict
        elif method == "all_data":
            for i in range(len(data['Domain Name'])):
                split_data = data['Domain Name'][i]
                tmp_data_list = list()
                for k in range(len(split_data)):
                    tmp_list = list(split_data[k])
                    tmp_data_list += tmp_list
        return split_data_list

    def char_to_ascii(self, split_char_list):
        for i in range(len(split_char_list)):
            for k in range(len(split_char_list[i])):
                split_char_list[i][k] = str(ord(split_char_list[i][k]))
                if split_char_list[i][k] == None :
                    ## not contain ascii, convert 128
                    split_char_list[i][k] = '128'

        # save the data
        split_char_list.to_csv("split_char_list.csv", index=False)
        return split_char_list

    def vectorizer(self, filename, split_data_list, method):
        ## word2vec, char2vec
        if method == "skipgram":
            model = gensim.models.word2vec.Word2Vec(split_data_list,
                                                    workers = self.num_workers,
                                                    size = self.num_features,
                                                    min_count = self.min_word_count,
                                                    window = self.context,
                                                    sample = self.sampling)
            model.init_sims(replace = True)
        else :
            model = gensim.models.word2vec.Word2Vec(split_data_list,
                                                    workers = self.num_workers,
                                                    size = self.num_features,
                                                    min_count = self.min_word_count,
                                                    window = self.context,
                                                    sample = self.sampling,
                                                    sg = 0)
            model.init_sims(replace = True)

        # save the model
        model.wv.save_word2vec_format(filename, binary= False)

        weight_embedding = model.wv.vectors
        vocab_size, embedding_size = weight_embedding.shape

        return vocab_size, embedding_size, weight_embedding

    ## data input for modeling
    def data_reshape(self, data):
        X_train, X_test, y_train, y_test = train_test_split(data['Domain Name'], data['0(legit) / 1(dga)'], test_size = 0.3, shuffle = True)

        #print(y_train.value_counts())
        #print(y_test.value_counts())
        return X_train, X_test, y_train, y_test

    def load_embedding_index(self, filename):
        embeddings_index = {}

        f = open(os.path.join('', filename), encoding = "utf-8")
        for line in f :
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:])
            embeddings_index[word] = coefs
        f.close()

        return embeddings_index

    def load_embedding_matrix(self, embedding_dim, num_words, word_index, embedding_index):
        embedding_matrix = np.zeros((num_words, embedding_dim))

        for word, idx in word_index.items():
            if idx > num_words :
                continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None :
                embedding_matrix[idx] = embedding_vector

        return embedding_matrix


    def tokenize(self,X_train, X_test, y_train, y_test, method):
        # train sequences
        domain_train = X_train.values
        domain_test = X_test.values

        if method == "word2vec":
            tokenize_obj = Tokenizer()
        else :
            tokenize_obj = Tokenizer(char_level = True)

        tokenize_obj.fit_on_texts(domain_train) # fit
        word_index = tokenize_obj.word_index
        vocab_size = len(tokenize_obj.word_index) + 1
        #max_length = max([len(s.split('.')) for s in domain_train])
        sequences = tokenize_obj.texts_to_sequences(domain_train) # transform
        max_length = max([len(s) for s in sequences])
        #print("max length :",max_length) # max_length = 248

        domain_pad = pad_sequences(sequences, maxlen = max_length)
        target_train = y_train.values

        # test sequences
        sequences_test = tokenize_obj.texts_to_sequences(domain_test)
        domain_pad_test = pad_sequences(sequences_test, maxlen = max_length)
        target_test = y_test.values

        return domain_pad, domain_pad_test, target_train, target_test, word_index, max_length

    def tokenize_char_embedding(self, X_train, X_test, y_train, y_test):
        tokenize_obj = Tokenizer(char_level = True)
        domain_train = X_train.values
        domain_test = X_test.values

        split_data_list = list()

        for i in range(domain_train.shape[0]):
            split_data = domain_train[i]
            tmp_data_list = list()
            for k in range(len(split_data)):
                tmp_list = list(split_data[k])
                tmp_data_list += tmp_list
            split_data_list.append(tmp_data_list)
        sequences = self.char_to_ascii(split_data_list)
        max_length = max([len(s) for s in sequences])

        domain_pad = pad_sequences(sequences, maxlen = max_length)
        target_train = y_train.values

        # test sequences
        split_data_list_test = list()

        for i in range(domain_test.shape[0]):
            split_data = domain_test[i]
            tmp_data_list = list()
            for k in range(len(split_data)):
                tmp_list = list(split_data[k])
                tmp_data_list += tmp_list
            split_data_list.append(tmp_data_list)
        sequences_test = self.char_to_ascii(split_data_list_test)

        domain_pad_test = pad_sequences(sequences_test, maxlen = max_length)
        target_test = y_test.values

        return domain_pad, domain_pad_test, target_train, target_test


    ## re sampling
    def re_sampling(self, method, domain, target):
        if method == "downsampling":
            # random downsampling
            rus = imblearn.under_sampling.RandomUnderSampler(random_state = 0)
            rus.fit(domain, target)
            domain_pad_resampled, target_train_resampled = rus.fit_resample(domain, target)
            return domain_pad_resampled, target_train_resampled
        elif method == "oversampling" :
            # SMOTE
            smote = imblearn.over_sampling.SMOTE()
            smote.fit(domain, target)
            domain_pad_resampled, target_train_resampled = smote.fit_resample(domain_pad, target_train)
            return domain_pad_resampled, target_train_resampled





if __name__ == '__main__':
    # preprocessing : char2vec
    preprocess = preprocessing('param_list.json')
    data = preprocess.load_data()
    #data['Domain Name'] = data['Domain Name'].map(lambda x : preprocess.clean_data(x))

    filename = 'char2vec_embedding_file'
    split_data_list = preprocess.char_delimiter(data, "true_data")
    vocab_size, embedding_size, weight_embedding = preprocess.vectorizer(filename, split_data_list, "skipgram")

    # reshape the dataset to put in model and save
    print("save original data...")
    X_train, X_test, y_train, y_test = preprocess.data_reshape(data)
    np.save('./X_train', X_train)
    np.save('./X_test',X_test)
    np.save('./y_train', y_train)
    np.save('./y_test', y_test)


    embedding_index = preprocess.load_embedding_index('char2vec_embedding_file')
    domain_pad, domain_pad_test, target_train, target_test, word_index, max_length = preprocess.tokenize(X_train, X_test, y_train, y_test,"char2vec")
    embedding_matrix = preprocess.load_embedding_matrix(256, len(word_index)+1, word_index, embedding_index)

    #domain_pad_resampled, target_train_resampled = preprocess.re_sampling("downsampling", domain_pad, target_train) # for test
    domain_pad_resampled, target_train_resampled = preprocess.re_sampling("oversampling", domain_pad, target_train)


    # save the preprocessed data
    print('save vectorized data...')
    np.save('./domain_pad', domain_pad)
    np.save('./X_train_char2vec', domain_pad_resampled)
    np.save('./X_test_char2vec', target_train_resampled)
    np.save('./y_train_char2vec',domain_pad_test)
    np.save('./y_test_char2vec', target_test)
    np.save('./char2vec_embedding_file',embedding_matrix)

