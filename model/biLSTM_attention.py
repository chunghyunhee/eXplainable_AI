# bidirectional LSTM with attention model

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout
from tensorflow.keras import Input, Model, Sequential

from model.BahdanauAttention import BahdanauAttention
from sklearn.pipeline import make_pipeline

class biLSTM_attention(object):
    def __init__(self):
        #super(biLSTM_attention, self).__init__(**kwargs)
        self.max_length = 20
        self.num_words = 38

    def load_data(self):
        domain_pad_resampled = np.load("X_train_char2vec.npy")
        target_train_resampled = np.load("X_test_char2vec.npy")

        domain_pad_test = np.load("y_train_char2vec.npy")
        target_test = np.load("y_test_char2vec.npy")

        embedding_matrix_char = np.load("char2vec_embedding_file.npy")

        return embedding_matrix_char, domain_pad_resampled, domain_pad_test, target_train_resampled, target_test

    def base_model(self, embedding_matrix_char, X_train, X_test, y_train, y_test):
        sequence_input = Input(shape = (self.max_length,), dtype = 'int32')
        embedded_sequences = Embedding(self.num_words,
                                       256,
                                       input_length = self.max_length,
                                       weights = [embedding_matrix_char],
                                       mask_zero = True)(sequence_input)
        lstm = Bidirectional(LSTM(64,
                                  dropout=0.5,
                                  return_sequences=True))(embedded_sequences)
        lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(64, dropout=0.5, return_sequences=True, return_state=True))(lstm)

        state_h = Concatenate()([forward_h, backward_h]) # 은닉 상태
        state_c = Concatenate()([forward_c, backward_c]) # 셀 상태

        # bi-LSTM에 attention concat
        attention = BahdanauAttention(64)
        context_vector, attention_weights = attention(lstm, state_h)

        dense1 = Dense(32, activation="relu")(context_vector)
        dropout = Dropout(0.3)(dense1)
        output = Dense(1, activation="sigmoid")(dropout)
        model = Model(inputs = sequence_input, outputs = output)
        model.summary()
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs = 3, batch_size = 256, validation_data=(X_test, y_test), verbose=1)

        return model

    def predict(self, model, X_test, y_test):
        score = model.evaluate(X_test, y_test)
        pred_acc = score[0]
        pred_mse = score[1]

        return pred_acc, pred_mse


    ### test
    def test(self, model):
        test_data = pd.read_csv('./new_test_data.csv')

        ## tokenize
        tokenizer_obj = Tokenizer(char_level = True)
        total_test_domain = test_data['Domain Name'].values
        for i in range(total_test_domain.shape[0]):
            total_test_domain[i] = str(total_test_domain[i])
        tokenizer_obj.fit_on_texts(total_test_domain)

        ## train sequences
        #word_index = tokenizer_obj.word_index
        #vocab_size = len(tokenizer_obj.word_index) + 1
        sequences = tokenizer_obj.texts_to_sequences(total_test_domain)

        ## final data
        domain_pad_new_test = pad_sequences(sequences , maxlen = self.max_length)
        target_new_test = test_data['target'].values

        ## predict
        score = model.evaluate(domain_pad_new_test, target_new_test, verbose = 0)
        acc = score[0]
        mse = score[1]

        return acc, mse


if __name__ == "__main__":
    test_model = biLSTM_attention()
    embedding_matrix, X_train, X_test, y_train, y_test = test_model.load_data()
    model = test_model.base_model(embedding_matrix, X_train, X_test, y_train, y_test)
    pred_acc, pred_mse = test_model.predict(model, X_test, y_test)
    print("acc :", pred_acc)
    print("mse :", pred_mse)

    test_acc, test_mse = test_model.test(model)
    print("acc on test data : ", test_acc)
    print("mse on test data : ", test_mse)








