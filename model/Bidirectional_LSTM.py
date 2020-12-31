from keras.models import Sequential
import numpy as np
from tensorflow.keras.layers import Dense, Embedding,LSTM
from tensorflow.keras import Input, Model, Sequential
from keras.utils import to_categorical

class Bidirectinal_LSTM(object):
    def __init__(self):
        self.max_length = 248
        self.num_words = 41
        self.EMBEDDING_DIM = 256

    def load_dataset(self):
        # 필요한 데이터 load
        embedding_matrix_char = np.load("embedding_matrix_char.npy")
        domain_pad_resampled = np.load("domain_pad_resampled.npy")
        target_train = np.load("target_train.npy")
        domain_pad_test = np.load("domain_pad_test.npy")
        target_test = np.load("target_test.npy")

        ## recategorical
        target_train_recat = to_categorical(target_train).astype(int)
        target_test_tecat = to_categorical(target_test).astype(int)
        return domain_pad_resampled, target_train_recat, domain_pad_test,target_test_tecat, embedding_matrix_char

    def get_model(self, embedding_matrix_char):
        model = Sequential()
        embedding_layer = Embedding(self.num_words,
                                    self.EMBEDDING_DIM,
                                    weights = [embedding_matrix_char],
                                    input_length = self.max_length,
                                    trainable = False)
        model.add(embedding_layer)
        model.add(LSTM(units = 32, dropout = 0.2, recurrent_dropout = 0.2))
        model.add(Dense(2, activation = 'sigmoid'))
        return model

    def learn(self, model, domain_pad_resampled, target_train_recat):
        model.summary()
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics= ['accuracy'])
        model.fit(domain_pad_resampled, target_train_recat, batch_size=2000, epochs=1)
