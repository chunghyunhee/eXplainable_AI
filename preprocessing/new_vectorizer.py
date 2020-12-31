# vectorizer define for lime ( require : fit, transform, fit_transform )
## required to do form check
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class new_vectorizer(object):

    def __init__(self):
        #super(new_vectorizer, self).__init__()
        self.max_length = 248

    def fit(self, raw_data):
        ## for train data : fit
        domain_data = raw_data
        #domain_data = raw_data.values

        tokenize_obj = Tokenizer(char_level = True)
        tokenize_obj.fit_on_texts(domain_data)
        word_index = tokenize_obj.word_index
        vacab_size = len(tokenize_obj.word_index) + 1
        return tokenize_obj

    def transform(self, raw_data):
        ## for train / test data : transform
        token_obj = self.fit(raw_data)
        sequences_test = token_obj.texts_to_sequences(raw_data)
        #max_length = max([len(s) for s in sequences_test])
        domain_pad_test = pad_sequences(sequences_test, maxlen = self.max_length)

        return domain_pad_test

    def fit_transform(self, raw_data):
        # for train data
        token_obj = self.fit(raw_data)
        vectorized_data = self.transform(raw_data)
        return vectorized_data

if __name__ == '__main__':
    new_vectorizer_obj = new_vectorizer()