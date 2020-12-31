import numpy as np
import pandas as pd
from os.path import join
import glob
import tensorflow as tf

class cicids_dataset(object):

    @classmethod
    def get_dataset(cls):
        data, preprocessed_data, label, y_raw = cls.load_data()
        print(data.shape, preprocessed_data.shape, label.shape, y_raw.shape)
        data = cls.normalize(data.to_numpy())
        data, label = cls.balancing_data(data,label)
        # (data, label) : preprocessed numpy data
        # preprocessed data : dataframe
        # y_raw : non encoded label
        return data, preprocessed_data, label, y_raw

    @classmethod
    def load_data(cls):
        filenames = [i for i in glob.glob(join("/../../../../cicids", "*.pcap_ISCX.csv"))]
        #filenames = [i for i in glob.glob(join("/../../../cicids", "*.pcap_ISCX.csv"))]
        #print(filenames)
        if filenames == None:
            print("check file path")

        combined_csv = pd.concat([pd.read_csv(f,dtype=object) for f in filenames],sort=False)
        data = combined_csv.rename(columns=lambda x: x.strip()) #strip() : delete \n & space
        label = data['Label']
        data = data.drop(columns=['Label'])

        #nan_count = data.isnull().sum().sum()
        #if nan_count>0:
            #data.fillna(data.mean(), inplace=True) #replace Nan
            #data.fillna(0, inplace=True)

        #print(data.columns)
        #data = data.dropna(axis = 0, how = 'any')
        #data = data.replace(',,', np.nan, inplace= False)
        data = data.drop(columns=['Fwd Header Length.1'], axis = 1, inplace=False)

        ## inf, ,, del
        data.replace("Infinity", 0, inplace=True)
        data['Flow Bytes/s'].replace("Infinity", 0,inplace=True)

        data["Flow Packets/s"].replace("Infinity", 0, inplace=True)
        data["Flow Packets/s"].replace(np.nan, 0, inplace=True)

        data['Flow Bytes/s'].replace(np.nan, 0,inplace=True)

        data["Bwd Avg Bulk Rate"].replace("Infinity", 0, inplace = True)
        data["Bwd Avg Bulk Rate"].replace(",,",0, inplace = True)
        data["Bwd Avg Bulk Rate"].replace(np.nan, 0, inplace = True)

        data["Bwd Avg Packets/Bulk"].replace("Infinity", 0, inplace= True)
        data["Bwd Avg Packets/Bulk"].replace(",,", 0 ,inplace = True)
        data["Bwd Avg Packets/Bulk"].replace(np.nan, 0, inplace= True)

        data["Bwd Avg Bytes/Bulk"].replace("Infinity", 0, inplace=True)
        data["Bwd Avg Bytes/Bulk"].replace(",,", 0, inplace=True)
        data["Bwd Avg Bytes/Bulk"].replace(np.nan, 0, inplace=True)


        data["Fwd Avg Bulk Rate"].replace("Infinity", 0, inplace=True)
        data["Fwd Avg Bulk Rate"].replace(",,", 0, inplace=True)
        data["Fwd Avg Bulk Rate"].replace(np.nan, 0, inplace=True)


        data["Fwd Avg Packets/Bulk"].replace("Infinity", 0, inplace=True)
        data["Fwd Avg Packets/Bulk"].replace(",,", 0, inplace=True)
        data["Fwd Avg Packets/Bulk"].replace(np.nan, 0, inplace=True)


        data["Fwd Avg Bytes/Bulk"].replace("Infinity", 0, inplace=True)
        data["Fwd Avg Bytes/Bulk"].replace(",,", 0, inplace=True)
        data["Fwd Avg Bytes/Bulk"].replace(np.nan, 0, inplace=True)


        data["CWE Flag Count"].replace("Infinity", 0, inplace=True)
        data["CWE Flag Count"].replace(",,", 0, inplace=True)
        data["CWE Flag Count"].replace(np.nan, 0, inplace=True)

        data["Bwd URG Flags"].replace("Infinity", 0, inplace=True)
        data["Bwd URG Flags"].replace(",,", 0, inplace=True)
        data["Bwd URG Flags"].replace(np.nan, 0, inplace=True)

        data["Bwd PSH Flags"].replace("Infinity", 0, inplace=True)
        data["Bwd PSH Flags"].replace(",,", 0, inplace=True)
        data["Bwd PSH Flags"].replace(np.nan, 0, inplace=True)

        data["Fwd URG Flags"].replace("Infinity", 0, inplace=True)
        data["Fwd URG Flags"].replace(",,", 0, inplace=True)
        data["Fwd URG Flags"].replace(np.nan, 0, inplace=True)

        ## as_type("float64)
        #features=["Fwd Packet Length Max","Flow IAT Std","Fwd Packet Length Std" ,"Fwd IAT Total","Flow Packets/s", "Fwd Packet Length Mean",  "Flow Bytes/s",  "Flow IAT Mean", "Bwd Packet Length Mean",  "Flow IAT Max", "Bwd Packet Length Std"]
        preprocessed_data = data.copy()
        #data = data.astype(features, "float64")
        preprocessed_data.replace('Infinity',0.0, inplace=True)
        preprocessed_data = preprocessed_data.astype(float).apply(pd.to_numeric)

        encoded_label = cls.encode_label(label.values)

        return data, preprocessed_data, encoded_label, label

    @classmethod
    def make_dataset(cls, train:np.ndarray, test:pd.DataFrame, train_label, test_label):
        ## tensor df type
        ds_train = tf.data.Dataset.from_tensor_slices((train, train_label))
        ds_test = tf.data.Dataset.from_tensor_slices((test, test_label))

        test_label = tf.data.Dataset.from_tensor_slices(test_label)

        ds_train = ds_train.shuffle(20000, seed=2).batch(1000, drop_remainder=True)


        ds_test = ds_test.shuffle(20000, seed=2).batch(1000, drop_remainder=True)
        test_label = test_label.shuffle(20000, seed=2).batch(1000, drop_remainder=True)

        test_label = np.stack(list(test_label))

        return ds_train, ds_test, train_label, test_label

    @classmethod
    def split_data(cls, dataset, label) ->(np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        data_size = int(len(dataset))
        train_size = int(0.65 * data_size)

        temp= np.split(dataset,[train_size, data_size, len(dataset)])
        temp_label = np.split(label,[train_size, data_size, len(dataset)])
        return temp[0], temp[1], temp_label[0], temp_label[1]

    @classmethod
    def encode_label(cls, y_str):
        labels_d = cls.make_value2index(np.unique(y_str))
        y = [labels_d[y_str] for y_str  in y_str]
        y = np.array(y)
        return np.array(y)

    @classmethod
    def make_value2index(cls, attacks):
        #make dictionary
        attacks = sorted(attacks)
        d = {}
        counter=0
        for attack in attacks:
            d[attack] = counter
            counter+=1
        return d

    @classmethod
    def normalize(cls, data):
        data = data.astype(np.float32)

        eps = 1e-15

        mask = data==-1
        data[mask]=0
        mean_i = np.mean(data,axis=0)
        min_i = np.min(data,axis=0)
        max_i = np.max(data,axis=0)

        r = max_i-min_i+eps
        data = (data-mean_i)/r

        data[mask] = 0
        return data

    @classmethod
    def balancing_data(cls, train : np.ndarray ,train_label : np.ndarray, seed=2):
        np.random.seed(seed)
        unique, counts = np.unique(train_label, return_counts=True)
        mean_samples_per_class = int(round(np.mean(counts)))
        (n, dtype) = np.shape(train)
        new_x = np.empty((0,dtype))
        new_y = np.empty((0), dtype=int)

        for i,c in enumerate(unique):
            temp_x = train[train_label==c]
            indices = np.random.choice(temp_x.shape[0],mean_samples_per_class)
            new_x = np.concatenate((new_x, temp_x[indices]),axis=0)
            temp_y = np.ones(mean_samples_per_class, dtype=int)*c
            new_y = np.concatenate((new_y, temp_y), axis=0)

        indices = np.arange(new_y.shape[0])
        np.random.shuffle(indices)
        balanced_train =  new_x[indices,:]
        balanced_label = new_y[indices]

        return balanced_train,balanced_label

if __name__ == '__main__' :

    train, test, train_label, test_label= cicids_dataset.get_dataset()

    print("train = {}".format(train.shape))
    print("test = {}".format(test.shape))
    print("train_label = {}".format(train_label.shape))
    print("test_label = {}".format(test_label.shape))

