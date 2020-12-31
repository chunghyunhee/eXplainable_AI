## xai with LIME & SHAP / Debugging LIME

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import  Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer
from lime import submodular_pick

from preprocessing.new_vectorizer import new_vectorizer
from model.BahdanauAttention import BahdanauAttention
from data.cicids_dataset import cicids_dataset

## lrp
from numpy import newaxis as na

class explainer(object):
    def __init__(self):
        self.max_length = 248
        self.num_words = 41

    # tabular data
    def get_tabular_data(self):
        ## cicids
        data, preprocessed_data, label, raw_label = cicids_dataset.get_dataset()
        print(data.shape, label.shape)

        return data, preprocessed_data, label, raw_label

    # text data
    def load_text_resampled_data(self):
        domain_pad_resampled = np.load("X_train_char2vec.npy")
        target_train_resampled = np.load("X_test_char2vec.npy")

        domain_pad_test = np.load("y_train_char2vec.npy")
        target_test = np.load("y_test_char2vec.npy")

        embedding_matrix_char = np.load("char2vec_embedding_file.npy")
        domain_pad = np.load("domain_pad.npy")

        ## check
        print(domain_pad_resampled.shape)
        print(domain_pad_test.shape)

        return embedding_matrix_char, domain_pad_resampled, domain_pad_test, target_train_resampled, target_test, domain_pad

    def load_data(self):
        data = pd.read_csv("./DGA_dataset_labeling.csv")
        X_train, X_test, y_train, y_test = train_test_split(data['Domain Name'], data['0(legit) / 1(dga)'], test_size = 0.3, shuffle = True)
        return X_train, X_test, y_train, y_test


    ## model learn, predict
    def train_model(self, model, data, labels, test_data, test_labels, method):
        if method == "text":
            pipe = Pipeline([('clf', model["clf"])])
        elif method == "tabular":
            pipe = Pipeline([('scaler', StandardScaler()),('clf', model["clf"])])
        else : # image
            pipe = Pipeline([('clf', model["clf"])])

        start_time = time.time()
        pipe.fit(data, labels)
        train_time = time.time() - start_time

        pred = model["clf"].predict(data)
        train_accuracy = accuracy_score(labels, pred)

        pred_test = model["clf"].predict(test_data)
        test_accuracy = accuracy_score(test_labels, pred_test)
        model_details = {"name": model["name"], "train_accuracy":train_accuracy, "test_accuracy":test_accuracy, "train_time": train_time, "model": pipe}

        return model_details


    def sklearn_model(self, domain_pad_resampled, target_train_resampled, domain_pad_test, target_test, method):
        trained_models = list()

        models = [
            {"name": "Naive Bayes" ,"clf": GaussianNB()}]#,
            #{"name": "logistic regression", "clf": LogisticRegressionCV(max_iter=500)},
            #{"name": "Decision Tree", "clf": DecisionTreeClassifier()},
            #{"name": "Random Forest", "clf": RandomForestClassifier(n_estimators=100)},
            #{"name": "Gradient Boosting", "clf": GradientBoostingClassifier(n_estimators=100)},
            #{"name": "MLP Classifier", "clf": MLPClassifier(solver='adam', alpha=1e-1, hidden_layer_sizes=(32,32,16,2), max_iter=500, random_state=42)}]

        for model in models:
            model_details = self.train_model(model, domain_pad_resampled, target_train_resampled, domain_pad_test, target_test, method)
            print(model_details)
            trained_models.append(model_details)

        return trained_models

    def keras_model(self, embedding_matrix_char, X_train, X_test, target_train, target_test):
        ## data shape
        target_train_recat = to_categorical(target_train).astype(int)
        target_test_tecat = to_categorical(target_test).astype(int)

        ## length check
        print(target_train_recat.shape)
        print(target_test_tecat.shape)
        print(len(X_train))
        print(len(X_test))

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

        state_h = Concatenate()([forward_h, backward_h])
        #state_c = Concatenate()([forward_c, backward_c])

        # bi-LSTM에 attention concat
        attention = BahdanauAttention.BahdanauAttention(64)
        context_vector, attention_weights = attention(lstm, state_h)

        dense1 = Dense(32, activation="relu")(context_vector)
        dropout = Dropout(0.3)(dense1)
        #output = Dense(1, activation="sigmoid")(dropout)
        output = Dense(2, activation="sigmoid")(dropout)
        model = Model(inputs = sequence_input, outputs = output)

        model.summary()
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy','mse'])
        #history = model.fit(X_train, target_train_recat, epochs = 1, batch_size = 256, validation_data=(X_test, target_test_tecat), verbose=1)
        model.fit(X_train, target_train_recat, epochs = 1, batch_size = 512, validation_data=(X_test, target_test_tecat), verbose=1)

        output_pred = model.predict(X_train)
        return model, output_pred


    def visualize_models(self, trained_models):
        model_df = pd.DataFrame(trained_models)
        model_df.sort_values("test_accuracy", inplace=True)
        ax = model_df[["train_accuracy","test_accuracy", "name"]].plot(kind="line", x="name", figsize=(19,5), title="Classifier Performance Sorted by Test Accuracy")
        ax.legend(["Train Accuracy", "Test Accuracy"])

        for p in ax.patches:
            ax.annotate( str( round(p.get_height(),3) ), (p.get_x() * 1.005, p.get_height() * 1.005))
        ax.title.set_size(20)
        plt.box(False)
        #fig = ax.get_figure()
        #fig.savefig('train performance_output.obg')

        model_df.sort_values("train_time", inplace=True)
        ax= model_df[["train_time","name"]].plot(kind="line", x="name",figsize=(19,5),grid=True,title="Classifier Training Time (seconds)")
        ax.title.set_size(20)
        ax.legend(["Train Time"])
        plt.box(False)
        #fig = ax.get_figure()
        #fig.savefig("train_time_output.png")


    ### LIME
    def get_lime_explainer(self,method, model = None, X_data = None, y_data = None, raw_y_data = None):
        if method == "text":
            class_names = ['0(legit)', '1(dga)']
            lime_explainer = LimeTextExplainer(class_names = class_names,
                                        char_level = True,
                                        bow = False)
        elif method == "tabular":
            cat_feat_ix = [i for i,c in enumerate(X_data.columns) if pd.api.types.is_categorical_dtype(X_data[c])]
            feat_names = list(X_data.columns)
            class_names = list(raw_y_data.unique())
            scaler = model["model"]["scaler"]
            data = scaler.transform(X_data)

            lime_explainer = LimeTabularExplainer(data,
                                                  feature_names = feat_names,
                                                  class_names=class_names,
                                                  categorical_features = cat_feat_ix,
                                                  mode ="classification")
        else : # image
            class_names = ['0(legit)', '1(dga)']
            lime_explainer = LimeTextExplainer(class_names = class_names,
                                               char_level = True,
                                               bow = False)


        return lime_explainer

    def lime_explain(self, explainer, data, predict_method, num_features = 6):
        explanation = explainer.explain_instance(data, predict_method, num_features=num_features)
        #explanation.show_in_notebooks(text = True)
        print(explanation.as_list())
        return explanation

## explainer
    def sklearn_explainer(self, X_train, trained_models, method, domain_pad = None, new_vectorizer_obj = None, data = None, raw_y = None  ):
        ## 지정한 5개의 모델에 대해 판단 기준이 되는 weight를 print
        if method == "text":
            lime_data_explainations = []
            lime_metrics = []
            tmp_idx = 0 # for domain pad
            lime_explanation_time = []
            test_data_index = X_train.index[tmp_idx]

            for current_model in trained_models:
                feat_names = list(X_train[test_data_index])
                predict_method = make_pipeline(new_vectorizer_obj, current_model["model"]["clf"])
                start_time = time.time()
                lime_explainer = self.get_lime_explainer(method)
                explanation = self.lime_explain(lime_explainer,
                                                X_train[test_data_index],
                                                #np.reshape([X_train[test_data_index]], (-1,1)),
                                                predict_method.predict_proba)
                elapsed_time = time.time() - start_time

                ## save each result
                ex_holder = {}
                for feat_index,ex in explanation.as_map()[1] :
                    ex_holder[feat_names[feat_index]] = ex
                lime_explanation_time.append({"time": elapsed_time, "model": current_model["name"] })
                print(lime_explanation_time)
                lime_data_explainations.append(ex_holder)
                actual_pred = current_model["model"]["clf"].predict_proba(domain_pad[tmp_idx].reshape(1,-1))
                perc_pred_diff =  abs(actual_pred[0][1] - explanation.local_pred[0])
                lime_metrics.append({"lime class1": explanation.local_pred[0], "actual class1": actual_pred[0][1], "class_diff": round(perc_pred_diff,3), "model": current_model["name"] })
            return lime_metrics

        elif method == "tabular":
            lime_data_explainations = []
            lime_metrics = []
            lime_explanation_time = []
            feat_names = list(data.columns)
            test_data_index = 6

            for current_model in trained_models :
                #feat_names = list(X_train[test_data_index])
                scaler = current_model["model"]["scaler"]
                scaled_test_data = scaler.transform(X_train)
                predict_method = current_model["model"]["clf"].predict_proba
                start_time = time.time()
                lime_explainer = self.get_lime_explainer(method, model=current_model, X_data=preprocessed_data, y_data=label, raw_y_data=raw_y)
                explanation = self.lime_explain(lime_explainer,
                                                scaled_test_data[test_data_index],
                                                #np.reshape([X_train[test_data_index]], (-1,1)),
                                                predict_method)
                elapsed_time = time.time() - start_time

                ## save each result
                ex_holder = {}
                for feat_index, ex in explanation.as_map()[1]:
                    ex_holder[feat_names[feat_index]] = ex
                lime_explanation_time.append({"time": elapsed_time, "model": current_model["name"] })
                print(lime_explanation_time)
                lime_data_explainations.append(ex_holder)
                actual_pred = predict_method(scaled_test_data[test_data_index].reshape(1, -1))
                perc_pred_diff =  abs(actual_pred[0][1] - explanation.local_pred[0])
                lime_metrics.append({"lime class1": explanation.local_pred[0], "actual class1": actual_pred[0][1], "class_diff": round(perc_pred_diff,3), "model": current_model["name"] })
            return lime_metrics

#### CASE 1 : 하나의 instance에 대한 output
    def keras_explainer(self, domain_pad, model, new_vectorizer_obj):
        ## 앞에서 학습한 모델 그대로 사용
        lime_data_explainations = []
        lime_metrics = []
        index_list = []
        ## idx check
        tmp_idx = 0
        test_data_index = X_train.index[tmp_idx]

        feat_names = list(X_train[test_data_index])
        predict_method = make_pipeline(new_vectorizer_obj, model)
        lime_explainer = self.get_lime_explainer("text")
        explanation = self.lime_explain(lime_explainer,
                                        X_train[test_data_index],
                                        #np.reshape([X_train[test_data_index]], (-1,1)),
                                        predict_method.predict)

        ## save each result
        ex_holder = {}
        index_list = list()
        for feat_index,ex in explanation.as_map()[1] :
            ex_holder[feat_names[feat_index]] = ex
            index_list.append(feat_index)

        index_list.append(index_list)
        lime_data_explainations.append(ex_holder)
        actual_pred = model.predict(domain_pad[tmp_idx].reshape(1,-1))
        perc_pred_diff =  abs(actual_pred[0][1] - explanation.local_pred[0])
        lime_metrics.append({"lime class1": explanation.local_pred[0], "actual class1": actual_pred[0][1], "class_diff": round(perc_pred_diff,3), "model": "LSTM attention" })
        print(lime_metrics)
        return lime_metrics

    def debugging_lime(self, lime_metrics):
        ## local model이 global에 대해 설명력있는지 확인
        lime_metrics_df = pd.DataFrame(lime_metrics)
        lime_metrics_df_ax = lime_metrics_df[["lime class1", "actual class1", "model"]].plot(kind="line", x="model", title="LIME Actual Prediction vs Local Prediction ", figsize=(22,6))
        lime_metrics_df_ax.title.set_size(20)
        lime_metrics_df_ax.legend(["Lime Local Prediction", "Actual Prediction"])
        plt.box(False)

        # save img
        fig = lime_metrics_df_ax.get_figure()
        fig.savefig("explainer_result.png")

### CASE 2 : 예측이 틀린값에 대해 explain되는 index 확인
    def Idx(self, pred_result, target):
        pred_result_list = list()
        tmp_idx_list = list()
        tmp_idx_cor_list = list()
        for i in range(pred_result.shape[0]):
            if pred_result[i][0] > pred_result[i][1]:
                pred_result_list.append(0)
            else :
                pred_result_list.append(1)

        for i in range(len(target)):
            if target[i] != pred_result_list[i]:
                tmp_idx_list.append(i)
            else :
                tmp_idx_cor_list.append(i)

        return tmp_idx_list, tmp_idx_cor_list

    def evidence_all(self, model, pred_result, target_train, tmp_idx_list, X_train ):
        predict_method = make_pipeline(new_vectorizer_obj, model)

        actual_label = list()
        pred_label = list()
        value_list= list()
        class_0 = list()
        class_1 = list()
        #iter_num = 5 # for test
        iter_num = len(tmp_idx_list)

        for i in range(iter_num):
            # append value
            value_list.append(X_train[X_train.index[tmp_idx_list[i]]])
            actual_label.append(target_train[tmp_idx_list[i]])
            pred_label.append(pred_result[tmp_idx_list[i]])
            print(X_train[X_train.index[tmp_idx_list[i]]])

            # explainer
            lime_explainer = self.get_lime_explainer(method = "text")
            explanation = self.lime_explain(lime_explainer,
                                            X_train[X_train.index[tmp_idx_list[i]]],
                                            predict_method.predict)

            # result
            tmp_0_list = list()
            tmp_1_list = list()

            for feat_idx, value in explanation.as_map()[1]:
                if value < 0 :
                    tmp_0_list.append(feat_idx)
                else :
                    tmp_1_list.append(feat_idx)
            class_0.append(tmp_0_list)
            class_1.append(tmp_1_list)

        # save to csv
        all_list = [ value_list, actual_label, pred_label, class_0, class_1 ]
        df = pd.DataFrame(all_list).transpose()
        df.columns = ['Instance', 'Actual label', 'Pred label', 'class0 Evidence', 'class1 Evidence']
        df.to_csv("explainer_1203.csv", index = False)

    def SP_LIME(self, X_train, trained_models, method, domain_pad = None, new_vectorizer_obj = None, data = None, raw_y = None  ):
        if method == "text":
            tmp_idx = 0 # for domain pad
            test_data_index = X_train.index[tmp_idx]

            for current_model in trained_models:
                feat_names = list(X_train[test_data_index])
                predict_method = make_pipeline(new_vectorizer_obj, current_model["model"]["clf"])
                start_time = time.time()
                lime_explainer = self.get_lime_explainer(method)
                sp_obj = submodular_pick.SubmodularPick(lime_explainer, X_train.values, predict_method.predict_proba, num_features = 5, num_exps_desired = 4)
                elapsed_time = time.time() - start_time
                print("SP-LIME execution time : ", elapsed_time)
                #fig_list = [exp.as_pyplot_figure(label=1) for exp in sp_obj.sp_explanations]
                fig_list = [exp.show_in_notebook() for exp in sp_obj.sp_explanations]

                ## save the image
                for idx in range(len(fig_list)):
                    fig_tmp = fig_list[idx]
                    file_name = str(idx) + "sp-lime" + "text"
                    fig_tmp.savefig(file_name)

        elif method == "tabular":
            test_data_index = 6

            for current_model in trained_models :
                #feat_names = list(X_train[test_data_index])
                scaler = current_model["model"]["scaler"]
                scaled_test_data = scaler.transform(X_train)
                predict_method = current_model["model"]["clf"].predict_proba
                start_time = time.time()
                lime_explainer = self.get_lime_explainer(method, model=current_model, X_data=preprocessed_data, y_data=label, raw_y_data=raw_y)
                sp_obj = submodular_pick.SubmodularPick(lime_explainer, scaled_test_data.values, predict_method, num_features = 5, num_exps_desired = 4)
                elapsed_time = time.time() - start_time
                print("SP-LIME execution time : ", elapsed_time)
                #fig_list = [exp.as_pyplot_figure(label=1) for exp in sp_obj.sp_explanations]
                #fig_list = [exp.show_in_notebook() for exp in sp_obj.sp_explanations]
                fig_list = list()
                for exp in sp_obj.sp_explanations:
                    fig_list.append(exp.to_list())
                    print(exp.to_list())

                ## save the image
                for idx in range(len(fig_list)):
                    fig_tmp = fig_list[idx]
                    file_name = str(idx) + "sp-lime" + "tabular"
                    fig_tmp.savefig(file_name)











if __name__ == '__main__':
    explainers = explainer()

    # text
    ## load data
    print("domain text data ")
    embedding_matrix_char, domain_pad_resampled, domain_pad_test, target_train_resampled, target_test, domain_pad = explainers.load_text_resampled_data()
    X_train, X_test, y_train, y_test = explainers.load_data()

    ## learn model
    ### sklearn
    #trained_models = explainers.sklearn_model(domain_pad_resampled, target_train_resampled, domain_pad_test, target_test, "text")
    #explainers.visualize_models(trained_models)
    ### keras
    trained_models_keras, output_pred = explainers.keras_model(embedding_matrix_char, domain_pad_resampled, domain_pad_test, target_train_resampled, target_test)

    ## LIME & output csv file
    print("start lime explainer ")
    new_vectorizer_obj = new_vectorizer.new_vectorizer()
    #lime_metrics_sklearn = explainers.sklearn_explainer(X_train, trained_models, "text", domain_pad = domain_pad, new_vectorizer_obj=new_vectorizer_obj)
    lime_metrics_keras = explainers.keras_explainer(domain_pad, trained_models_keras, new_vectorizer_obj)
    print("save the results with index value ")
    idx_not_list, idx_list = explainers.Idx(output_pred, target_train_resampled)
    explainers.evidence_all(trained_models_keras, output_pred, target_train_resampled, idx_not_list, X_train)

    ## debugging LIME & SP-LIME
    #explainers.debugging_lime(lime_metrics_sklearn)
    #print("start SP-LIME ")
    #explainers.SP_LIME(X_train, trained_models, "text", domain_pad = domain_pad, new_vectorizer_obj=new_vectorizer_obj)

    '''
    #tabular
    ## load data
    print("cicids tabular data ")
    data, preprocessed_data, label, raw_y = explainers.get_tabular_data() # pre-processed 같이 return
    X_train_table, X_test_table, y_train_table, y_test_table = train_test_split(data, label)

    ## learn model
    trained_models_table = explainers.sklearn_model(X_train_table, y_train_table, X_test_table, y_test_table, "tabular")
    explainers.visualize_models(trained_models_table)

    ## LIME
    print("start lime explainer ")
    lime_metrics_sklearn_tabular = explainers.sklearn_explainer(X_train_table, trained_models_table, "tabular", data = preprocessed_data, raw_y=raw_y)

    ## debudding LIME
    explainers.debugging_lime(lime_metrics_sklearn_tabular)

    ## SP-LIME
    #print("start SP-LIME ")
    #explainers.SP_LIME(X_train_table, trained_models_table, "tabular", data = preprocessed_data, raw_y=raw_y)
    '''




