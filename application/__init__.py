from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

#load data
df_r = pd.read_csv("./wineData/winequality-red.csv", sep = ';')
df_r.drop(['residual sugar', 'chlorides', 'density', 'pH'], axis = 1)
data_r = df_r.values

#Oversample the rare labels
strategy_r = {3.0: 20, 8.0: 20}
OverSample_random = RandomOverSampler(sampling_strategy = strategy_r)
X_res, y_res = OverSample_random.fit_resample(data_r[:,:-1], data_r[:,-1])
X_25percentile = np.percentile(X_res, 25, axis = 0)
X_75percentile = np.percentile(X_res, 75, axis = 0)


#train the model
RF = RandomForestClassifier(n_estimators=300)
RF.fit(X_res, y_res)

#create flask instance
app = Flask(__name__)


#replace -1 with 25th and 75th percentile data of the corresponding features
def process_missing_value(data):
    complete_data_lower = []
    complete_data_upper = []
    for i in range(len(data)):
        if data[i] != -1:
            complete_data_lower.append(data[i])
            complete_data_upper.append(data[i])
        else:
            complete_data_lower.append(X_25percentile[i])
            complete_data_upper.append(X_75percentile[i])
    complete_data_lower = np.array(complete_data_lower)
    complete_data_upper = np.array(complete_data_upper)
    return complete_data_lower, complete_data_upper


#create api
@app.route('/api', methods=['GET', 'POST'])
def predict():
    #get data from request
    data = request.get_json(force=True)
    data = data.values
    #process missing values
    complete_data_lower, complete_data_upper = process_missing_value(data)
    #make predicon using model
    all_quality_scores = data_r[:, -1]
    prediction_lower = RF.predict(complete_data_lower.reshape(1,-1))
    prediction_upper = RF.predict(complete_data_upper.reshape(1,-1))
    prediction_bound_1 = prediction_lower[0]
    prediction_bound_2 = prediction_upper[0]
    if prediction_bound_1 == prediction_bound_2:
        prediction = [prediction_bound_1]
    else:
        prediction = [min(prediction_bound_1, prediction_bound_2), max(prediction_bound_1, prediction_bound_2)]
    #show the rank of the qualitu score
    quality_freq_tabel = Counter(all_quality_scores)
    quality_rank = len(all_quality_scores[all_quality_scores<=prediction])/len(all_quality_scores)
    #feature importance
    feature_importance = RF.feature_importances_
    feature_importance_std = np.std([tree.feature_importances_ for tree in RF.estimators_], axis=0)
    #show high quality wine data
    good_wine_data = data_r[all_quality_scores>=8]
    #construct dict
    statistics = dict({
        'User input data': data, #np.array with dimension of 1: np.array([.....])
        'Predcited quality score': prediction, #if there is no range, return [prediction(integer)], else return [lower bound(integer), upper bound(integer)]
        'All quality scores frequency table': quality_freq_tabel,
        'The rank of the wine among the dataset': quality_rank, #a dictionary{7: 13; 8: 20....}
        # np.array with dimension of 1: np.array([.....]), 11 values corresponding to 11 features
        'Feature importance': feature_importance,
        'Feature importance std': feature_importance_std,
        #np.array (n by 12 columns, including the quality score) of all the data points with a score >= 8
        'High quality wine data': good_wine_data,
    })
    
    return Response(json.dumps(statistics))


temperary_dataset = []
@app.route('/api/getDonatedData', methods=['GET', 'POST'])
def donate_model():
    donated_data = request.get_json(force=True)
    donated_data = donated_data.values.tolist()
    temperary_dataset.append(donated_data)
    if len(temperary_dataset) > 10:
        data_r = np.append(data_r, np.array(temperary_dataset), axis = 0)
        temperary_dataset = []
        RF.fit(data_r)
