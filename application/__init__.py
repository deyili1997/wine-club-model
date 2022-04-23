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

#train the model
RF = RandomForestClassifier(n_estimators=300)
RF.fit(X_res, y_res)

#create flask instance
app = Flask(__name__)

#create api
@app.route('/api', methods=['GET', 'POST'])
def predict():
    #get data from request
    data = request.get_json(force=True)
    data = data.values
    #make predicon using model
    all_quality_scores = data_r[:, -1]
    prediction = RF.predict(data)
    prediction = prediction[0]
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
        'Predcited quality score': prediction, #an integer value
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
