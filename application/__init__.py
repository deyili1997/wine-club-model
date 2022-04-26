from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

#load data
df_r = pd.read_csv("./wineData/winequality-red.csv", sep = ';')
df_r.drop(['residual sugar', 'chlorides', 'density', 'pH'], axis = 1, inplace=True)
data_r = df_r.values

df_w = pd.read_csv("./wineData/winequality-white.csv", sep = ';')
df_w.drop(['citric acid', 'density', 'pH', 'sulphates'], axis = 1, inplace=True)
data_w = df_w.values

#Oversample the rare labels
strategy_r = {3.0: 20, 8.0: 20}
OverSample_random_r = RandomOverSampler(sampling_strategy = strategy_r)
X_res_r, y_res_r = OverSample_random_r.fit_resample(data_r[:,:-1], data_r[:,-1])
X_25percentile_r = np.percentile(data_r, 25, axis = 0)
X_50percentile_r = np.percentile(data_r, 50, axis = 0)
X_75percentile_r = np.percentile(data_r, 75, axis = 0)


strategy_w = {3.0: 30, 9.0: 30}
OverSample_random_w = RandomOverSampler(sampling_strategy = strategy_w)
X_res_w, y_res_w = OverSample_random_w.fit_resample(data_w[:,:-1], data_w[:,-1])
X_25percentile_w = np.percentile(data_w, 25, axis = 0)
X_50percentile_w = np.percentile(data_w, 50, axis = 0)
X_75percentile_w = np.percentile(data_w, 75, axis = 0)

#train the model
RF_r = RandomForestClassifier(n_estimators=300)
RF_r.fit(X_res_r, y_res_r)

RF_w = RandomForestClassifier(n_estimators=300)
RF_w.fit(X_res_w, y_res_w)

#create flask instance
app = Flask(__name__)


#replace -1 with 25th and 75th percentile data of the corresponding features
def process_missing_value(data, wine_type):
    complete_data_lower = []
    complete_data_mid = []
    complete_data_upper = []
    for i in range(len(data)):
        if data[i] != -1:
            complete_data_lower.append(data[i])
            complete_data_mid.append(data[i])
            complete_data_upper.append(data[i])
        else:
            if wine_type == 'red':
                complete_data_lower.append(X_25percentile_r[i])
                complete_data_mid.append(X_50percentile_r[i])
                complete_data_upper.append(X_75percentile_r[i])
            elif wine_type == 'white':
                complete_data_lower.append(X_25percentile_w[i])
                complete_data_mid.append(X_50percentile_w[i])
                complete_data_upper.append(X_75percentile_w[i])
    complete_data_lower = np.array(complete_data_lower)
    complete_data_upper = np.array(complete_data_upper)
    return complete_data_lower, complete_data_mid, complete_data_upper

def translate_rw_data(data):
    result = [data['fixedAcidity'], data['volatileAcidity'], data['citricAcid'], data['freeSulfurDioxide'], data['totalSulfurDioxide'],
        data['sulfates'], data['alcohol']]
    result = np.array(result)
    return result

def translate_ww_data(data):
    result = [data['fixedAcidity'], data['volatileAcidity'], data['residualSugar'], data['chlorides'], 
        data['freeSulfurDioxide'], data['totalSulfurDioxide'], data['alcohol']]
    result = np.array(result)
    return result


#create api
@app.route('/api', methods=['GET', 'POST'])
def predict():
    #get data from request
    data = request.get_json(force=True)
    
    #inspect whether the data comes from red wine or white wine
    wine_type = data['wineType']

    if wine_type == 'red':
        dp_r = translate_rw_data(data)
        #process missing values
        complete_data_lower, _, complete_data_upper = process_missing_value(dp_r, wine_type)
        all_quality_scores = data_r[:, -1]

        #make predicon
        prediction_lower = RF_r.predict(complete_data_lower.reshape(1,-1))
        prediction_upper = RF_r.predict(complete_data_upper.reshape(1,-1))

        #feature importance
        feature_importance = RF_r.feature_importances_
        feature_importance_std = np.std([tree.feature_importances_ for tree in RF_r.estimators_], axis=0)

        #show high quality wine data
        good_wine_data = data_r[all_quality_scores>=8]

    else:
        dp_w = translate_ww_data(data)
        #process missing values
        complete_data_lower, _, complete_data_upper = process_missing_value(dp_w, wine_type)
        all_quality_scores = data_w[:, -1]

        #make predicon 
        prediction_lower = RF_w.predict(complete_data_lower.reshape(1,-1))
        prediction_upper = RF_w.predict(complete_data_upper.reshape(1,-1))

        #feature importance
        feature_importance = RF_w.feature_importances_
        feature_importance_std = np.std([tree.feature_importances_ for tree in RF_w.estimators_], axis=0)

        #show high quality wine data
        good_wine_data = data_w[all_quality_scores>=8]

   
    prediction_bound_1 = prediction_lower[0]
    prediction_bound_2 = prediction_upper[0]

    if prediction_bound_1 == prediction_bound_2:
        prediction = [prediction_bound_1]
    else:
        prediction = [min(prediction_bound_1, prediction_bound_2), max(prediction_bound_1, prediction_bound_2)]
    #show the rank of the qualitu score
    quality_freq_tabel = Counter(all_quality_scores)
    quality_rank = len(all_quality_scores[all_quality_scores<=prediction[0]])/len(all_quality_scores)
    
    #construct dict
    statistics = dict({
        'qualityScore': prediction, #if there is no range, return [prediction(integer)], else return [lower bound(integer), upper bound(integer)]
        'allQualityScoresFreq': quality_freq_tabel,
        'qualityRank': quality_rank, #a dictionary{7: 13; 8: 20....}
        # np.array with dimension of 1: np.array([.....]), 11 values corresponding to 11 features
        'featureImportance': feature_importance.tolist(),
        'featureImportanceStd': feature_importance_std.tolist(),
        #np.array (n by 12 columns, including the quality score) of all the data points with a score >= 8
        'highQualityWineData': good_wine_data.tolist(),
    })
    
    return Response(json.dumps(statistics))


temperary_dataset_r = []
temperary_dataset_w = []
@app.route('/api/getDonatedData', methods=['GET', 'POST'])
def donate_model():
    global X_25percentile_r
    global X_50percentile_r
    global X_75percentile_r
    global X_25percentile_w
    global X_50percentile_w
    global X_75percentile_w
    global temperary_dataset_r
    global temperary_dataset_w

    donated_data = request.get_json(force=True)
    #inspect whether the data comes from red wine or white wine
    wine_type = donated_data['wineType']

    if wine_type == 'red':
        dp_r = translate_rw_data(donated_data)
        #process missing values
        _, complete_data_mid_r, _ = process_missing_value(dp_r, wine_type)
        temperary_dataset_r.append(complete_data_mid_r)
    elif wine_type == 'white':
        dp_w = translate_ww_data(donated_data)
        #process missing values
        _, complete_data_mid_w, _ = process_missing_value(dp_w, wine_type)
        temperary_dataset_w.append(complete_data_mid_w)
        
    if len(temperary_dataset_r) > 10:
        data_r = np.append(data_r, np.array(temperary_dataset_r), axis = 0)
        #Update 25th, 50th and 75th
        X_25percentile_r = np.percentile(data_r, 25, axis = 0)
        X_50percentile_r = np.percentile(data_r, 50, axis = 0)
        X_75percentile_r = np.percentile(data_r, 75, axis = 0)    
        temperary_dataset_r = []
        RF_r.fit(data_r[:, :-1], data_r[:, -1])
    
    if len(temperary_dataset_w) > 10:
        data_w = np.append(data_w, np.array(temperary_dataset_w), axis = 0)
        #Update 25th, 50th and 75th
        X_25percentile_w = np.percentile(data_w, 25, axis = 0)
        X_50percentile_w = np.percentile(data_w, 50, axis = 0)
        X_75percentile_w = np.percentile(data_w, 75, axis = 0)    
        temperary_dataset_w = []
        RF_w.fit(data_w[:, :-1], data_w[:, -1])
    
    return Response("Data successfully submitted. Thank you for contributing!")
