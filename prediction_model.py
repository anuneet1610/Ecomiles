import pandas as pd
data_df=pd.read_csv('/content/processed_air_quality_data_gurugram_repeated_ids.csv')
print(data_df.shape[0])

print(data_df.columns)

print(data_df.info())

"""Data Preparation"""

data_df = data_df.dropna(subset=['AT'])
data_df = data_df.dropna(subset=['RH'])
data_df = data_df.dropna(subset=['PM2_5_cat'])
data_df = data_df.dropna(subset=['PM10_cat'])
data_df = data_df.dropna(subset=['CO2_cat'])

from sklearn.preprocessing import MinMaxScaler

scalers = {}
for feature in ['PM2_5', 'AT', 'RH', 'PM10', 'CO2']:
    scaler = MinMaxScaler()
    data_df[feature + '_scaled'] = scaler.fit_transform(data_df[[feature]])
    scalers[feature] = scaler

"""Splitting the dataset"""

from sklearn.model_selection import train_test_split

X = data_df[['id','weekday', 'weekend', 'evening','night', 'morning', 'afternoon', 'month_sin', 'month_cos','date_bin_early', 'date_bin_late', 'date_bin_mid',
       'month_bin_spring', 'month_bin_summer', 'month_bin_winter']]
y_pm_25 = data_df[['PM2_5_cat']]

X_train_pm_25, X_test_pm_25, y_train_pm_25, y_test_pm_25 = train_test_split(X, y_pm_25, test_size=0.2, shuffle=True, random_state=42)

"""Model"""

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

model_pm_25 = XGBClassifier(
    n_estimators=100,
    learning_rate=0.5,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model_pm_25.fit(X_train_pm_25, y_train_pm_25.values.ravel())

import joblib

joblib.dump(model_pm_25, 'PM25_model.joblib')
