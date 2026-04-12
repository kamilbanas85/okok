#%%------------------------------------------------------------------------

import os
from pathlib import Path
from datetime import date
from pyexpat import model
import random

import requests

import mlflow
import mlflow.keras
import mlflow.lightgbm
import mlflow.sklearn

from lightgbm import LGBMRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

import statsmodels.formula.api as smf

import tempfile

from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import (RandomizedSearchCV, GridSearchCV,
                                     TimeSeriesSplit, ParameterSampler)

import tensorflow as tf
import random
from tensorflow.keras.callbacks import EarlyStopping

from darts import TimeSeries
from darts.models import RNNModel

#%%------------------------------------------------------------------------
# read functions from GitHub
#--------------------------------------------------------------------------

# Function to extract code from GitHub:
def GetGitHubCode(GitUrl):

    response = requests.get(GitUrl) #get data from json file located at specified URL 

    if response.status_code == requests.codes.ok:
        contentOfUrl = response.content
        exec(contentOfUrl, globals() )
    else:
        print('Content was not found.')

# Download functions from GitHub:
GitUrl__Prepare_Data_For_Regression = 'https://raw.githubusercontent.com/kamilbanas85/Phyton_usefull_functions/main/Prepare_Data_For_Regression.py'
GetGitHubCode(GitUrl__Prepare_Data_For_Regression)

GitUrl__ANN_Keras_functions = 'https://raw.githubusercontent.com/kamilbanas85/Phyton_usefull_functions/main/ANN_Keras_functions.py'
GetGitHubCode(GitUrl__ANN_Keras_functions)

GitUrl__Make_TS_Regression = 'https://raw.githubusercontent.com/kamilbanas85/Phyton_usefull_functions/main/Make_TS_Regression.py'
GetGitHubCode(GitUrl__Make_TS_Regression)

#%%------------------------------------------------------------------------
# Set up variables
#--------------------------------------------------------------------------

country = 'PL'
main_var_file_name = 'solar'
main_var = "solar_generation"

#%%------------------------------------------------------------------------
# set up project direcory
#--------------------------------------------------------------------------

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # mlruns folder path
# mlruns_path = os.path.join(project_root, "mlruns")
# os.makedirs(mlruns_path, exist_ok=True)

# # Convert to Windows-friendly file URI
# mlruns_uri = f"file:///{mlruns_path.replace(os.sep, '/')}"

# mlflow.set_tracking_uri(mlruns_uri)
# mlflow.set_experiment(f"power_generation_{main_var_file_name}_{country}_hourly_1_month_ahead")


#%%------------------------------------------------------------------------
# read data
#--------------------------------------------------------------------------

 # Path to data folder
data_dir = os.path.join(project_root, "data")

# load CSV
file_name = f'{main_var_file_name}_data_{country}.csv'
data_file_path = os.path.join(data_dir, file_name)
data_analysis = pd.read_csv(data_file_path)

data_analysis = data_analysis\
    .assign(datetimeCET = lambda x: pd.to_datetime(x['datetimeCET']))\
    .set_index('datetimeCET')


#%%---------------------------------------------------------------
# Veriables and hyperparameters configuration
# ----------------------------------------------------------------

# ---------------------------------------
# Variable sets for keras and LightGBM models
# ---------------------------------------

#data_analysis.columns

variable_sets = {
     "set1": ["shortwave_radiation_mean", "capacity_solar", ]
    ,"set2": ["shortwave_radiation_mean", "capacity_solar", 'hour', 'month']
    ,"set3": ["shortwave_radiation_mean", "capacity_solar", 'cloud_cover_mean', 
              'temperature_2m_mean', 'hour', 'month', 'workday', 'is_dst', 'precipitation_mean',
              'snow_depth_mean']
    ,"set4": ["shortwave_radiation_mean", "capacity_solar", 'cloud_cover_mean',
              'temperature_2m_mean', 'hour', 'month', 'workday', 'is_dst', 'precipitation_mean',
              'snow_depth_mean', 
              'direct_radiation_mean', 'diffuse_radiation_mean', 'direct_normal_irradiance_mean']
    ,"set5": ["shortwave_radiation_mean", "capacity_solar", 'cloud_cover_mean',
              'temperature_2m_mean', 'hour', 'month', 'workday', 'is_dst', 'precipitation_mean',
              'snow_depth_mean', 
              'direct_radiation_mean', 'diffuse_radiation_mean', 'direct_normal_irradiance_mean']
    #,"set5": ["shortwave_radiation_wmean_solar", "capacity_solar", 'hour', 'month']
    #,"set6": ["shortwave_radiation_mean_sel1_solar", "capacity_solar", 'hour', 'month']
    ,"set6": ["shortwave_radiation_wmean_solar", "capacity_solar", 'cloud_cover_wmean_solar',
              'temperature_2m_mean', 'hour', 'month', 'workday', 'is_dst', 'precipitation_wmean_solar',
              'snow_depth_wmean_solar', 
              'direct_radiation_wmean_solar', 'diffuse_radiation_wmean_solar', 'direct_normal_irradiance_wmean_solar']
    ,"set7": ["shortwave_radiation_mean_sel1_solar", "capacity_solar", 'cloud_cover_mean_sel1_solar',
              'temperature_2m_mean', 'hour', 'month', 'workday', 'is_dst', 'precipitation_mean_sel1_solar',
              'snow_depth_mean_sel1_solar', 
              'direct_radiation_mean_sel1_solar', 'diffuse_radiation_mean_sel1_solar', 'direct_normal_irradiance_mean_sel1_solar']
              }

#!!!!! dodac jeszcze dst  !!!!!!!!!!!!!!!!!!!! <- dlatego ze to jest powiazane ze zuzeyciem residenitail
# plus zanlesc ograniczenaia generacji systemowe
# 

dummies_sets = {
    "set1": []
    ,"set2": ['hour', 'month']
    ,"set3": ['hour', 'month']
    ,"set4": ['hour', 'month']
    ,"set5": []
    ,"set6": ['hour', 'month']
    ,"set7": ['hour', 'month']
}

# ----------------------------------------
# Formulas for Statsmodels
# ----------------------------------------
formulas_statsmodels = {
    "formula01": f"{main_var} ~ shortwave_radiation_mean * capacity_solar",
    "formula02": f"{main_var} ~ shortwave_radiation_wmean_solar * capacity_solar",
    "formula03": f"{main_var} ~ shortwave_radiation_mean_sel1_solar * capacity_solar",

    "formula04": f"{main_var} ~ shortwave_radiation_mean * capacity_solar * temperature_2m_mean",
    "formula05": f"{main_var} ~ shortwave_radiation_mean * capacity_solar * C(month)",
    "formula06": f"{main_var} ~ shortwave_radiation_mean * capacity_solar * (C(month) + C(hour))",
    "formula07": f"{main_var} ~ shortwave_radiation_mean : capacity_solar : (C(month) + C(hour))",
    "formula08": f"{main_var} ~ shortwave_radiation_mean : capacity_solar : (C(month) + C(hour) + C(workday) + C(is_dst))",
    "formula09": f"{main_var} ~ (shortwave_radiation_mean + direct_radiation_mean + diffuse_radiation_mean + direct_normal_irradiance_mean) : capacity_solar : (C(month) + C(hour))",
    "formula10": f"{main_var} ~ shortwave_radiation_wmean_solar : capacity_solar : (C(month) + C(hour))",
    "formula11": f"{main_var} ~ shortwave_radiation_mean_sel1_solar : capacity_solar : (C(month) + C(hour))"
}

# ---------------------------------------
# Model configurations
# ---------------------------------------

# start training date list:
start_date = data_analysis.index.min().strftime('%Y-%m-%d')
start_train_date_list = [start_date, '2022-01-01']

# -----------------------------------------------------
# Loop over models
# -----------------------------------------------------
test_days_nr = 7*24  # 7 days * 24 hours/day = 168 hours

today = date.today().isoformat()  # e.g. "2025-09-13"

start_train_date = start_train_date_list[0]  # or start_train_date_list[1] for the second option


data_subset = data_analysis[data_analysis.index >= start_train_date]

X, y = data_subset.drop(columns=[main_var]), data_subset[[main_var]]

X_train, X_test = X.iloc[:-test_days_nr], X.iloc[-test_days_nr:]
y_train, y_test = y.iloc[:-test_days_nr], y.iloc[-test_days_nr:]

#test_set_date = y_test.sort_index().index[0].strftime('%Y-%m-%d')
test_set_date = y_test.sort_index().index[0]

# val set
val_start = test_set_date - pd.DateOffset(months=7)


###3#
vars_set_ind = 'set4'
vars_set_list = variable_sets[vars_set_ind]
#####

# take dummies for specified variable set
dummy_for_columns = dummies_sets[vars_set_ind]
if len(dummy_for_columns) == 0:
    dummy_for_columns = None

lag_list = None

#---------------------------------------
# Prepare Data For ANN
#---------------------------------------                
X, y =  DevideOnXandY_CreateDummies(data_subset, 
                                    DependentVar = main_var,
                                    IndependentVar = vars_set_list,
                                    DummyForCol = dummy_for_columns,
                                    drop_first = False)

cols_to_select = (
        (set(vars_set_list) - set(dummy_for_columns or []))
    | {
        c for c in X.columns
        if any(c.startswith(f"{d}__") for d in (dummy_for_columns or []))
    }
)
cols_to_select = list(cols_to_select)

X_train_sld_tmp, y_train_sld_tmp,\
X_val_sld_tmp, y_val_sld_tmp,\
X_test_sld_tmp, y_test_sld_tmp,\
scaler_X_tmp, scaler_y_tmp = \
            PrepareDataForRegression(X.loc[:,cols_to_select], y,
                                    TestSplitInd = test_set_date.strftime('%Y-%m-%d %H:%M'),
                                    ValSplitInd = val_start.strftime('%Y-%m-%d %H:%M'), 
                                    ScalerType = 'MinMax',
                                    ScalerRange = (0,1),                             
                                    BatchSize = None,
                                    WindowLength = 1)


#%%------------------------------------------------------------------------
# data for darts series
#------------------------------------------------------------------------

X_train_sld_tmp_darts = TimeSeries.from_dataframe(X_train_sld_tmp)
X_val_sld_tmp_darts = TimeSeries.from_dataframe(X_val_sld_tmp)
X_test_sld_tmp_darts = TimeSeries.from_dataframe(X_test_sld_tmp)

y_train_sld_tmp_darts = TimeSeries.from_dataframe(y_train_sld_tmp)
y_val_sld_tmp_darts = TimeSeries.from_dataframe(y_val_sld_tmp)
y_test_sld_tmp_darts = TimeSeries.from_dataframe(y_test_sld_tmp)

X_sld_tmp_darts = \
    TimeSeries.from_dataframe(\
        pd.concat([X_train_sld_tmp, X_val_sld_tmp, X_test_sld_tmp], axis = 0))

# comabain to train on whole data (train + val) and predict test set
y_train_and_val_sld_tmp_darts = \
    TimeSeries.from_dataframe(\
        pd.concat([y_train_sld_tmp, y_val_sld_tmp], axis = 0))

X_train_and_val_sld_tmp_darts = \
    TimeSeries.from_dataframe(\
        pd.concat([X_train_sld_tmp, X_val_sld_tmp], axis = 0))


#####################################################``

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import Callback


# Early stopping callback
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=5,          # stop after 10 epochs without improvement
    min_delta=1e-4,       # minimum improvement threshold
    mode="min",
    verbose=True
)

class StoreLossCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        train_loss = metrics.get("train_loss")
        val_loss   = metrics.get("val_loss")

        if train_loss is not None:
            self.train_losses.append(train_loss.item())

        if val_loss is not None:
            self.val_losses.append(val_loss.item())
loss_cb = StoreLossCallback()

#-------------------------------------------------------
# define model
#-------------------------------------------------------

param_lstm = {
    'model': 'LSTM',              # <-- important change
    'n_rnn_layers': 2,
    'hidden_dim': 40,
    'n_epochs': 80,
    'batch_size': 70,
    'input_chunk_length': 72,  # use past 72 hours to predict the next hour
    #'output_chunk_length': 1,     # predict 1 hour ahead
    'training_length': 144
}

# Model with early stopping
lstm_model = RNNModel(
    **param_lstm,
    force_reset=True,
    random_state=42,
    pl_trainer_kwargs={
        "callbacks": [early_stop_callback, loss_cb],
    #    "logger": csv_logger
    }
)

# print(lstm_model.supports_past_covariates)
# print(lstm_model.supports_future_covariates)

lstm_model.fit(
    series=y_train_sld_tmp_darts,
    future_covariates=X_train_sld_tmp_darts,
    val_series= y_val_sld_tmp_darts,
    val_future_covariates=X_val_sld_tmp_darts,
    verbose=True
)

# plot rtain-val loss

plt.plot(loss_cb.train_losses, label="train_loss")
plt.plot(loss_cb.val_losses, label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()



#---------------------------------------
# retrain on whole data (train + val) and predict test set
#----------------------------------------

# take the number of epochs from the model trained with early stopping
param_lstm_full = param_lstm.copy()
param_lstm_full['n_epochs'] = lstm_model.epochs_trained

# define full model
lstm_model_full = RNNModel(
    **param_lstm_full,
    force_reset=True,   # important!
    random_state=42
)

# train on whole data (train + val)
lstm_model_full.fit(
    series=y_train_and_val_sld_tmp_darts,
    future_covariates=X_sld_tmp_darts,
    verbose=True
)

# predict test set
yhat_test_sld = lstm_model_full.predict(
    n=len(X_test_sld_tmp)
    #, future_covariates=X_sld_tmp_darts
)


# inverse transform predictions and create DataFrame
yhat_test = scaler_y_tmp.inverse_transform(yhat_test_sld.values())
yhat_test = pd.DataFrame(yhat_test,\
                                index = X_test_sld_tmp.index, columns = ['Predicted-Test'])



# Plot Fitted Data
pd.concat([y_test
           ,yhat_test], axis=1)\
        .iloc[:24*7].plot()
        
plt.ylabel('Solar Generation MWh/h', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Solar generation - test set', fontsize=20)
plt.legend(fontsize=14)
plt.grid()
plt.rcParams['figure.figsize'] = [15, 10]
plt.show()