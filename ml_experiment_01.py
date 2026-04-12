#%%------------------------------------------------------------------------
import os
from pathlib import Path
from datetime import date
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

# mlruns folder path
mlruns_path = os.path.join(project_root, "mlruns")
os.makedirs(mlruns_path, exist_ok=True)

# Convert to Windows-friendly file URI
mlruns_uri = f"file:///{mlruns_path.replace(os.sep, '/')}"

mlflow.set_tracking_uri(mlruns_uri)
mlflow.set_experiment(f"power_generation_{main_var_file_name}_{country}_hourly_1_month_ahead")


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


#%%
# -----------------------------------------------------
# Helper: Metrics + plot logging
# -----------------------------------------------------

def log_metrics_and_plot(y_train, y_train_pred, y_test, y_test_pred):
    # --- 1. Compute metrics ---
    metrics = {
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "train_mape": mean_absolute_percentage_error(y_train, y_train_pred),
        "test_mape": mean_absolute_percentage_error(y_test, y_test_pred),
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred)
    }
    mlflow.log_metrics(metrics)

    # --- 2. Prepare data for plotting ---
    df_plot = pd.concat([
        y_train_pred.rename('Fitted-Train').to_frame(),
        pd.concat([y_train, y_test], axis=0),
        y_test_pred.rename('Predicted-Test').to_frame()
    ], axis=1).iloc[-365*24:]

    # --- Helper function to save and log plot ---
    def save_and_log_plot(df, filename, title):
        plt.figure(figsize=(10, 6))  # MLflow-friendly size
        df.plot(ax=plt.gca())
        plt.ylabel('Generation MWh/day', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.title(title, fontsize=16)
        plt.legend(fontsize=10)
        plt.grid()
        plt.tight_layout()

        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_path = os.path.join(tmp_dir, filename)
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
        plt.close()

    # --- 3. Log full dataset plot - 355 days ---
    save_and_log_plot(df_plot, "pred_vs_actual_full-365_days.png", "Train/Test Predictions (Full) - Last 365 Days")

    # --- 4. Log cut dataset plot - 30 days ---
    df_cut = df_plot.iloc[-30*24:]
    save_and_log_plot(df_cut, "pred_vs_actual_test_30_days.png", "Train/Test Predictions - Last 30 Days")

    # --- 5. Log cut dataset plot - 7 days
    df_cut = df_plot.iloc[-7*24:]
    save_and_log_plot(df_cut, "pred_vs_actual_test_7_days.png", "Train/Test Predictions - Last 7 Days")

    # df_cut = df_plot.loc['2025-07-15':'2025-07-30']
    # save_and_log_plot(df_cut, "pred_vs_actual_train_set_1.png", "Train/Test Predictions (From 2025-12-15)")

# -----------------------------------------------------
# Log loss curve for ANN models - with early stopping
# -----------------------------------------------------
def plot_training_history(history_dict, 
                          title="Training vs Validation Loss",
                          filename="loss_curve.png"):


    fig, ax = plt.subplots()

    # Plot train loss
    if 'loss' in history_dict:
        ax.plot(history_dict['loss'], label='train_loss')

    # Plot validation loss
    if 'val_loss' in history_dict:
        ax.plot(history_dict['val_loss'], label='val_loss')

        # Mark best epoch
        best_epoch = np.argmin(history_dict['val_loss'])
        ax.axvline(best_epoch, linestyle='--', label=f'best_epoch={best_epoch}')

    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

    # Log to MLflow
    mlflow.log_figure(fig, filename)

    plt.close(fig)

# -----------------------------------------------------
# Log feature importance for LightGBM models
# -----------------------------------------------------

def log_lgbm_feature_importance(model, feature_names, 
                                top_n=20,
                                log_to_mlflow=True):

    importance = model.feature_importances_

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    # 🔥 keep only top features (prevents overcrowding)
    df_imp = df_imp.head(top_n)

    # 🔥 dynamic height based on number of features
    fig_height = max(6, 0.4 * len(df_imp))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    ax.barh(df_imp["feature"], df_imp["importance"])
    ax.invert_yaxis()
    ax.set_title("LightGBM Feature Importance")
    ax.set_xlabel("Importance")

    # 🔥 improve spacing
    plt.tight_layout()
    plt.subplots_adjust(left=0.35)   # 👈 KEY FIX for long labels

    ax.grid(True)

    if log_to_mlflow:
        mlflow.log_figure(fig, "feature_importance.png")

    plt.close(fig)

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
# Hyperparameters for ANN and LightGBM models
# ---------------------------------------

input_shape_tmp = 1
# Define parameters set
ann_param_grid = dict(#epochs = [10,15, 20,25, 30,40,50],
                      batch_size = [10,20,50,70],
                      model__loss = ['mean_squared_error'],
                      model__optimizer = ['adam','nadam'],
                      model__neurons_nr = [25, 50, 150, 200, 250],
                      model__hidden_layers_nr = [1,2,3,4,5],
                      model__input_shape = [(input_shape_tmp, )],
                      model__output_nodes_nr = [1],                      
                      model__add_batch_norm = [False, True],
                      model__activation_fun = ['relu',  'swish'],
                      model__activation_out = ['linear'],
                      model__dropout = [0.1,  0.2],
                      model__init = ['glorot_uniform','normal'],
                      model__regression_type = [True])

# function to remove 'model__' prefix - manual trainig of ANN with early stopping doesn't use scikeras wrapper, so we need to clean the params
def clean_params(params):
    return {k.replace('model__', ''): v for k, v in params.items()}

# function to take 'batch_size' and 'epochs' parameters - manual trainig of ANN with early stopping doesn't use scikeras wrapper, so we need to clean the params
def split_params(params):
    model_params = {}
    fit_params = {}

    for k, v in params.items():
        if k in ['batch_size', 'epochs']:
            fit_params[k] = v
        else:
            model_params[k] = v

    return model_params, fit_params


LightGBM_param_grid = {
    "n_estimators": [300, 500, 800],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [15, 31, 63],
    "max_depth": [-1, 3, 5, 7],
    "min_child_samples": [10, 20, 50, 100],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0, 1, 5, 10],
    "reg_alpha": [0, 0.1, 1]
}


# ---------------------------------------
# Model configurations
# ---------------------------------------

# start training date list:
start_date = data_analysis.index.min().strftime('%Y-%m-%d')
start_train_date_list = [start_date, '2022-01-01', '2024-01-01', '2025-01-01']


models = [
    {
        "name": "Statsmodels_OLS",
        "type": "statsmodels",
        "params": {
            "formulas": formulas_statsmodels
            ,"start_train_date_list": start_train_date_list
        }
    },
    {
        "name": "Keras_Regressor",
        "type": "keras",
        "params": {
            "variable_sets": variable_sets
           ,"start_train_date_list": start_train_date_list
           ,"n_iter_search": 20
        }
    },
    {
        "name": "LightGBM",
        "type": "lightgbm",
        "params": {
            "variable_sets": variable_sets
           ,"start_train_date_list": start_train_date_list
           ,"n_iter_search": 20
        }
    }
]

# models = [
#     {
#         "name": "LightGBM",
#         "type": "lightgbm",
#         "params": {
#             "variable_sets": variable_sets
#            ,"start_train_date_list": start_train_date_list
#            ,"n_iter_search": 20
#         }
#     }

# ]


# -----------------------------------------------------
# Loop over models
# -----------------------------------------------------
test_days_nr = 30*24  # 30 days * 24 hours/day = 720 hours

today = date.today().isoformat()  # e.g. "2025-09-13"
results = []

for cfg in models:

    for start_train_date in cfg["params"]["start_train_date_list"]:

        data_subset = data_analysis[data_analysis.index >= start_train_date]

        X, y = data_subset.drop(columns=[main_var]), data_subset[[main_var]]
        
        X_train, X_test = X.iloc[:-test_days_nr], X.iloc[-test_days_nr:]
        y_train, y_test = y.iloc[:-test_days_nr], y.iloc[-test_days_nr:]

        #test_set_date = y_test.sort_index().index[0].strftime('%Y-%m-%d')
        test_set_date = y_test.sort_index().index[0]

        # val set
        val_start = test_set_date - pd.DateOffset(months=7)

        #---------------------------------------------------------------------
        ### STAT MODELS ###
        #---------------------------------------------------------------------
        if cfg["type"] == "statsmodels":
            for formula_name, formula in cfg["params"]["formulas"].items():

                #-----------------------------------
                # train model model and make predictions
                #-----------------------------------
                model = smf.ols(formula=formula, data=pd.concat([X_train, y_train], axis=1)).fit()
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                #-----------------------------------
                # log data to mlflow
                #-----------------------------------
                with mlflow.start_run(run_name=f"{cfg['name']}__{formula_name}__{start_train_date}"):
                    
                    mlflow.set_tag("run_date", today) 
                    mlflow.log_params({"model": cfg['name'], "formula": formula})
                    mlflow.log_text(model.summary().as_text(), "ols_summary.txt")
                    log_metrics_and_plot(y_train, y_train_pred, y_test, y_test_pred)
                    # select used variables
                    design_info = model.model.data.design_info
                    used_features = list({
                            factor.name().replace("C(", "").replace(")", "") 
                            for term in design_info.terms 
                            for factor in term.factors
                            if factor.name() != "Intercept"
                        })
                    description = f"""Features: {", ".join(used_features)}
                    Training Data Range: {X_train.index[0]} to {X_train.index[-1]}
                    Test Data Range: {X_test.index[0]} to {X_test.index[-1]}
                    Formula: {formula}"""
                    mlflow.set_tag("mlflow.note.content", description)

        #---------------------------------------------------------------------
        ### KERAS MODELS ###
        #---------------------------------------------------------------------
        if cfg["type"] == "keras":
            for vars_set_ind, vars_set_list in cfg["params"]["variable_sets"].items():

                tf.random.set_seed(42)
                np.random.seed(42)
                random.seed(42)

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
                
                # #---------------------------------------
                # # Define grid search model and search hyperparameters
                # #---------------------------------------  
                # ann_param_grid_tmp =  ann_param_grid.copy()          
                # ann_param_grid_tmp['model__input_shape'] = [(X_train_sld_tmp.shape[1], )]

                # # Define wrapper
                # wraped_ann_model = KerasRegressor(model = create_feed_forward_model)

                # # Search hyperparamers
                # model_ann_random_search = \
                #     RandomizedSearchCV(param_distributions = ann_param_grid_tmp,
                #                        estimator = wraped_ann_model,
                #                        n_iter = cfg["params"]['n_iter_search'],
                #                        cv = TimeSeriesSplit(n_splits=4).split(X_train_sld_tmp),\
                #                        verbose=1,
                #                        n_jobs=-1)

                # model_ann_random_search.fit( X_train_sld_tmp, y_train_sld_tmp )

                # # Print the best parameters
                # print("Best parameters found: ", model_ann_random_search.best_params_)
                
                # #---------------------------------------
                # # Extract the best model and make prediction on test set
                # #---------------------------------------  
                # model_ann_final = model_ann_random_search.best_estimator_

                #---------------------------------------
                # Define grid search model and search hyperparameters - with early stopping
                #---------------------------------------  



                ann_param_grid_tmp =  ann_param_grid.copy()          
                ann_param_grid_tmp['model__input_shape'] = [(X_train_sld_tmp.shape[1], )]

                ann_param_grid_tmp = clean_params(ann_param_grid_tmp)

                param_list = list(ParameterSampler(
                    ann_param_grid_tmp,
                    n_iter=cfg["params"]['n_iter_search'],
                    random_state=42
                ))

                best_score = np.inf
                best_model = None
                best_params = None

                for params in param_list:
                    early_stop = EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )

                    print("Testing:", params)
                    model_params, fit_params = split_params(params)


                    model = create_feed_forward_model(**model_params)

                    history = model.fit(
                        X_train_sld_tmp,
                        y_train_sld_tmp,
                        validation_data=(X_val_sld_tmp, y_val_sld_tmp),
                        epochs=50,   # large → early stopping will cut
                        batch_size=fit_params.get('batch_size', 32),
                        callbacks=[early_stop],
                        verbose=0
                    )

                    val_loss = min(history.history['val_loss'])
                    # Find epoch with lowest val_loss
                    best_epoch = np.argmin(history.history['val_loss']) + 1  # +1 because index starts at 0
                    
                    if val_loss < best_score:
                        best_score = val_loss
                        best_model = model
                        best_params = params
                        best_epoch_for_retrain = best_epoch
                        best_history = history.history

                #---------------------------------------
                # retrain on whole data with best hyperparameters
                #---------------------------------------  

                X_train_sld_tmp_full = pd.concat([X_train_sld_tmp, X_val_sld_tmp])
                y_train_sld_tmp_full = pd.concat([y_train_sld_tmp, y_val_sld_tmp])

                model_params, fit_params = split_params(best_params)

                model_ann_final = create_feed_forward_model(**model_params)

                model_ann_final.fit(
                    X_train_sld_tmp_full,
                    y_train_sld_tmp_full,
                    epochs=best_epoch_for_retrain,
                    batch_size=fit_params.get('batch_size'),
                    verbose=0
                )
                #---------------------------------------
                # Make prediction on test set
                #---------------------------------------  



                # Make prediction on test set
                y_test_pred_ann, X_test_ann = \
                    MakeTSforecast(X_test_sld_tmp,\
                                Model = model_ann_final,\
                                DependentVar = main_var,\
                                Intecept = False,\
                                LagsList = lag_list,\
                                Scaler_y = scaler_y_tmp,\
                                Scaler_X = scaler_X_tmp,\
                                Test_or_Forecast = 'Test')

                data_with_prediction =\
                    MakeANNfinalData(Model = model_ann_final,\
                                    Train_X_Scaled = X_train_sld_tmp_full,\
                                    Val_X_Scaled = None,\
                                    Scaler_y = scaler_y_tmp,\
                                    MainDF = data_subset,\
                                    yhat_Test_DF = y_test_pred_ann,\
                                    yhat_Forecast_DF = None)

                #-----------------------------------
                # log data to mlflow
                #-----------------------------------
                with mlflow.start_run(run_name=f"{cfg['name']}__{vars_set_ind}__{start_train_date}"):
                    
                    mlflow.set_tag("run_date", today)
                    mlflow.log_params(best_params)
                    y_train_pred = data_with_prediction.loc[X_train_sld_tmp_full.index,'Fitted-Train']
                    y_test_pred = y_test_pred_ann.iloc[:, 0]
                    log_metrics_and_plot(y_train, y_train_pred, y_test, y_test_pred)
                    plot_training_history(best_history)

                    used_features = list(X_train_sld_tmp.columns)
                    description = f"""Features: {", ".join(used_features)}
                    Training Data Range: {X_train_sld_tmp.index[0]} to {X_train_sld_tmp.index[-1]}
                    Validating Data Range: {X_val_sld_tmp.index[0]} to {X_val_sld_tmp.index[-1]}
                    Test Data Range: {X_test_sld_tmp.index[0]} to {X_test_sld_tmp.index[-1]}
                    Hyperparameter Search Space:
                    {ann_param_grid_tmp}

                    The best hyperparameters found: {best_params}
                    """
                    mlflow.set_tag("mlflow.note.content", description)

        #---------------------------------------------------------------------
        ### LIGHTGBM MODELS ###
        #---------------------------------------------------------------------        
        if cfg["type"] == "lightgbm":
            for vars_set_ind, vars_set_list in cfg["params"]["variable_sets"].items():
                

                #---------------------------------------
                # Prepare Data For LIGHTGBM
                #---------------------------------------  
                X_train_tmp = X_train.copy().loc[:,vars_set_list]
                X_test_tmp = X_test.copy().loc[:,vars_set_list]            
                
                lgb_reg = LGBMRegressor(random_state = 42
                                        ,linear_tree=True)

                random_search_LightGBM = RandomizedSearchCV(
                    estimator= lgb_reg,
                    param_distributions= LightGBM_param_grid,
                    n_iter= cfg["params"]['n_iter_search'],
                    cv= TimeSeriesSplit(n_splits=5),
                    verbose= 1,
                    random_state= 42,
                    n_jobs= 1
                )

                random_search_LightGBM.fit(X_train_tmp, y_train)

                #---------------------------------------
                # Extract the best model
                #--------------------------------------- 
                best_params = random_search_LightGBM.best_params_
                best_model = random_search_LightGBM.best_estimator_

                #-----------------------------------
                # log data to mlflow
                #-----------------------------------
                with mlflow.start_run(run_name=f"LightGBM__{vars_set_ind}__{start_train_date}"):

                    mlflow.set_tag("run_date", today)
                    # log parametry najlepszego modelu
                    mlflow.log_params(best_params)
                    # ewaluacja
                    y_train_pred = best_model.predict(X_train_tmp)
                    y_test_pred = best_model.predict(X_test_tmp)
                    y_train_pred = pd.Series(y_train_pred.flatten(), index=y_train.index, name='Fitted-Train')
                    y_test_pred  = pd.Series(y_test_pred.flatten(),  index=y_test.index,  name='Predicted-Test')

                    log_metrics_and_plot(y_train, y_train_pred, y_test, y_test_pred)
                    log_lgbm_feature_importance(
                        best_model,
                        X_train_tmp.columns
                    )
                    used_features = list(X_train_tmp.columns)
                    description = f"""Features: {", ".join(used_features)}
                    Training Data Range: {X_train_tmp.index[0]} to {X_train_tmp.index[-1]}
                    Test Data Range: {X_test_tmp.index[0]} to {X_test_tmp.index[-1]}
                    Hyperparameter Search Space:
                    {LightGBM_param_grid}

                    The best hyperparameters found: {best_params}
                    """
                    mlflow.set_tag("mlflow.note.content", description)

print("✅ All models logged to MLflow. Run `mlflow ui` to inspect results.")
# %%
