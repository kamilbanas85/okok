# Databricks notebook source
import datetime
import json
import os

# COMMAND ----------

# read 'BRANCH_NAME'

with open("../../../../config_SUPPLY_DEMAND_MODELS.json", "r") as json_file:
    config_data = json.load(json_file)

# Assign variables
BRANCH_NAME = config_data.get("BRANCH_NAME")
SCHEMA = config_data.get("SCHEMA")

# COMMAND ----------

# read 'end_forcast'

file_path = f"/dbfs/tmp/{SCHEMA}/global_vars.json"

# Read JSON file
with open(file_path, "r") as f:
    global_vars = json.load(f)

end_forecast = global_vars['end_forecast']

# convert to datetime object
end_forecast = datetime.datetime.strptime(end_forecast, '%Y-%m-%d').date()

# COMMAND ----------

main_var = 'Gas_power'

var_read_write = 'power_gas_consumption'

# COMMAND ----------

if 'dev' in BRANCH_NAME:
  schema_experiment= 'models_dev'
  schema_model_name = 'dev'
elif 'test' in BRANCH_NAME:
  schema_experiment= 'models_test'
  schema_model_name = 'test'
elif 'main' in BRANCH_NAME:
  schema_experiment= 'models_prod'
  schema_model_name = 'prod'

# COMMAND ----------

major_version = 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## import LIBRARIES and FUNCTIONS

# COMMAND ----------

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

import numpy as np
import pandas as pd

import json

import mlflow
import matplotlib.pyplot as plt

# COMMAND ----------

from scikeras.wrappers import KerasRegressor

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import uniform, randint, loguniform

# COMMAND ----------

# MAGIC %run /FUNCTIONS_usefull/Prepare_Data_For_Regression

# COMMAND ----------

# MAGIC %run /FUNCTIONS_usefull/ANN_Keras_functions

# COMMAND ----------

# MAGIC %run /FUNCTIONS_usefull/Make_TS_Regression

# COMMAND ----------

results_json = dbutils.notebook.run(f"./{var_read_write}_v{major_version}_prepare_features", 90)

# COMMAND ----------

# Parse JSON
results = json.loads(results_json)

# Convert back to DataFrames
analysis_data = process_json_output(results["analysis_data"])
analysis_data = analysis_data.sort_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ## FIT MODEL

# COMMAND ----------

experiment_name = f"/Workspace/MODELS_experiments/{schema_experiment}/experiment_{var_read_write}_v{major_version}_{schema_model_name}"

mlflow.set_experiment(experiment_name)

# COMMAND ----------

import tempfile
import os

def log_metrics_and_plot(y_train, y_train_pred, y_test, y_test_pred):
    # --- 1. Compute metrics ---
    metrics = {
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "test_mape": mean_absolute_percentage_error(y_test, y_test_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred)
    }
    mlflow.log_metrics(metrics)

    # --- Helper function to save and log plot ---
    def save_and_log_plot(df, filename, title):
        plt.figure(figsize=(7, 5))
        df.plot(ax=plt.gca())
        plt.ylabel('load GWh/day', fontsize=10)
        plt.xlabel('Date', fontsize=10)
        plt.title(title, fontsize=12)
        plt.legend(fontsize=8)
        plt.grid()
        plt.tight_layout()

        with tempfile.TemporaryDirectory() as tmp_dir:
            plot_path = os.path.join(tmp_dir, filename)
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)

    # --- 2. Convert predictions to Series ---
    # dopasuj index, żeby się zgadzał z y_train / y_test
    y_train_pred = pd.Series(y_train_pred, index=y_train.index, name="Fitted-Train")
    y_test_pred = pd.Series(y_test_pred, index=y_test.index, name="Predicted-Test")

    # --- 3. Log full dataset plot ---
    df_plot = pd.concat([
        y_train_pred,
        pd.concat([y_train, y_test], axis=0),
        y_test_pred
    ], axis=1)

    save_and_log_plot(df_plot, "pred_vs_actual_full.png", "Train/Test Predictions (Full)")

    # --- 4. Log cut dataset plot (ostatnie 2 lata / np. 730 dni) ---
    df_cut = df_plot.iloc[-2*365:]
    save_and_log_plot(df_cut, "pred_vs_actual_test_set.png", "Train/Test Predictions - short")


# COMMAND ----------

import pickle
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from typing import List, Union

# -----------------------------------------------------
# PyFunc wrapper
# -----------------------------------------------------
class StatsmodelsOLSWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle
        with open(context.artifacts["model_path"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        return self.model.predict(model_input)

# COMMAND ----------

import tensorflow as tf
from tensorflow import keras

class KerasWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = tf.keras.models.load_model(context.artifacts["model_path"])

    def predict(self, context, model_input):
        return self.model.predict(model_input).flatten()

# COMMAND ----------

import xgboost as xgb

class XGBoostWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["model_path"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        return self.model.predict(model_input).flatten()

# COMMAND ----------

import lightgbm as lgb

class LightGBMWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["model_path"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        return self.model.predict(model_input).flatten()

# COMMAND ----------

analysis_data.head()

# COMMAND ----------

# -----------------------------------------------------
# Hyperparametrs sets for Keras and LXBoost
# -----------------------------------------------------

temp_input_shape = 1

# Define parameters set
ann_param_grid = dict(epochs = [10, 15, 20, 25, 30, 40],
                      batch_size = [10,20,50,70],
                      model__loss = ['mean_squared_error'],
                      model__optimizer = ['adam','nadam'],
                      model__neurons_nr = [25, 50, 100, 150, 200],
                      model__hidden_layers_nr = [1,2,3,4,5],
                      model__input_shape = [(temp_input_shape, )],
                      model__output_nodes_nr = [1],                      
                      model__add_batch_norm = [False, True],
                      model__activation_fun = ['relu', 'swish'],
                      model__activation_out = ['linear'],
                      model__dropout = [0.1,  0.2],
                      model__init = ['glorot_uniform','normal'],
                      model__regression_type = [True])

LightGBM_param_grid = {
    "n_estimators": [300, 500, 800, 1000],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [15,31,63],
    "max_depth": [-1,3,5,7],
    "min_child_samples": [10, 20, 50, 100],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0, 1, 5, 10],
    "reg_alpha": [0, 0.1, 1]
}

XGBoost_param_grid = {
                "colsample_bytree": uniform(0.2, 0.7),
                "learning_rate": loguniform(0.001, 0.1),
                "max_depth": randint(1, 4),
                "n_estimators": randint(500, 5000),
                "subsample": uniform(0.3, 0.7),
                "gamma": uniform(0, 0.4),
                "alpha": loguniform(0.001, 10),
                "lambda": loguniform(0.001, 10),
                "min_child_weight": randint(1, 10),
                "scale_pos_weight": uniform(0.5, 2),
                "max_delta_step": randint(0, 10),
                "tree_method": ["auto", "hist"],
                "grow_policy": ["depthwise", "lossguide"]
            }



# COMMAND ----------

# -----------------------------------------------------
# Model configurations
# -----------------------------------------------------

start_train_date_list = ['2016-01', '2018-01','2023-03']

variable_sets = {
    "set1": ['Temp_avg'
            ,'wind_speed_100m'
            ,'shortwave_radiation_sum'
            ,'Gas_capacity_avl', 'Lignite_capacity_avl', 'Nuclear_capacity_avl', 'Solar_capacity_avl', 'Wind_capacity_avl'
            ,'switching_gas_coal_bin'
            ,'Month'
            ,'Ukraine_war'
            ,'Covid'
            ,'WorkDay']
}


# -----------------------------------------------------
# Model configurations
# -----------------------------------------------------
models = [
    {
        "name": "Statsmodels_OLS",
        "type": "statsmodels",
        "params": {
            "formulas": [
                f"{main_var} ~ Temp_avg	+ wind_speed_100m + shortwave_radiation_sum + Gas_capacity_avl + Hard_Coal_capacity_avl +	Lignite_capacity_avl + Nuclear_capacity_avl + Solar_capacity_avl + Wind_capacity_avl + C(Week) + C(switching_gas_coal_bin) + C(Ukraine_war) + C(Covid) + C(WorkDay)",
                f"{main_var} ~ Temp_avg + Temp_avg:C(Week) + wind_speed_100m + wind_speed_100m:C(Week) + shortwave_radiation_sum + shortwave_radiation_sum:C(Week) + Gas_capacity_avl + Hard_Coal_capacity_avl +	Lignite_capacity_avl + Nuclear_capacity_avl + Solar_capacity_avl + Wind_capacity_avl + C(Week) + C(switching_gas_coal_bin) + C(Ukraine_war) + C(Covid) + C(WorkDay)",
                ]
            ,"start_train_date":start_train_date_list
        }
    },
    {
        "name": "LightGBM",
        "type": "lightgbm",
        "params": {
             "variable_sets": variable_sets
            ,"start_train_date":start_train_date_list
            ,"n_iter_search": 30
        }
    },
    {
        "name": "Keras_Regressor",
        "type": "keras",
        "params": {
             "variable_sets": variable_sets
            ,"start_train_date":start_train_date_list
            ,"n_iter_search": 30
        }
    }
    ,{
       "name": "XGBoost",
       "type": "xgboost",
        "params": {
             "variable_sets": variable_sets
            ,"start_train_date":start_train_date_list
            ,"n_iter_search": 30
        }
    }
]

# COMMAND ----------

# test_days_nr = 182
# test_set_date = analysis_data.sort_index().iloc[-test_days_nr:].index[0].strftime('%Y-%m-%d')

# # Select Main Data
# main_var = 'Gas_power'

# independent_vars = ['Temp_avg'
#                     ,'wind_speed_100m'
#                     ,'shortwave_radiation_sum'
#                     ,'Gas_capacity_avl'
#                     ,'Lignite_capacity_avl'
#                     ,'Nuclear_capacity_avl'
#                     ,'Solar_capacity_avl'
#                     ,'Wind_capacity_avl'
#                     ,'switching_gas_coal_bin'
#                     ,'Month'
#                     ,'Ukraine_war'
#                     ,'Covid'
#                     ,'WorkDay'
#                    ]

# dummy_for_columns = ['Month', 'switching_gas_coal_bin']
# lag_list = None


# # Prepare Data For ANN
# X, y =  DevideOnXandY_CreateDummies(analysis_data, 
#                                     DependentVar = main_var,
#                                     IndependentVar = independent_vars,
#                                     DummyForCol = dummy_for_columns,
#                                     drop_first = False)



# X_train_sld, y_train_sld,\
# X_test_sld, y_test_sld,\
# scaler_X, scaler_y = \
#             PrepareDataForRegression(X, y, 
#                                      TestSplitInd = test_set_date,
#                                      ValSplitInd = None,     
#                                      ScalerType = 'MinMax',
#                                      ScalerRange = (0,1),                             
#                                      BatchSize = None,
#                                      WindowLength = 1)
            


# X_train, y_train,\
# X_test, y_test = \
#             PrepareDataForRegression(X, y, 
#                                      TestSplitInd = test_set_date,
#                                      ValSplitInd = None,     
#                                      ScalerType = None,
#                                      ScalerRange = None,                             
#                                      BatchSize = None,
#                                      WindowLength = 1)

# COMMAND ----------

import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb

# -----------------------------------------------------
# Loop over models
# -----------------------------------------------------
test_days_nr = 290

today = datetime.date.today().isoformat()  # e.g. "2025-09-13"
results = []

for cfg in models:

    for start_train_date in cfg["params"]["start_train_date"]:

        data_subset = analysis_data[analysis_data.index >= start_train_date]

        X, y = data_subset.drop(columns=[main_var]), data_subset[[main_var]]
        
        X_train, X_test = X.iloc[:-test_days_nr], X.iloc[-test_days_nr:]
        y_train, y_test = y.iloc[:-test_days_nr], y.iloc[-test_days_nr:]

        test_set_date = y_test.sort_index().index[0].strftime('%Y-%m-%d')

############################################### STAT MODELS ##########################################################
        if cfg["type"] == "statsmodels":
            for formula in cfg["params"]["formulas"]:

                with mlflow.start_run(run_name=f"{cfg['name']}"):
                    
                    mlflow.set_tag("run_date", today) 

                    model = smf.ols(formula=formula, data=pd.concat([X_train, y_train], axis=1)).fit()
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
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
                    description = f"""Cechy: {", ".join(used_features)}
                    Zakres danych treningowych: {X_train.index[0]} do {X_test.index[-1]}
                    Formuła: {formula}"""
                    
                    mlflow.set_tag("mlflow.note.content", description)

                    with tempfile.TemporaryDirectory() as tmpdir:
                        model_path = os.path.join(tmpdir, "ols_model.pkl")

                        # Save statsmodels object
                        #model_path = "ols_model.pkl"
                        with open(model_path, "wb") as f:
                            pickle.dump(model, f)

                        # Create signature
                        signature = infer_signature(X_train[used_features].iloc[:1], model.predict(X_train.iloc[:1]))

                        mlflow.pyfunc.log_model(
                            artifact_path="statsmodels_model",
                            python_model=StatsmodelsOLSWrapper(),
                            artifacts={"model_path": model_path},
                            input_example=X_test.iloc[[0]],
                            signature=signature
                        )

                    # Track results, Compute metrics - local
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    
                    results.append({
                            "run_id": mlflow.active_run().info.run_id,
                            "run_name":f"{cfg['name']}__{formula}",
                            "test_mae": test_mae,
                            "model_type": cfg["type"],
                            "model_name": "statsmodels_model"
                        })
                    
    ############################################### KERAS MODELS ##########################################################
        if cfg["type"] == "keras":
            for vars_set_ind, vars_set_list in cfg["params"]["variable_sets"].items():

                dummy_for_columns = ['Month', 'switching_gas_coal_bin']
                lag_list = None

                # Prepare Data For ANN
                X, y =  DevideOnXandY_CreateDummies(data_subset, 
                                                    DependentVar = main_var,
                                                    IndependentVar = vars_set_list,
                                                    DummyForCol = dummy_for_columns,
                                                    drop_first = False)

                cols_to_select = (
                      (set(vars_set_list) - set(dummy_for_columns))
                    | {
                        c for c in X.columns
                        if any(c.startswith(f"{d}__") for d in dummy_for_columns)
                    }
                )
                cols_to_select = list(cols_to_select)

                X_train_sld_tmp, y_train_sld_tmp,\
                X_test_sld_tmp, y_test_sld_tmp,\
                scaler_X_tmp, scaler_y_tmp = \
                            PrepareDataForRegression(X.loc[:,cols_to_select], y, 
                                                    TestSplitInd = test_set_date,
                                                    ValSplitInd = None,     
                                                    ScalerType = 'MinMax',
                                                    ScalerRange = (0,1),                             
                                                    BatchSize = None,
                                                    WindowLength = 1)

                ann_param_grid_tmp =  ann_param_grid.copy()          
                ann_param_grid_tmp['model__input_shape'] = [(X_train_sld_tmp.shape[1], )]

                # Define wrapper
                wraped_ann_model = KerasRegressor(model = create_feed_forward_model)

                # Search hyperparamers
                model_ann_random_search = \
                    RandomizedSearchCV(param_distributions = ann_param_grid_tmp,
                                       estimator = wraped_ann_model,
                                       n_iter = cfg["params"]['n_iter_search'],
                                       cv = TimeSeriesSplit(n_splits=4).split(X_train_sld_tmp),\
                                       verbose=1,
                                       n_jobs=-1)

                model_ann_random_search.fit( X_train_sld_tmp, y_train_sld_tmp )

                # Print the best parameters
                print("Best parameters found: ", model_ann_random_search.best_params_)
                
                # Extract the best model
                model_ann_final = model_ann_random_search.best_estimator_

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
                                    Train_X_Scaled = X_train_sld_tmp,\
                                    Val_X_Scaled = None,\
                                    Scaler_y = scaler_y_tmp,\
                                    MainDF = data_subset,\
                                    yhat_Test_DF = y_test_pred_ann,\
                                    yhat_Forecast_DF = None)
                
                with mlflow.start_run(run_name=f"{cfg['name']}__{vars_set_ind}__{start_train_date}"):
                    mlflow.log_params(model_ann_random_search.best_params_)
                    y_train_pred = data_with_prediction.loc[X_train_sld_tmp.index,'Fitted-Train']
                    y_test_pred = y_test_pred_ann.iloc[:, 0]
                    log_metrics_and_plot(y_train, y_train_pred, y_test, y_test_pred)

                    mlflow.set_tag("run_date", today)
               
                    used_features = list(X_train_sld_tmp.columns)
                    description = f"""Cechy: {", ".join(used_features)}
                    Zakres danych treningowych: {X_train_sld_tmp.index[0]} do {X_train_sld_tmp.index[-1]}
                    Przeszukiwana przestrzeń parametrów:
                    {cfg["params"]["variable_sets"]}"""

                    mlflow.set_tag("mlflow.note.content", description)

                    # zapis modelu
                    with tempfile.TemporaryDirectory() as tmpdir:
                        model_path = os.path.join(tmpdir, "keras_best_model.keras")
                        model_ann_final.model_.save(model_path)

                        signature = infer_signature(X_train_sld_tmp[:1], model_ann_final.predict(X_train_sld_tmp[:1]))

                        mlflow.pyfunc.log_model(
                            artifact_path="keras_best_model",
                            python_model=KerasWrapper(),
                            artifacts={"model_path": model_path},
                            input_example=X_train_sld_tmp[:1],
                            signature=signature
                        )

                        results.append({
                                "run_id": mlflow.active_run().info.run_id,
                                "run_name":f"{cfg['name']}",
                                "test_mae": test_mae,
                                "model_type": cfg["type"],
                                "model_name": "keras_best_model"
                            })
                    
    ############################################### XGBOOST MODELS ##########################################################
        if cfg["type"] == "xgboost":
            for vars_set_ind, vars_set_list in cfg["params"]["variable_sets"].items():
                
                X_train_tmp = X_train.copy().loc[:,vars_set_list]
                X_test_tmp = X_test.copy().loc[:,vars_set_list]

                xgb_reg = xgb.XGBRegressor()

                with mlflow.start_run(run_name=f"XGBoost_RandomSearch__{vars_set_ind}__{start_train_date}"):

                    mlflow.set_tag("run_date", today)

                    used_features = list(X_train_tmp.columns)

                    # 📝 Opis parametrów
                    description = f"""Cechy: {", ".join(used_features)}
                    Zakres danych treningowych: {X_train_tmp.index[0]} do {X_train_tmp.index[-1]}
                    Przeszukiwana przestrzeń parametrów:
                    {cfg["params"]["variable_sets"]}
                    Liczba iteracji: {cfg["params"]['n_iter_search']}"""
                    mlflow.set_tag("mlflow.note.content", description)

                    # 🔍 RandomizedSearchCV
                    random_search = RandomizedSearchCV(
                        estimator=xgb_reg,
                        param_distributions=cfg["params"]["variable_sets"],
                        n_iter=cfg["params"]['n_iter_search'],
                        cv=TimeSeriesSplit(n_splits=5),
                        verbose=1,
                        random_state=42,
                        n_jobs=-1
                    )

                    random_search.fit(X_train_tmp, y_train)

                    # 🏆 Najlepszy model i parametry
                    best_params = random_search.best_params_
                    best_model = random_search.best_estimator_

                    mlflow.log_params(best_params)

                    # 🧪 Ewaluacja
                    y_train_pred = best_model.predict(X_train_tmp)
                    y_test_pred = best_model.predict(X_test_tmp)

                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    log_metrics_and_plot(y_train, y_train_pred, y_test, y_test_pred)

                    # 💾 Zapis i logowanie modelu do MLflow
                    with tempfile.TemporaryDirectory() as tmpdir:
                        model_path = os.path.join(tmpdir, "xgboost_best_model.pkl")

                        # pickle — bo używamy XGBRegressor
                        with open(model_path, "wb") as f:
                            pickle.dump(best_model, f)

                        # infer_signature
                        signature = infer_signature(
                            X_train_tmp[:1],
                            best_model.predict(X_train_tmp[:1])
                        )

                        mlflow.pyfunc.log_model(
                            artifact_path="xgboost_best_model",
                            python_model=XGBoostWrapper(),
                            artifacts={"model_path": model_path},
                            input_example=X_train_tmp[:1],
                            signature=signature
                        )

                        results.append({
                            "run_id": mlflow.active_run().info.run_id,
                            "run_name": f"{cfg['name']}",
                            "test_mae": test_mae,
                            "model_type": cfg["type"],
                            "model_name": "xgboost_best_model"
                        })

    ############################################### LIGHTGBM MODELS ##########################################################
        if cfg["type"] == "lightgbm":
            for vars_set_ind, vars_set_list in cfg["params"]["variable_sets"].items():
                
                X_train_tmp = X_train.copy().loc[:,vars_set_list]
                X_test_tmp = X_test.copy().loc[:,vars_set_list]            
                
                lgb_reg = lgb.LGBMRegressor(random_state = 42
                                            ,linear_tree=True)

                with mlflow.start_run(run_name=f"LightGBM__{vars_set_ind}__{start_train_date}"):

                    mlflow.set_tag("run_date", today)
                    random_search = RandomizedSearchCV(
                        estimator= lgb_reg,
                        param_distributions= LightGBM_param_grid,
                        n_iter= cfg["params"]['n_iter_search'],
                        cv= TimeSeriesSplit(n_splits=5),
                        verbose= 1,
                        random_state= 42,
                        n_jobs= 1
                    )

                    used_features = list(X_train_tmp.columns)
                    description = f"""Cechy: {", ".join(used_features)}
                    Zakres danych treningowych: {X_train_tmp.index[0]} do {X_train_tmp.index[-1]}
                    Przeszukiwana przestrzeń parametrów:
                    {cfg["params"]["variable_sets"]}
                    Liczba iteracji: {cfg["params"]['n_iter_search']}"""

                    mlflow.set_tag("mlflow.note.content", description)
                    random_search.fit(X_train_tmp, y_train)

                    best_params = random_search.best_params_
                    best_model = random_search.best_estimator_

                    # log parametry najlepszego modelu
                    mlflow.log_params(best_params)

                    # ewaluacja
                    y_train_pred = best_model.predict(X_train_tmp).flatten()
                    y_test_pred = best_model.predict(X_test_tmp).flatten()

                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    log_metrics_and_plot(y_train, y_train_pred, y_test, y_test_pred)

                    # zapis modelu do tymczasowego folderu
                    with tempfile.TemporaryDirectory() as tmpdir:
                        model_path = os.path.join(tmpdir, "lightgbm_best_model.pkl")
                        with open(model_path, "wb") as f:
                            pickle.dump(best_model, f)

                        signature = infer_signature(
                            X_train_tmp[:1], best_model.predict(X_train_tmp[:1])
                        )

                        mlflow.pyfunc.log_model(
                            artifact_path="lightgbm_best_model",
                            python_model=LightGBMWrapper(),
                            artifacts={"model_path": model_path},
                            input_example=X_train[:1],
                            signature=signature
                        )

                        results.append({
                            "run_id": mlflow.active_run().info.run_id,
                            "run_name": f"{cfg['name']}",
                            "test_mae": test_mae,
                            "model_type": cfg["type"],
                            "model_name": "lightgbm_best_model"
                        })

print("All models logged to MLflow. Run `mlflow ui` to inspect results.")

# COMMAND ----------

# Wybierz pierwszy model typu LightGBM z listy wyników
best = next((r for r in results if r["model_type"] == "lightgbm"), None)

if best is None:
    raise ValueError("Nie znaleziono modelu LightGBM w wynikach!")

best_run_id = best["run_id"]
best_model_uri = f"runs:/{best_run_id}/{best['model_name']}"

print("MAE:", best["test_mae"])
print("Model URI:", best_model_uri)

# COMMAND ----------

# best = min(results, key=lambda x: x["test_mae"])
# best_run_id = best["run_id"]
# best_model_uri = f"runs:/{best_run_id}/{best['model_name']}"

# print("Best MAE:", best["test_mae"])
# print("Model URI:", best_model_uri)

# COMMAND ----------

mlflow.set_registry_uri("databricks")  # legacy registry, not UC

registered_model_name = f"{var_read_write}_v{major_version}_{schema_model_name}"

description = f'Model trained on history: {X_train.index[0].strftime("%Y-%m-%d")} - {X_train.index[-1].strftime("%Y-%m-%d")}'


# Register the best model
result = mlflow.register_model(
    model_uri=best_model_uri,
    name=registered_model_name
    )

# Add alias "best" to this version
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name=registered_model_name,
    version=result.version,
    stage="Production",  # acts like "best"
    archive_existing_versions=True  # archive older versions
)

client.update_registered_model(
  name=registered_model_name,
  description=description
)

# COMMAND ----------


'''
# delete model from registry
from mlflow import MlflowClient
import mlflow

mlflow.set_registry_uri("databricks")  # legacy registry, not UC
client = MlflowClient()

model_name = "power_nuclear_v2_dev"

# List all versions of the model
versions = client.search_model_versions(f"name='{model_name}'")

for v in versions:
    if v.current_stage in ["Production", "Staging"]:
        print(f"Archiving version {v.version} currently in {v.current_stage}")
        client.transition_model_version_stage(
            name=model_name,
            version=v.version,
            stage="Archived"
        )


client.delete_registered_model(name=model_name)
'''

# COMMAND ----------

model_fitted = mlflow.pyfunc.load_model(f"models:/{registered_model_name}/Production")
# todo fix model differentiation
try:
  preds = model_fitted.predict(scaled_X_test)
except:
  preds = model_fitted.predict(X_test)

# COMMAND ----------

# todo fix model differentiation
try:
    y_train_pred = pd.Series(model_fitted.predict(scaled_X_train), name='Fitted-Train').to_frame()
    y_test_pred = pd.Series(best_model.predict(scaled_X_test).flatten(), name='Predicted').to_frame()
except:               
    y_train_pred = pd.Series(model_fitted.predict(X_train), name='Fitted-Train').to_frame()
    y_test_pred = pd.Series(model.predict(X_test), name='Predicted').to_frame()                  

# plot Fitted Data
pd.concat([  y_train_pred
            ,pd.concat([ train[[main_var]],test[[main_var]] ], axis=0)
            ,y_test_pred
            ], axis=1).plot()
   

#CalculateR2andR2adjForSatModelsWithFormula(test, main_var, model_fitted)

#print( 'MAE: ' + str(round( mean_absolute_error(test[[main_var]], yhat_test), 2) ))
#print( 'MAPE: ' + str(round( 100*mean_absolute_percentage_error(test[[main_var]], yhat_test), 2)) )
#print( 'RMSE: ' + str(round( np.sqrt(mean_squared_error(test[[main_var]], yhat_test)), 2) ))
