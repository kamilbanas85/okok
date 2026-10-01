#%%-----------------------------------------------------------------------
### START FILE

###------------------------------------------
# set up working directory - project root
###------------------------------------------

import os
from pathlib import Path

# Set working directory to the project root
os.chdir(Path(__file__).resolve().parents[1])

# take projet root path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


#%%------------------------------------------
# import libraries
###------------------------------------------

import os
from pathlib import Path
from datetime import date
import random

from types import MappingProxyType


import mlflow


from lightgbm import LGBMRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    StandardScaler, PowerTransformer
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.base import clone

import statsmodels.formula.api as smf


from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import (
    RandomizedSearchCV, GridSearchCV, TimeSeriesSplit, ParameterSampler
)

import tensorflow as tf
import random
from tensorflow.keras.callbacks import EarlyStopping

from src.mlflow.log_mlflow import (
    log_metrics_and_plot,
    plot_training_history, 
    log_tree_model_feature_importance,
    log_model_generic
)
from src.modelling.models_utils.ffnn_model import create_feed_forward_model_pipe


from scipy.stats import loguniform
from random import randint, uniform

from src.modelling.models_utils.input_models import (
    select_models_and_inputs,
    build_train_test_windows
)


from src.modelling.models_utils.predict_with_lags import (
    clean_lag_bloks_for_dependent_var,
    make_ts_with_lags_forecast
)

from src.features.transormers.seasonal_var_filter import SeasonalFeatureFilter
from src.features.transormers.dummy_encoder import DummyEncoder
from src.features.transormers.lags_fwds_generator import LagsAndFwdsGenerator

from src.features.features_utils import remove_var_if_data_to_short

from config.mlflow_config import setup_mlflow, MODEL_ARTIFACT_NAME
from config.project_config import DATA_DIR

from src.mlflow.save_model import export_model_from_run

from src.utils.acf_pcf import plot_acf_pacf
from experiments.hyperarameters_def import (
    ann_param_grid,
    lightgbm_param_grid,
    xgboost_param_grid
)

from experiments.variables_sets_def import (
    variable_sets,
    dummies_sets,
    lags_sets,
    #fwrd_sets,
    features_type,
    formulas_statsmodels,
    lags_sets_formula,
    #fwrd_sets_formula,
    features_type_formula
)

#%%------------------------------------------------------------------------
# Set up variables
#--------------------------------------------------------------------------

main_var = "gen_solar_full"

#%%------------------------------------------------------------------------
# set up mlflow project
#--------------------------------------------------------------------------

# set up tracing for mlflow runs
setup_mlflow()

EXPERIMENT_NAME = f"power_generation_solar_pl_general_variables_without_negative_price"
mlflow.set_experiment(EXPERIMENT_NAME)

#%%------------------------------------------------------------------------
# read data
#--------------------------------------------------------------------------

# load CSV
data_file_path = DATA_DIR / 'analysis_data.csv'

data_analysis = pd.read_csv(data_file_path)

data_analysis = data_analysis\
    .assign(dtimeUTC = lambda x: pd.to_datetime(x['dtimeUTC']))\
    .set_index('dtimeUTC')

# remove rows with negative price days
data_analysis.query('neg_price_day == 0', inplace=True)

#data_analysis.loc['2025-03-29':'2025-03-31']


#%%---------------------------------------------------------------
# Model configurations
# ----------------------------------------------------------------

numbers_obs_allowed_for_seasonal = 24*30*24  # 2 years

# start training date list:
start_date = data_analysis.index.min().strftime('%Y-%m-%d')
test_days_nr = 60  # 60 days * 24 hours/day = 1440 hours
train_period_list = [ 12, 24, 36, 48, start_date]

train_test_window_list = build_train_test_windows(
    index=data_analysis.index,
    test_days_nr=test_days_nr,
    train_period_list=train_period_list
)


models = [
    {
        "name": "linear_regression_ols",
        "type": "statsmodels",
        "params": {
            "formulas": formulas_statsmodels
            ,"train_val_test_start_list": train_test_window_list
            ,"lags_sets": lags_sets_formula
            ,"features_type": features_type_formula
            ,"days_to_retrain":14

        }
    },
    {
        "name": "lightgbm",
        "type": "lightgbm",
        "params": {
            "variable_sets": variable_sets
           ,"train_val_test_start_list": train_test_window_list
           ,"n_iter_search": 20
           ,"lags_sets": lags_sets
           ,"features_type": features_type
           ,"days_to_retrain":14
        }
    },
    {
        "name": "nn_feedforward",
        "type": "keras",
        "params": {
            "variable_sets": variable_sets
           ,"train_val_test_start_list": train_test_window_list
           ,"n_iter_search": 20
           ,"lags_sets": lags_sets
           ,"features_type": features_type
           ,"max_epochs": 50
           ,"days_to_retrain":14
        }
    }
]


# -----------------------------------------------------
# Select model to run
# -----------------------------------------------------
types_sel = ["lightgbm", "statsmodels", "keras"]
#types_sel = None
types_sel = ["keras"]

var_set_selected = ["set8"]

selected_formulas = ["formula09", "formula12"]

train_start_sel = ['2024-03-23']

model_sel = select_models_and_inputs(
    models=models,
    selected_models_types=types_sel,
    selected_variable_sets=var_set_selected,
    selected_formulas=None,
    seleted_train_start=train_start_sel
)

# make immutable
model_sel = tuple([MappingProxyType(d) for d in model_sel])

# -----------------------------------------------------
# Loop over models
# -----------------------------------------------------

today = date.today().isoformat()  # e.g. "2025-09-13"
results = []

for cfg in model_sel:

    for i, start_set_dates in enumerate(cfg["params"]["train_val_test_start_list"]):

        train_start = pd.to_datetime(start_set_dates["train_start"])
        test_start = pd.to_datetime(start_set_dates["test_start"])

        #---------------------------------------------------------------------
        ### STAT MODELS ###
        #---------------------------------------------------------------------
        if cfg["type"] == "statsmodels":
            for formula_name, formula in cfg["params"]["formulas"].items():

                data_subset = data_analysis[data_analysis.index >= train_start]

                # extract model set up params
                lags_direct_list = cfg["params"].get('lags_sets', {}).get(formula_name, {})
                fwds_direct_list = cfg["params"].get('fwrd_sets', {}).get(formula_name, {})
                dummy_for_columns = []
                days_to_retrain = cfg["params"].get('days_to_retrain', 7)
                train_period = start_set_dates.get('train_period', None)
                
                #-----------------------------------
                # add lags into set if required
                #-----------------------------------
                if lags_direct_list:

                    add_lags_and_fwds = LagsAndFwdsGenerator(
                        lags_direct=lags_direct_list,
                        fwds_direct=fwds_direct_list,
                        cuts_for_na=True
                    )

                    data_subset = add_lags_and_fwds.fit_transform(data_subset).copy()

                #-----------------------------------
                # prepare train, test sets
                #-----------------------------------               
                X, y = data_subset.drop(columns=[main_var]).copy(), data_subset[[main_var]].copy()
        
                X_train = X.loc[train_start : test_start - pd.Timedelta(seconds=1)].copy()
                X_test = X.loc[test_start:].copy()
                y_train = y.loc[train_start : test_start - pd.Timedelta(seconds=1)].copy()
                y_test = y.loc[test_start:].copy()
                
                #-----------------------------------
                # remove seasonal variables if data is too short
                #-----------------------------------
                if len(X_train) <= numbers_obs_allowed_for_seasonal:
                    formula = remove_var_if_data_to_short(
                        ['week', 'month'],
                        'stat_formula',
                        formula = formula
                    )

                #-----------------------------------
                # train model
                #-----------------------------------
                model_lr = smf.ols(
                    formula=formula,
                    data=pd.concat([X_train, y_train], axis=1)
                ).fit()

                y_train_pred = model_lr.predict(X_train)
                y_train_pred = pd.DataFrame(
                    y_train_pred,
                    index=X_train.index,
                    columns=["Fitted-Train"]
                )
                
                #---------------------------------------
                # Make prediction on test set - evaluation
                #--------------------------------------- 
                train_start_c = train_start
                test_start_c = test_start

                y_test_pred_all = []

                while test_start_c <= X.index.max():

                    test_end_c = test_start_c + pd.Timedelta(days=days_to_retrain) - pd.Timedelta(seconds=1)
                    
                    X_train_c = X.loc[train_start_c : test_start_c - pd.Timedelta(seconds=1)].copy()
                    y_train_c = y.loc[train_start_c : test_start_c - pd.Timedelta(seconds=1)].copy()

                    X_test_c = X.loc[test_start_c : test_end_c].copy()
                    y_test_c = y.loc[test_start_c : test_end_c].copy()


                    # add if not seqence data
                    if len(y_test_c) == 0:
                        if isinstance(train_period, int):
                            train_start_c = train_start_c + pd.Timedelta(days=1)
                        test_start_c = test_start_c + pd.Timedelta(days=1)
                        continue

                    model_lr_c = smf.ols(
                        formula=formula,
                        data=pd.concat([X_train_c, y_train_c], axis=1)
                    ).fit()

                    if lags_direct_list and (main_var in lags_direct_list.keys()):
                        
                        # clean lags for test safty
                        X_test_zeros = clean_lag_bloks_for_dependent_var(
                            X = X_test_c,
                            lags_dict = lags_direct_list,
                            dependent_var = main_var,
                            block_size = 24,
                            fill_value = np.nan
                        )

                        y_test_pred, X_test_with_lags = make_ts_with_lags_forecast(
                            X_test = X_test_zeros,
                            model = model_lr_c,
                            dependent_var = main_var,
                            lags_dict = lags_direct_list,
                            add_intercept = False,
                            test_or_forecast = 'Test',
                            horizon_forecast = 24
                        )

                    else:
                        y_test_pred = model_lr_c.predict(X_test_c)
                        y_test_pred = pd.DataFrame(
                            y_test_pred,
                            index=X_test_c.index,
                            columns=['Predicted-Test']
                        )

                    # append results
                    y_test_pred_all.append(y_test_pred)

                    # move rolling widnow
                    if isinstance(train_period, int):
                        train_start_c = train_start_c + pd.Timedelta(days=days_to_retrain)
                    test_start_c = test_start_c + pd.Timedelta(days=days_to_retrain)

                y_test_pred_all = pd.concat(y_test_pred_all)

                #-----------------------------------
                # log data to mlflow
                #-----------------------------------
                with mlflow.start_run(run_name=f"{cfg['name']}__{formula_name}__{train_start}"):
                    
                    # select used variables
                    design_info = model_lr.model.data.design_info
                    used_features = list({
                            factor.name().replace("C(", "").replace(")", "") 
                            for term in design_info.terms 
                            for factor in term.factors
                            if factor.name() != "Intercept"
                        })

                    # log model config for model registry
                    model_config = {
                        "model_type": cfg["type"],
                        "features":used_features,
                        "lags_direct_list": lags_direct_list,
                        "fwds_direct_list": fwds_direct_list,
                        "dummy_for_columns": [],
                        "formula": formula,
                        "train_period":train_period,
                        "train_start":train_start,
                        "test_start":test_start
                    }

                    mlflow.log_dict(model_config, 'model_config.json')

                    # log tags for comparing-filter models
                    features_type = cfg["params"].get('features_type',{}).get(f'{formula_name}',{})

                    mlflow.set_tag("model_type", cfg["type"]) 
                    mlflow.set_tag("train_start", train_start) 
                    mlflow.set_tag("run_date", today) 
                    mlflow.set_tag("features_type", features_type) 

                    # log best model params
                    mlflow.log_params({"model": cfg['name'], "formula": formula})
                    mlflow.log_text(model_lr.summary().as_text(), "ols_summary.txt")
                    log_metrics_and_plot(y_train, y_train_pred, y_test, y_test_pred_all)

                    # description
                    description = f"""
                        Features: {", ".join(used_features)}
                        Training Data Range: {train_start} to {X_train.index[-1]}
                        Test Data Range: {test_start} to {X_test.index[-1]}
                        Formula: {formula}
                    """
                    mlflow.set_tag("mlflow.note.content", description)

                    #log model
                    log_model_generic(model_lr, cfg["type"], MODEL_ARTIFACT_NAME)

        #---------------------------------------------------------------------
        ### KERAS MODELS ###
        #---------------------------------------------------------------------
        if cfg["type"] == "keras":
            for vars_set_ind, vars_set_list in cfg["params"]["variable_sets"].items():

                data_subset = data_analysis[data_analysis.index >= train_start]

                # extract model set up params
                lags_direct_list = cfg["params"].get('lags_sets', {}).get(vars_set_ind, {})
                fwds_direct_list = cfg["params"].get('fwrd_sets', {}).get(vars_set_ind, {})
                dummy_for_columns = dummies_sets[vars_set_ind]
                days_to_retrain = cfg["params"].get('days_to_retrain', 7)
                train_period = start_set_dates.get('train_period', None)
                
                #-----------------------------------
                # add lags into set if required
                #-----------------------------------
                if lags_direct_list:

                    add_lags_and_fwds = LagsAndFwdsGenerator(
                        lags_direct=lags_direct_list,
                        fwds_direct=fwds_direct_list,
                        cuts_for_na=True
                    )

                    data_subset = add_lags_and_fwds.fit_transform(data_subset).copy()
                    vars_set_list = vars_set_list.copy()
                    vars_set_list.extend( add_lags_and_fwds.lags_dict_.keys() )
                    vars_set_list.extend( add_lags_and_fwds.fwds_dict_.keys() )

                #-----------------------------------
                # prepare train, test sets
                #-----------------------------------               
                X, y = data_subset.drop(columns=[main_var]).copy(), data_subset[[main_var]].copy()
        
                X_train = X.loc[train_start : test_start - pd.Timedelta(seconds=1)].copy()
                X_test = X.loc[test_start:].copy()
                y_train = y.loc[train_start : test_start - pd.Timedelta(seconds=1)].copy()
                y_test = y.loc[test_start:].copy()
                
                train_set_len = len(X_train)

                #-----------------------------------
                # prepare pipeline
                #-----------------------------------  
                tf.keras.backend.clear_session()

                pipe_nn_forward_base = Pipeline([
                    ('seasonal_filter', SeasonalFeatureFilter(
                        vars_set_list=vars_set_list,
                        min_obs=numbers_obs_allowed_for_seasonal,
                        train_set_len=train_set_len,
                        vars_to_check=['week','month'],
                        model_type='ml',
                        dummy_cols=dummy_for_columns
                    )),
                    ('dummy_encoder', DummyEncoder(
                        dummy_cols=dummy_for_columns,
                        drop_first=False
                    )),
                    ('scaler_X',
                     ColumnTransformer(
                        transformers=[],
                        remainder=PowerTransformer(method="yeo-johnson")
                     )),
                     ('model', 
                      KerasRegressor(
                        model = create_feed_forward_model_pipe,
                        verbose=0
                     ))

                ])

                pipe_nn_feedforward = TransformedTargetRegressor(
                    regressor=pipe_nn_forward_base,
                    transformer=PowerTransformer(method="yeo-johnson")
                )

                #-----------------------------------
                # define grid search model and search hyperparameters - with early stopping
                #-----------------------------------
                tf.random.set_seed(42)
                np.random.seed(42)
                random.seed(42)


                param_list = list(ParameterSampler(
                    ann_param_grid,
                    n_iter=cfg["params"]['n_iter_search'],
                    random_state=42
                ))

                best_score = np.inf
                best_model = None
                best_params = None
                best_epoch_for_retrain = None
                best_history = None

                for params in param_list:
                    early_stop = EarlyStopping(
                        monitor='val_loss',
                        patience=15,
                        restore_best_weights=True
                    )

                    # important - clone pipeline
                    pipe_nn = clone(pipe_nn_feedforward)
                    pipe_nn.set_params(**params)

                    # training
                    pipe_nn.fit(
                        X_train,
                        y_train,
                        model__epochs=cfg["params"]["max_epochs"],   # large → early stopping will cut
                        model__callbacks=[early_stop],
                        model__validation_split=0.2,
                        model__shuffle=False,
                        model__verbose=0
                    )

                    # extract keras model inside pipeline
                    keras_model = pipe_nn.regressor_.named_steps["model"]
                    history = keras_model.history_

                    val_losses = history['val_loss']
                    val_loss = min(val_losses)

                    # Find epoch with lowest val_loss
                    best_epoch = int(np.argmin(val_losses)) + 1

                    if val_loss < best_score:
                        best_score = val_loss
                        best_model = pipe_nn
                        best_params = params
                        best_epoch_for_retrain = best_epoch
                        best_history = history

                #---------------------------------------
                # retrain on whole data with best hyperparameters
                #---------------------------------------  
                best_model_ann = clone(pipe_nn_feedforward)
                best_model_ann.set_params(**best_params)
                tf.keras.backend.clear_session()

                # retrain
                best_model_ann.fit(
                    X_train,
                    y_train,
                    model__epochs=best_epoch_for_retrain,
                    model__shuffle=False,
                    model__verbose=0
                )                
                y_train_pred = best_model_ann.predict(X_train)
                y_train_pred = pd.DataFrame(
                    y_train_pred.flatten(),
                    index=y_train.index,
                    columns=["Fitted-Train"]
                )

                #---------------------------------------
                # Make prediction on test set - evaluation
                #---------------------------------------
                train_start_c = train_start
                test_start_c = test_start

                y_test_pred_all = []

                while test_start_c <= X.index.max():

                    test_end_c = test_start_c + pd.Timedelta(days=days_to_retrain) - pd.Timedelta(seconds=1)
                    
                    X_train_c = X.loc[train_start_c : test_start_c - pd.Timedelta(seconds=1)].copy()
                    y_train_c = y.loc[train_start_c : test_start_c - pd.Timedelta(seconds=1)].copy()

                    X_test_c = X.loc[test_start_c : test_end_c].copy()
                    y_test_c = y.loc[test_start_c : test_end_c].copy()


                    # add if not seqence data
                    if len(y_test_c) == 0:
                        if isinstance(train_period, int):
                            train_start_c = train_start_c + pd.Timedelta(days=1)
                        test_start_c = test_start_c + pd.Timedelta(days=1)
                        continue

                    tf.keras.backend.clear_session()                    
                    best_model_ann = clone(pipe_nn_feedforward)
                    best_model_ann.set_params(**best_params)
                    
                    best_model_ann.fit(
                        X_train_c, y_train_c, 
                        model__epochs=best_epoch_for_retrain,
                        #model__shuffle=False,
                        #model__verbose=0
                    )

                    if lags_direct_list and (main_var in lags_direct_list.keys()):
                        
                        # clean lags for test safty
                        X_test_zeros = clean_lag_bloks_for_dependent_var(
                            X = X_test_c,
                            lags_dict = lags_direct_list,
                            dependent_var = main_var,
                            block_size = 24,
                            fill_value = np.nan
                        )

                        y_test_pred, X_test_with_lags = make_ts_with_lags_forecast(
                            X_test = X_test_zeros,
                            model = best_model_ann,
                            dependent_var = main_var,
                            lags_dict = lags_direct_list,
                            add_intercept = False,
                            test_or_forecast = 'Test',
                            horizon_forecast = 24
                        )

                    else:
                        y_test_pred = best_model_ann.predict(X_test_c)
                        y_test_pred = pd.DataFrame(
                            y_test_pred,
                            index=X_test_c.index,
                            columns=['Predicted-Test']
                        )

                    # append results
                    y_test_pred_all.append(y_test_pred)

                    # move rolling widnow
                    if isinstance(train_period, int):
                        train_start_c = train_start_c + pd.Timedelta(days=days_to_retrain)
                    test_start_c = test_start_c + pd.Timedelta(days=days_to_retrain)

                y_test_pred_all = pd.concat(y_test_pred_all)

                #-----------------------------------
                # log data to mlflow
                #-----------------------------------
                with mlflow.start_run(run_name=f"{cfg['name']}__{vars_set_ind}__{train_start}"):
                    
                    # log model config for model registry
                    model_config = {
                        "model_type": cfg["type"],
                        "features":vars_set_list,
                        "lags_direct_list": lags_direct_list,
                        "fwds_direct_list": fwds_direct_list,
                        "dummy_for_columns": dummy_for_columns,
                        "best_params": best_params,
                        "n_epoch":best_epoch_for_retrain,
                        "feature_after_prepocessing":list(best_model_ann.regressor_[:-2].get_feature_names_out()),
                        "train_period":train_period,
                        "train_start":train_start,
                        "test_start":test_start
                    }

                    mlflow.log_dict(model_config, 'model_config.json')

                    # log tags for comparing-filter models
                    features_type = cfg["params"].get('features_type',{}).get(f'{vars_set_ind}',{})

                    mlflow.set_tag("model_type", cfg["type"]) 
                    mlflow.set_tag("train_start", train_start) 
                    mlflow.set_tag("run_date", today) 
                    mlflow.set_tag("features_type", features_type) 

                    # log best model params
                    mlflow.log_params(best_params)
                    log_metrics_and_plot(y_train, y_train_pred, y_test, y_test_pred_all)
                    plot_training_history(best_history)

                    # description
                    description = f"""
                        Features: {", ".join(vars_set_list)}
                        Training Data Range: {train_start} to {X_train.index[-1]}
                        Validation Data Range: 0.2% of train
                        Test Data Range: {test_start} to {X_test.index[-1]}
                        Hyperparameter Search Space:
                        {ann_param_grid}
                        The best hyperparameters found: {best_params}
                    """

                    mlflow.set_tag("mlflow.note.content",description)

                    #log model
                    log_model_generic(best_model_ann, cfg["type"], MODEL_ARTIFACT_NAME)

        #---------------------------------------------------------------------
        ### LIGHTGBM MODELS ###
        #---------------------------------------------------------------------        
        if cfg["type"] == "lightgbm":
            for vars_set_ind, vars_set_list in cfg["params"]["variable_sets"].items():
                
                data_subset = data_analysis[data_analysis.index >= train_start]

                # extract model set up params
                lags_direct_list = cfg["params"].get('lags_sets', {}).get(vars_set_ind, {})
                fwds_direct_list = cfg["params"].get('fwrd_sets', {}).get(vars_set_ind, {})
                dummy_for_columns = []
                days_to_retrain = cfg["params"].get('days_to_retrain', 7)
                train_period = start_set_dates.get('train_period', None)

                #-----------------------------------
                # add lags into set if required
                #-----------------------------------
                if lags_direct_list or fwds_direct_list:

                    add_lags_and_fwds = LagsAndFwdsGenerator(
                        lags_direct=lags_direct_list,
                        fwds_direct=fwds_direct_list,
                        cuts_for_na=True
                    )

                    data_subset = add_lags_and_fwds.fit_transform(data_subset).copy()
                    vars_set_list = vars_set_list.copy()
                    vars_set_list.extend( add_lags_and_fwds.lags_dict_.keys() )
                    vars_set_list.extend( add_lags_and_fwds.fwds_dict_.keys() )

                #-----------------------------------
                # prepare train, test sets
                #-----------------------------------               
                X, y = data_subset.drop(columns=[main_var]).copy(), data_subset[[main_var]].copy()
        
                X_train = X.loc[train_start : test_start - pd.Timedelta(seconds=1)].copy()
                X_test = X.loc[test_start:].copy()
                y_train = y.loc[train_start : test_start - pd.Timedelta(seconds=1)].copy()
                y_test = y.loc[test_start:].copy()
                
                train_set_len = len(X_train)

                #-----------------------------------
                # prepare pipeline
                #-----------------------------------  
                pipe_lightgbm = Pipeline([
                    ('seasonal_filter', SeasonalFeatureFilter(
                        vars_set_list=vars_set_list,
                        min_obs=numbers_obs_allowed_for_seasonal,
                        train_set_len=train_set_len,
                        vars_to_check=['week','month'],
                        model_type='ml',
                        dummy_cols=dummy_for_columns
                    )),
                    ('dummy_encoder', DummyEncoder(
                        dummy_cols=dummy_for_columns,
                        drop_first=False
                    )),
                     ('model', LGBMRegressor(
                        random_state = 42,
                        deterministic=True,
                        force_col_wise=True,
                        n_jobs=1,
                        baggin_freq=0,
                        linear_tree=True
                    ))
                ])       
                
                #---------------------------------------
                # Prepare data for lightgbm
                #--------------------------------------- 
                random_search_lightgbm = RandomizedSearchCV(
                    estimator= pipe_lightgbm,
                    param_distributions= lightgbm_param_grid,
                    n_iter= cfg["params"]['n_iter_search'],
                    cv= TimeSeriesSplit(n_splits=5),
                    scoring='neg_mean_absolute_error',
                    verbose= 1,
                    random_state= 42,
                    n_jobs= -1
                )

                random_search_lightgbm.fit(X_train, y_train)
                #---------------------------------------
                # Extract the best model
                #--------------------------------------- 
                best_params = random_search_lightgbm.best_params_
                model_templeate = random_search_lightgbm.best_estimator_

                y_train_pred = model_templeate.predict(X_train)
                y_train_pred = pd.DataFrame(
                    y_train_pred.flatten(),
                    index=X_train.index,
                    columns=["Fitted-Train"]
                )
                #---------------------------------------
                # Make prediction on test set - evaluation
                #--------------------------------------- 
                train_start_c = train_start
                test_start_c = test_start

                y_test_pred_all = []

                while test_start_c <= X.index.max():

                    test_end_c = test_start_c + pd.Timedelta(days=days_to_retrain) - pd.Timedelta(seconds=1)
                    
                    X_train_c = X.loc[train_start_c : test_start_c - pd.Timedelta(seconds=1)].copy()
                    y_train_c = y.loc[train_start_c : test_start_c - pd.Timedelta(seconds=1)].copy()

                    X_test_c = X.loc[test_start_c : test_end_c].copy()
                    y_test_c = y.loc[test_start_c : test_end_c].copy()


                    # add if not seqence data
                    if len(y_test_c) == 0:
                        if isinstance(train_period, int):
                            train_start_c = train_start_c + pd.Timedelta(days=1)
                        test_start_c = test_start_c + pd.Timedelta(days=1)
                        continue

                    best_model = clone(model_templeate)
                    best_model.fit(X_train_c, y_train_c)

                    if lags_direct_list and (main_var in lags_direct_list.keys()):
                        
                        # clean lags for test safty
                        X_test_zeros = clean_lag_bloks_for_dependent_var(
                            X = X_test_c,
                            lags_dict = lags_direct_list,
                            dependent_var = main_var,
                            block_size = 24,
                            fill_value = np.nan
                        )

                        y_test_pred, X_test_with_lags = make_ts_with_lags_forecast(
                            X_test = X_test_zeros,
                            model = best_model,
                            dependent_var = main_var,
                            lags_dict = lags_direct_list,
                            add_intercept = False,
                            test_or_forecast = 'Test',
                            horizon_forecast = 24
                        )

                    else:
                        y_test_pred = best_model.predict(X_test_c)
                        y_test_pred = pd.DataFrame(
                            y_test_pred,
                            index=X_test_c.index,
                            columns=['Predicted-Test']
                        )

                    # append results
                    y_test_pred_all.append(y_test_pred)

                    # move rolling widnow
                    if isinstance(train_period, int):
                        train_start_c = train_start_c + pd.Timedelta(days=days_to_retrain)
                    test_start_c = test_start_c + pd.Timedelta(days=days_to_retrain)

                y_test_pred_all = pd.concat(y_test_pred_all)

                #-----------------------------------
                # log data to mlflow
                #-----------------------------------
                with mlflow.start_run(run_name=f"{cfg['name']}__{vars_set_ind}__{train_start}"):
                    
                    # log model config for model registry
                    model_config = {
                        "model_type": cfg["type"],
                        "features":vars_set_list,
                        "lags_direct_list": lags_direct_list,
                        "fwds_direct_list": fwds_direct_list,
                        "dummy_for_columns": dummy_for_columns,
                        "best_params": best_params,
                        "train_start":train_start,
                        "test_start":test_start,
                        "train_period":train_period
                    }

                    mlflow.log_dict(model_config, 'model_config.json')

                    # log tags for comparing-filter models
                    features_type = cfg["params"].get('features_type',{}).get(f'{vars_set_ind}',{})

                    mlflow.set_tag("model_type", cfg["type"]) 
                    mlflow.set_tag("train_start", train_start) 
                    mlflow.set_tag("run_date", today) 
                    mlflow.set_tag("features_type", features_type) 

                    # log best model params
                    mlflow.log_params(best_params)
                    log_metrics_and_plot(y_train, y_train_pred, y_test, y_test_pred_all)
                    
                    log_tree_model_feature_importance(
                        model=best_model.named_steps['model'],
                        feature_names=best_model[:-1].get_feature_names_out()
                    )
                    # description
                    description = f"""
                        Features: {", ".join(vars_set_list)}
                        Training Data Range: {train_start} to {X_train.index[-1]}
                        Test Data Range: {test_start} to {X_test.index[-1]}
                        Hyperparameter Search Space:
                        {lightgbm_param_grid}
                        The best hyperparameters found: {best_params}
                    """

                    mlflow.set_tag("mlflow.note.content",description)

                    #log model
                    log_model_generic(best_model, cfg["type"], MODEL_ARTIFACT_NAME)



print("✅ All models logged to MLflow. Run `mlflow ui` to inspect results.")

#%%--------------------------------------------------------------------------
# SAVE/ REGISTER MODEL
#--------------------------------------------------------------------------

#------------------------------
# find the best model
#------------------------------
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=['metrics_test_MAE ASC']
)

best_run = runs.iloc[0]

print("Best run_id: ", best_run.run_id)
print("Best test MAE: ", best_run["metrics.test_MAE"])

model_uri = f"runs:/{best_run.run_id}/"

#------------------------------
# find last run
#------------------------------

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=['start_time DESC']
)

last_run = runs.iloc[0]

print("Best run_id: ", last_run.run_id)
print("Best test MAE: ", last_run["metrics.test_MAE"])

model_uri = f"runs:/{last_run.run_id}/"

#------------------------------
# REGISTER MODEL
#------------------------------


#------------------------------
# SAVE MODEL
#------------------------------

model_name = 'model_1'
export_model_from_run(best_run.run_id,model_name)

###-----------------------------------------------------------------------
# END OF FILE
