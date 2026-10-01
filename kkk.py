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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from types import MappingProxyType

import mlflow

from lightgbm import LGBMRegressor
from scikeras.wrappers import KerasRegressor
import statsmodels.formula.api as smf


from sklearn.preprocessing import (
    StandardScaler, PowerTransformer
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor


from src.mlflow.log_mlflow import log_mlflow_metrics_plots_and_model

from src.modelling.models_utils.ffnn_model import create_feed_forward_model_pipe

from src.modelling.models_utils.input_models import (
    select_models_and_inputs,
    build_train_test_windows
)

from src.features.transormers.seasonal_var_filter import SeasonalFeatureFilter
from src.features.transormers.dummy_encoder import DummyEncoder
from src.features.transormers.lags_fwds_generator import LagsAndFwdsGenerator


from config.mlflow_config import setup_mlflow
from config.project_config import DATA_DIR, MODEL_PATH

from src.mlflow.save_model import export_model_from_run

from src.modelling.models_utils.wrapper_sklearn import SklearnPipelineWrapper
from src.modelling.models_utils.wrapper_keras import KerasPipelineWrapper
from src.modelling.models_utils.wrapper_stats import StatsmodelWrapper
from src.modelling.models_utils.wrapper_tft import (
    TFTdartWrapper, build_model
)

from torch.nn import HuberLoss, L1Loss

from src.utils.acf_pcf import plot_acf_pacf
from experiments.hyperarameters_def import (
    ann_param_grid,
    lightgbm_param_grid,
    xgboost_param_grid
)

from sklearn.base import clone

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
#data_analysis.query('neg_price_day == 0', inplace=True)

#data_analysis.loc['2025-03-29':'2025-03-31']

# result = plot_acf_pacf(data_analysis, main_var=main_var, nlags=48, alpha=0.05)

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

#input_chunk_length_list = [24, 48, 72, 168]
input_chunk_length_list = [24]

# loss_function_list ={
#     'mae': L1Loss(),
#     'quantile': QuantileRegressionLoss(quantiles=[0.1, 0.5, 0.9]),
#     'huber': HuberLoss(delta=1.0)
# }
loss_function_list ={
    'mae': L1Loss(),
}



models = [
    {
        "name": "linear_regression_ols",
        "type": "statsmodels_wrap",
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
        "type": "lightgbm_wrap",
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
        "type": "keras_wrap",
        "params": {
            "variable_sets": variable_sets
           ,"train_val_test_start_list": train_test_window_list
           ,"n_iter_search": 20
           ,"lags_sets": lags_sets
           ,"features_type": features_type
           ,"max_epochs": 50
           ,"days_to_retrain":14
        }
    },
    {
        "name": "tft_dart",
        "type": "tft_dart_wrap",
        "params": {
            "variable_sets": variable_sets
           ,"train_val_test_start_list": train_test_window_list
           ,"n_iter_search": 10
           ,"lags_sets": lags_sets
           ,"features_type": features_type
           ,"input_chunk_length": input_chunk_length_list
           ,"loss_function_list": loss_function_list
           ,"days_to_retrain":14
        }
    }    
]


# -----------------------------------------------------
# Select model to run
# -----------------------------------------------------
types_sel = ["lightgbm_wrap", "statsmodels_wrap", "keras_wrap"]
#types_sel = ["lightgbm_wrap", "keras_wrap"]

var_set_selected = ["set1", "set2", "set3", "set4", "set5", "set6", "set7", "set8"]
var_set_selected = ["set7"]

selected_formulas = ["formula09"]

train_start_sel = ['2020-04-09']

model_sel = select_models_and_inputs(
    models=models,
    selected_models_types=types_sel,
    selected_variable_sets=var_set_selected,
    selected_formulas=selected_formulas,
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
        if cfg["type"] == "statsmodels_wrap":
            for formula_name, formula in cfg["params"]["formulas"].items():

                data_subset = data_analysis[data_analysis.index >= train_start]

                # extract model set up params
                lags_direct_list = cfg["params"].get('lags_sets', {}).get(formula_name, {})
                fwds_direct_list = cfg["params"].get('fwrd_sets', {}).get(formula_name, {})
                days_to_retrain = cfg["params"].get('days_to_retrain', None)
                train_period = start_set_dates['train_period']
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

                
                model_lr_wrap = StatsmodelWrapper(
                    formula = formula,
                    main_var = main_var,
                    lags_direct_list = lags_direct_list,
                    block_size = 24,
                    train_period = train_period,
                    seasonal_vars_to_check = ['week', 'month'],
                    numbers_obs_allowed_for_seasonal = numbers_obs_allowed_for_seasonal
                )
                
                model_lr_wrap.fit(X=X_train, y=y_train)

                y_train_pred = model_lr_wrap.get_fitted_values(X_train)
                #---------------------------------------
                # evaluate rolling/expanding window
                #---------------------------------------
                y_test_pred = model_lr_wrap.evaluate_test_window(
                    X=X,
                    y=y,
                    train_start=train_start,
                    test_start=test_start,
                    days_to_retrain=days_to_retrain
                )

                #-----------------------------------
                # log data to mlflow
                #-----------------------------------
                with mlflow.start_run(run_name=f"{cfg['name']}__{formula_name}__{train_start}"):
                   
                    log_mlflow_metrics_plots_and_model(
                            cfg=cfg,
                            model=model_lr_wrap,
                            features_type=cfg["params"].get('features_type',{}).get(f'{formula_name}',{}),
                            y_train=y_train,
                            y_train_pred=y_train_pred,
                            y_test=y_test,
                            y_test_pred=y_test_pred,        
                            formula=formula,
                            lags_direct_list=lags_direct_list,
                            fwds_direct_list=fwds_direct_list,
                            train_start=train_start,
                            test_start=test_start,
                            today=today,
                            train_period=train_period,
                            features_basic=model_lr_wrap.used_features_,
                            features_after_prep=model_lr_wrap.used_features_,
                            days_to_retrain=days_to_retrain
                        )

        #---------------------------------------------------------------------
        ### KERAS MODELS ###
        #---------------------------------------------------------------------
        if cfg["type"] == "keras_wrap":
            for vars_set_ind, vars_set_list in cfg["params"]["variable_sets"].items():

                data_subset = data_analysis[data_analysis.index >= train_start]
                
                # extract model set up params
                lags_direct_list = cfg["params"].get('lags_sets', {}).get(vars_set_ind, {})
                fwds_direct_list = cfg["params"].get('fwrd_sets', {}).get(vars_set_ind, {})
                dummy_for_columns = dummies_sets[vars_set_ind]
                days_to_retrain = cfg["params"].get('days_to_retrain', None)
                train_period = start_set_dates['train_period']                
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

                #---------------------------------------
                # initiate pipeline
                #---------------------------------------
                model_nn_feedforward = KerasPipelineWrapper(
                    model=pipe_nn_feedforward,
                    main_var=main_var,
                    lags_direct_list=lags_direct_list,
                    block_size=24,
                    train_period=train_period,
                    features_basic=vars_set_list
                )
                #---------------------------------------
                # fit random search - find hyperparameters
                #---------------------------------------
                model_nn_feedforward.fit_random_search(
                    X=X_train,
                    y=y_train,
                    param_grid=ann_param_grid,
                    n_iter=cfg["params"]["n_iter_search"],
                    random_state=42,
                    fit_kwargs={
                        "model__epochs": cfg["params"]["max_epochs"],
                        "model__validation_split": 0.2,
                        "model__shuffle": False,
                        "model__verbose": 0,
                    },
                    early_stopping={
                        "monitor": "val_loss",
                        "patience": 15,
                        "restore_best_weights": True,
                    },
                    retrain_on_full_data=True,
                )

                #---------------------------------------
                # extract best params etc.
                #---------------------------------------
                best_epoch_for_retrain = model_nn_feedforward.best_epoch_
                best_params = model_nn_feedforward.model_params
                best_history = model_nn_feedforward.history_
                feature_after_prepocessing = model_nn_feedforward.model_fitted.regressor_[:-2].get_feature_names_out()

               #---------------------------------------
                # take train fitted data
                #---------------------------------------
                y_train_pred = model_nn_feedforward.get_fitted_values(
                    X=X_train
                )

                #---------------------------------------
                # evaluate rolling/expanding window
                #---------------------------------------
                y_test_pred = model_nn_feedforward.evaluate_test_window(
                    X=X,
                    y=y,
                    train_start=train_start,
                    test_start=test_start,
                    days_to_retrain=days_to_retrain,
                    epochs=best_epoch_for_retrain
                )

             
                with mlflow.start_run(run_name=f"{cfg['name']}__{vars_set_ind}__{train_start}"):

                    log_mlflow_metrics_plots_and_model(
                            cfg=cfg,
                            model=model_nn_feedforward,
                            features_type=cfg["params"].get('features_type',{}).get(f'{vars_set_ind}',{}),
                            y_train=y_train,
                            y_train_pred=y_train_pred,
                            y_test=y_test,
                            y_test_pred=y_test_pred,        
                            lags_direct_list=lags_direct_list,
                            fwds_direct_list=fwds_direct_list,
                            dummy_for_columns=dummy_for_columns,
                            train_start=train_start,
                            test_start=test_start,
                            today=today,
                            features_basic=vars_set_list,
                            features_after_prep=feature_after_prepocessing,
                            param_search_space=ann_param_grid,
                            best_params=best_params,
                            best_epoch_for_retrain=best_epoch_for_retrain,
                            best_history=best_history,
                            train_period = train_period,
                            days_to_retrain=days_to_retrain
                        )

        #---------------------------------------------------------------------
        ### LIGHTGBM MODELS ###
        #---------------------------------------------------------------------        
        if cfg["type"] == "lightgbm_wrap":
            for vars_set_ind, vars_set_list in cfg["params"]["variable_sets"].items():
                
                data_subset = data_analysis[data_analysis.index >= train_start]

                # extract model set up params
                lags_direct_list = cfg["params"].get('lags_sets', {}).get(vars_set_ind, {})
                fwds_direct_list = cfg["params"].get('fwrd_sets', {}).get(vars_set_ind, {})
                dummy_for_columns = dummies_sets[vars_set_ind]
                days_to_retrain = cfg["params"].get('days_to_retrain', None)
                train_period = start_set_dates['train_period']                
                #dummy_for_columns = []
                # zmienic wyzej !!!!!!
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
                # inisiate pipeline
                #--------------------------------------- 
                model_lightgbm = SklearnPipelineWrapper(
                    model = pipe_lightgbm,
                    main_var=main_var,
                    lags_direct_list=lags_direct_list,
                    block_size=24,
                    train_period=train_period,
                    features_basic=vars_set_list
                )

                #---------------------------------------
                # fit cv - search hyperparams
                #---------------------------------------
                model_lightgbm.fit_random_search_cv_ts(
                    X = X_train,
                    y = y_train,
                    param_grid = lightgbm_param_grid,
                    n_iter = cfg["params"]['n_iter_search'],
                    cv = 5,
                    n_jobs= -1
                )

                #---------------------------------------
                # take train fitted data
                #---------------------------------------
                y_train_pred = model_lightgbm.get_fitted_values(X=X_train)

                #---------------------------------------
                # evaluate rolling/expanding window
                #---------------------------------------
                y_test_pred = model_lightgbm.evaluate_test_window(
                    X = X,
                    y = y,
                    train_start = train_start,
                    test_start = test_start,
                    days_to_retrain = days_to_retrain,
                )

                # Extract the best params
                best_params = model_lightgbm.model_params
               
                #-----------------------------------
                # log data to mlflow
                #-----------------------------------
                with mlflow.start_run(run_name=f"{cfg['name']}__{vars_set_ind}__{train_start}"):
                    
                    log_mlflow_metrics_plots_and_model(
                            cfg=cfg,
                            model=model_lightgbm,
                            features_type=cfg["params"].get('features_type',{}).get(f'{vars_set_ind}',{}),
                            y_train=y_train,
                            y_train_pred=y_train_pred,
                            y_test=y_test,
                            y_test_pred=y_test_pred,        
                            lags_direct_list=lags_direct_list,
                            fwds_direct_list=fwds_direct_list,
                            dummy_for_columns=dummy_for_columns,
                            train_start=train_start,
                            test_start=test_start,
                            today=today,
                            features_basic=model_lightgbm.features_basic,
                            features_after_prep=model_lightgbm.model_fitted[:-1].get_feature_names_out(),
                            param_search_space=lightgbm_param_grid,
                            best_params=best_params,
                            train_period = train_period,
                            days_to_retrain=days_to_retrain
                        )

        #---------------------------------------------------------------------
        ### TFT DART MODELS ###
        #--------------------------------------------------------------------- 
        if cfg["type"] == "tft_dart":
            for vars_set_ind, vars_set_list in cfg["params"]["variable_sets"].items():
                for input_chunk_length in cfg["params"]["input_chunk_length"]:
                    for loss_func_name, loss_func in cfg["params"]["loss_function_list"].items():

                        data_subset = data_analysis[data_analysis.index >= train_start]

                        # extract model set up params
                        lags_direct_list = cfg["params"].get('lags_set', {}).get(vars_set_ind, {})
                        fwds_direct_list = cfg["params"].get('fwrd_set', {}).get(vars_set_ind, {})
                        dummy_for_columns = dummies_sets[vars_set_ind]
                        days_to_retrain = cfg["params"].get('days_to_retrain', None)
                        train_period = start_set_dates['train_period']                        
                        #-----------------------------------
                        # add lags into set if required
                        #-----------------------------------
                        if lags_direct_list:

                            add_lags_and_fwds = LagsAndFwdsGenerator(
                                lags_dict=lags_direct_list,
                                fwds_dict=None,
                                drop_for_na=True
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
                        #  pipeline
                        #-----------------------------------  

                        pipe_prep = Pipeline([
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
                        ])

                        model_params = dict(
                                hidden_size= 64,
                                num_attention_heads= 4,
                                lstm_layers= 2,
                                dropout= 0.1,
                                optimizer_kwargs= { "lr": 0.0001 },
                                batch_size=64,
                                n_epochs=100,
                                #likelihood=quantile_likelihood,
                                #likelihood=None,             # must explicitly disable
                                #loss_fn=nn.L1Loss(),         # MAE loss
                                force_reset=True,
                                output_chunk_length=24,
                                input_chunk_length=input_chunk_length,
                            )

                        if loss_func_name == 'quantile':
                            model_params['likelihood'] = loss_func
                        else:
                            model_params['loss_fn'] = loss_func

                        # 1. Instantiate ONE wrapper with the search-time model (early stopping ON)
                        tft_wrapper = TFTdartWrapper(
                            preprocessor_future=clone(pipe_prep),
                            model=lambda: build_model(use_early_stop=True, **model_params),
                            main_var=main_var,
                            train_period=train_period,
                        )

                        # 2. Find best epoch / params — stored as attributes on the SAME object
                        tft_wrapper.fit_and_select_epochs(
                            X_train, y_train,
                            param_grid=ann_param_grid,
                            n_iter=cfg["params"]["n_iter_search"],
                        )
                        # tft_wrapper.best_params, tft_wrapper.best_epoch_ now set
                        # tft_wrapper.model_template now rebuilt as: build_model(use_early_stop=False, **best_params)

                        # 3. Fit the same object on the FULL train set for reporting (train MAE, etc.)
                        tft_wrapper.fit(X_train, y_train)
                        y_train_pred = tft_wrapper.historical_forecast(
                            X=X_train, y_hist=y_train,
                            start=y_train.index[tft_wrapper.best_params["input_chunk_length"]],
                            forecast_horizon=1, stride=1, retrain=False, last_points_only=True
                        )

                        # 4. Rolling-window evaluation — internally builds a FRESH wrapper per window,
                        #    using the same preprocessor/model_template/model_params/best_params
                        y_test_pred_all = tft_wrapper.evaluate_test_window(
                            X, y, train_start=train_start, test_start=test_start,
                            days_to_retrain=days_to_retrain, forecast_horizon=24, stride=24,
    )
                        #-----------------------------------
                        # log data to mlflow
                        #-----------------------------------
                        with mlflow.start_run(run_name=f"{cfg['name']}__{input_chunk_length}__{vars_set_ind}__{train_start}"):
                            a=1
                            # log model config for model registry



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

selected_run = runs.iloc[1]

print("Best run_id: ", selected_run.run_id)
print("Best test MAE: ", selected_run["metrics.test_mae"])
print("Model type: ", selected_run["tags.model_type"])

model_uri = f"runs:/{selected_run.run_id}/"

#------------------------------
# REGISTER MODEL
#------------------------------


#------------------------------
# SAVE MODEL
#------------------------------

model_name = 'model_lightlbm_01'
export_model_from_run(selected_run.run_id, model_name, model_folder = MODEL_PATH)

###-----------------------------------------------------------------------
# END OF FILE


