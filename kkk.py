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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    StandardScaler, PowerTransformer
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.base import clone



import tensorflow as tf
import random
from tensorflow.keras.callbacks import EarlyStopping

from src.mlflow.log_mlflow import log_metrics_and_plot, plot_training_history, log_tree_model_feature_importance


from src.mlflow.log_mlflow import (
    log_metrics_and_plot,
    plot_training_history, 
    log_tree_model_feature_importance,
    log_model_generic
)

from src.modelling.models_utils.input_models import (
    select_models_and_inputs,
    build_train_test_windows
)


from src.features.transormers.seasonal_var_filter import SeasonalFeatureFilter
from src.features.transormers.dummy_encoder import DummyEncoder
from src.features.transormers.lags_fwds_generator import LagsAndFwdsGenerator


from src.mlflow.save_model import export_model_from_run

from src.modelling.models_utils.tft_model import (
    DartWrapper, build_model, extract_losses_from_dart_pipe
)

import torch.nn as nn
from torch.nn import HuberLoss, L1Loss


from config.mlflow_config import setup_mlflow, MODEL_ARTIFACT_NAME
from config.project_config import DATA_DIR


from experiments.hyperarameters_def import (
    tft_param_grid
)

from experiments.variables_sets_def import (
    variable_sets,
    dummies_sets,
    lags_sets,
    #fwrd_sets,
    features_type,
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

EXPERIMENT_NAME = f"power_generation_solar_pl_general_variables"
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
# Veriables and hyperparameters configuration
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
        "name": "tft_dart",
        "type": "tft_dart",
        "params": {
            "variable_sets": variable_sets
           ,"train_val_test_start_list": train_test_window_list
           ,"n_iter_search": 10
           ,"lags_sets": lags_sets
           ,"features_type": features_type
           ,"input_chunk_length": input_chunk_length_list
           ,"loss_function_list": loss_function_list
           ,"days_to_retrain": 14
        }
    }
]


# -----------------------------------------------------
# Select model to run
# -----------------------------------------------------
types_sel = ["tft_dart"]

var_set_selected = ["set6"]

selected_formulas = ["formula01", "formula02", "formula03", "formula04", "formula05", "formula06"]
seleted_train_start = ['2020-04-09']

model_sel = select_models_and_inputs(
    models=models,
    selected_models_types=types_sel,
    selected_variable_sets=var_set_selected,
    selected_formulas=None,
    seleted_train_start=seleted_train_start
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
        ### TFT MODEL ###
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
                        retrain_days_nr = cfg["params"]['days_to_retrain']

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
                                    remainder=PowerTransformer(method="yeo-johnson"),
                                    verbose_feature_names_out=False
                                )),
                        ])


                        #-----------------------------------
                        # define grid search model and search hyperparameters - with early stopping
                        #-----------------------------------
                        tf.random.set_seed(42)
                        np.random.seed(42)
                        random.seed(42)


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

 
                        pipe_nn_tft_dart = DartWrapper(
                            preprocessor=clone(pipe_prep),
                            model=lambda:build_model(
                                use_early_stop=True,
                                **model_params
                            ),
                            model_params=model_params
                        )


                        # training
                        pipe_nn_tft_dart.fit(
                            X_train,
                            y_train,
                            val_ratio=0.2
                        )

                        # extract losses
                        train_losses, val_losses = extract_losses_from_dart_pipe(pipe_nn_tft_dart)
                        if len(val_losses)==0:
                            print('WARNING: val losses not recorded.')

                        # if len(val_losses) == len(train_losses) + 1:
                        #     val_losses = val_losses[1:]

                        best_history = {
                            'val_loss':val_losses,
                            'loss':train_losses
                        }

                        val_loss = min(val_losses)
                        best_epoch = int(np.argmin(val_losses)) + 1

                        print("Score: ", val_loss, "Best epoch: ", best_epoch)

                        #---------------------------------------
                        # extract the best model
                        #---------------------------------------  
                        best_params = model_params.copy()
                        best_params['n_epochs'] = best_epoch

                        pipe_nn_tft_dart_final = DartWrapper(
                            preprocessor=clone(pipe_prep),
                            model=lambda:build_model(use_early_stop=False,
                                                    **best_params),
                            model_params=best_params
                        )                    

                        #---------------------------------------
                        # Make prediction on test set - evaluation
                        #---------------------------------------  
                        pipe_nn_tft_dart_final.fit(
                            X=X_train,
                            y=y_train
                        )

                        y_train_pred = pipe_nn_tft_dart_final.historical_forecast(
                            X=X_train,
                            y_hist=y_train,
                            start=y_train.index[ best_params["input_chunk_length"] ],
                            forecast_horizon=1,
                            stride=1,
                            retrain=False,
                            last_points_only=True
                        )

                        # if train start is constant - expansion widnow
                        if not isinstance(start_set_dates["train_period"],int):
                        
                            y_test_pred_all = pipe_nn_tft_dart_final.historical_forecast(
                                X=pd.concat([X_train,X_test]),
                                y_hist=pd.concat([y_train,y_test]),
                                start=y_test.index[0],
                                forecast_horizon=24,
                                stride=24,
                                retrain=retrain_days_nr,
                                last_points_only=False
                            )
                        
                        # if train start is rolling
                        elif isinstance(start_set_dates["train_period"],int):
                            train_start_c = train_start
                            test_start_c = test_start
                            y_test_pred_all = []

                            while test_start_c <= X.index.max():

                                test_end_c = test_start_c + pd.Timedelta(days=retrain_days_nr) - pd.Timedelta(seconds=1)
                        
                                X_train_c = X.loc[train_start_c : test_start_c - pd.Timedelta(seconds=1)].copy()
                                y_train_c = y.loc[train_start_c : test_start_c - pd.Timedelta(seconds=1)].copy()

                                X_test_c = X.loc[test_start_c:test_end_c].copy()
                                y_test_c = y.loc[test_start_c:test_end_c].copy()

                                pipe_nn_tft_dart_final = DartWrapper(
                                    preprocessor=clone(pipe_prep),
                                    model=lambda:build_model(
                                        use_early_stop=False,
                                        **best_params
                                    ),
                                    model_params=best_params
                                )

                                pipe_nn_tft_dart_final.fit(
                                    X=X_train_c,
                                    y=y_train_c
                                )

                                y_test_pred = pipe_nn_tft_dart_final.historical_forecast(
                                    X=pd.concat([X_train_c,X_test_c]),
                                    y_hist=pd.concat([y_train_c,y_test_c]),
                                    start=y_test_c.index[0],
                                    forecast_horizon=24,
                                    stride=24,
                                    retrain=False,
                                    last_points_only=False
                                )

                                y_test_pred_all.append(y_test_pred)
                        
                                # move rolling window
                                train_start_c = train_start_c + pd.Timedelta(days=retrain_days_nr)
                                test_start_c = test_start_c + pd.Timedelta(days=retrain_days_nr)
                
                            y_test_pred_all = pd.concat(y_test_pred_all)

                        #-----------------------------------
                        # log data to mlflow
                        #-----------------------------------
                        with mlflow.start_run(run_name=f"{cfg['name']}__{input_chunk_length}__{vars_set_ind}__{train_start}"):
                            
                            # log model config for model registry
                            model_config = {
                                "model_type": cfg["type"],
                                "features":vars_set_list,
                                "lags_direct_list": lags_direct_list,
                                "fwds_direct_list": fwds_direct_list,
                                "dummy_for_columns": dummy_for_columns,
                                "best_params": best_params,
                                "n_epoch":best_epoch,
                                "feature_after_prepocessing":list(pipe_nn_tft_dart_final._feature_columns)
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
                            log_metrics_and_plot(y_train.loc[y_train_pred.index].iloc[:, 0], y_train_pred.iloc[:, 0], y_test.loc[y_test_pred_all.index].iloc[:, 0], y_test_pred_all.iloc[:, 0])
                            plot_training_history(best_history)

                            # description
                            description = f"""Features: {", ".join(vars_set_list)}
                            Training Data Range: {X_train.index[0]} to {X_train.index[-1]}
                            Validation Data Range: 0.2% of train
                            Test Data Range: {X_test.index[0]} to {X_test.index[-1]}
                            Hyperparameter Search Space:
                            {tft_param_grid}
                            The best hyperparameters found: {best_params} 
                            """

                            mlflow.set_tag("mlflow.note.content",description)

                            #log model
                            log_model_generic(pipe_nn_tft_dart_final, cfg["type"], MODEL_ARTIFACT_NAME)


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
