

#---------------------------------------
                if cfg["type"] == "lightgbm_wrap":
            for vars_set_ind, vars_set_list in cfg["params"]["variable_sets"].items():
                
                data_subset = data_analysis[data_analysis.index >= train_start]

                # extract model set up params
                lags_direct_list = cfg["params"].get('lags_sets', {}).get(vars_set_ind, {})
                fwds_direct_list = cfg["params"].get('fwrd_sets', {}).get(vars_set_ind, {})
                dummy_for_columns = dummies_sets[vars_set_ind]
                days_to_retrain = cfg["params"].get('days_to_retrain', None)
                train_period = start_set_dates['train_period']
                model_type=cfg["type"]                 
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
                    features_basic=vars_set_list,
                    days_to_retrain=days_to_retrain,
                    model_type=model_type                    
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
                    #days_to_retrain = days_to_retrain
                )

                # Extract
                #features_after_prep=model_lightgbm.model_fitted[:-1].get_feature_names_out()

                #--------------------model_lightgbm---------------
                # log data to mlflow
                #-----------------------------------
                with mlflow.start_run(run_name=f"{cfg['name']}__{vars_set_ind}__{train_start}"):
                    
                    log_mlflow_metrics_plots_and_model(
                            model_wrap=model_lightgbm,
                            features_type=cfg["params"].get('features_type',{}).get(f'{vars_set_ind}',{}),
                            y_train=y_train,
                            y_train_pred=y_train_pred,
                            y_test=y_test,
                            y_test_pred=y_test_pred,        
                            train_start=train_start,
                            test_start=test_start,
                            today=today,
                            param_search_space=lightgbm_param_grid,
                        )



##########################################

import numpy as np
import pandas as pd
from sklearn.base import clone

import os
import joblib


from sklearn.model_selection import (
    RandomizedSearchCV, GridSearchCV, TimeSeriesSplit, ParameterSampler
)


from src.modelling.models_utils.predict_with_lags import (
    clean_lag_bloks_for_dependent_var,
    make_ts_with_lags_forecast
)


class SklearnPipelineWrapper:
    def __init__(self, model, **kwargs):
        self.model_template = model
         # Hyperparameters
        self.model_params = kwargs.get('model_params', {}).copy()
        # Fitted sklearn estimator
        self.model_fitted = None
        # Fitted flag
        self.fitted = False

        # additional attributes
        self.main_var = kwargs.get('main_var', None)
        self.lags_direct_list = kwargs.get('lags_direct_list', {}).copy()
        self.dummy_columns = kwargs.get('dummy_columns', {}).copy()
        self.train_period = kwargs.get('train_period', None)
        self.block_size = kwargs.get('block_size', 24)
        self.features_basic = kwargs.get('features_basic', None)
        self.model_type = kwargs.get('model_type', None)
        self.days_to_retrain = kwargs.get('days_to_retrain', None)

    #------------------------------
    # BUID MODEL
    #------------------------------
    def _build_model(self):
        """
        Build a fresh sklearn estimator.

        The model/factory defines the base configuration.
        self.model_params optionally overrides that configuration.
        """

        # Build fresh model
        if callable(self.model_template):
            model = self.model_template()
        else:
            model = clone(self.model_template)

        # Apply parameter overrides
        if self.model_params:
            model.set_params(**self.model_params)

        return model

    #------------------------------
    # FIT
    #------------------------------
    def fit(
        self,
        X:pd.DataFrame,
        y:pd.DataFrame,
        **kwargs
    ):
        """
        Fit a fresh model.

        Existing model_params are applied before fitting.
        """

        model = self._build_model()
        self.model_fitted = model.fit(X,y)
        self.fitted = True

        # take column name from last step model
        feature_cols = self.model_fitted[:-1].get_feature_names_out().tolist()\
            if len(self.model_fitted.steps) > 1 else self.features_basic
        self.features_after_prep_ = feature_cols


        return self


    #------------------------------
    # RANDOM SEARCH FIT
    #------------------------------
    def fit_random_search_cv_ts(
        self,
        X:pd.DataFrame,
        y:pd.DataFrame,
        **kwargs
    ):
        """
        Hyperparameter tuning with cv - ts split
        The best parameters are stored in:
            self.model_params
        The fitted best estimator is stored in:
            self.model_fitted
        """

        # take or assign parameters
        param_grid = kwargs.get('param_grid', {})
        n_iter = kwargs.get('n_iter', 10)
        n_jobs = kwargs.get('n_jobs', -1)
        cv_splits = kwargs.get("cv_splits",5)
        verbose = kwargs.get( "verbose",1)
        random_state = kwargs.get( "random_state",42)

        # Build fresh estimator
        model = self._build_model()

        # Randomized search
        random_search = RandomizedSearchCV(
            estimator= model,
            param_distributions= param_grid,
            n_iter= n_iter,
            cv= TimeSeriesSplit(n_splits=cv_splits),
            scoring='neg_mean_absolute_error',
            verbose= verbose,
            random_state= random_state,
            n_jobs= n_jobs,
            refit=True
        )

        random_search.fit(X, y)

        # Extract the best model
        self.model_params = random_search.best_params_.copy()
        self.model_fitted = random_search.best_estimator_
        self.fitted = True

        # take column name from last step model
        feature_cols = self.model_fitted[:-1].get_feature_names_out().tolist()\
            if len(self.model_fitted.steps) > 1 else self.features_basic
        self.features_after_prep_ = feature_cols        

        return self


    #------------------------------
    # PREDICT MODEL
    #------------------------------
    def _predict_model(
        self,
        model,
        X:pd.DataFrame,
        **kwargs
    )-> pd.DataFrame:

        """
        Predict using a supplied fitted sklearn model.

        This contains the wrapper-specific forecasting logic,
        including lag-based recursive/direct forecasting.
        """

        # Parameters

        # main_var
        if self.main_var is not None:
            main_var = self.main_var
        else:
            main_var = kwargs.get('main_var', None)

        # lags_direct_list
        if self.lags_direct_list is not None:
            lags_direct_list = self.lags_direct_list
        else:
            lags_direct_list = kwargs.get('lags_direct_list', {}).copy()

        # block_size
        if self.block_size is not None:
            block_size = self.block_size
        else:
            block_size = kwargs.get('block_size', 24)

        # forecast_steps
        forecast_steps = kwargs.get('forecast_steps', block_size)
        

        test_or_forecast = kwargs.get('test_or_forecast', 'Test')
        return_X_test = kwargs.get('return_X_test', False)

        # Forecast
        if lags_direct_list and (main_var in lags_direct_list.keys()):
            # LAGED FORECAST

            # clean lags for test safty
            X_test_zeros = clean_lag_bloks_for_dependent_var(
                X = X,
                lags_dict = lags_direct_list,
                dependent_var = main_var,
                block_size = block_size,
                fill_value = np.nan
            )

            # Recursive / direct forecasting
            y_test_pred_df, X_test_with_lags = make_ts_with_lags_forecast(
                X_test = X_test_zeros,
                model = model,
                dependent_var = main_var,
                lags_dict = lags_direct_list,
                add_intercept = False,
                test_or_forecast = test_or_forecast,
                horizon_forecast = forecast_steps
            )

            if return_X_test:
                return y_test_pred_df, X_test_zeros
            else:
                return y_test_pred_df
            
        else:
            # NORMAL SKLEARN PREDICTION

            y_test_pred = model.predict(X)

            y_test_pred_df = pd.DataFrame(
                y_test_pred,
                index=X.index,
                columns=[f'Predicted-{test_or_forecast}']
            )

            return y_test_pred_df


    #------------------------------
    # PREDICT
    #------------------------------
    def predict(
        self,
        X: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:

        # Check fitted
        if self.model_fitted is None:

            raise ValueError(
                "Model is not fitted yet. "
                "Please call fit() or "
                "fit_random_search_cv_ts() first."
            )


        return self._predict_model(
            model=self.model_fitted,
            X=X,
            **kwargs
        )

    #------------------------------
    # Return fitted values - train set
    #------------------------------
    def get_fitted_values(self, X: pd.DataFrame)-> pd.DataFrame:
        """
        Return fitted values on X.
        """

        if self.model_fitted is None:
            raise ValueError("Model is not fitted yet. Please fit the model before getting fitted values.")
        
        y_train_fitted = self.model_fitted.predict(X)

        y_train_fitted_df = pd.DataFrame(
            y_train_fitted.flatten(),
            index=X.index,
            columns=["Fitted-Train"]
        )
        
        return y_train_fitted_df


    #------------------------------
    # Test set evalaution - window rolling/sliding
    #------------------------------
    def evaluate_test_window(
            self,
            X:pd.DataFrame,
            y:pd.DataFrame,
            train_start:str,
            test_start:str,
            #train_period:str|int|None=None,
            #days_to_retrain:int=7,
            **kwargs
        )->pd.DataFrame:

        """
        Test set evaluation with a rolling or expandiing window.

        Parameters
            ----------
            train_start : str
                Initial training start timestamp.

            test_start : str
                First test timestamp.

            train_period : int or str
                If int:
                    rolling training window in months.

                Otherwise:
                    expanding window.

            days_to_retrain : int
                Number of days between retraining points.

            Notes
            -----
            self.model_params are reused for every retraining.

            Therefore, if hyperparameters were selected with:

                fit_random_search_cv_ts()

            the same selected hyperparameters are used
            throughout the rolling evaluation.        
        """

        # --------------------------------------
        # Parameters
        # --------------------------------------
        # main_var
        if self.main_var is not None:
            main_var = self.main_var
        else:
            main_var = kwargs.get('main_var', None)

        # lags_direct_list
        if self.lags_direct_list is not None:
            lags_direct_list = self.lags_direct_list
        else:
            lags_direct_list = kwargs.get('lags_direct_list', {}).copy()

        # block_size
        if self.block_size is not None:
            block_size = self.block_size
        else:
            block_size = kwargs.get('block_size', 24)

        # train_period
        if self.train_period is not None:
            train_period = self.train_period
        else:
            train_period = kwargs.get('train_period', None)

        # days_to_retrain
        if self.days_to_retrain is not None:
            days_to_retrain = self.days_to_retrain
        else:
            days_to_retrain = kwargs.get('days_to_retrain', None)

        # --------------------------------------------------------
        # Make sure indexes are datetime
        # --------------------------------------------------------
        if not isinstance( X.index,pd.DatetimeIndex):
            raise TypeError("X.index must be a pandas DatetimeIndex.")

        if not isinstance(y.index,pd.DatetimeIndex):
            raise TypeError("y.index must be a pandas DatetimeIndex.")

        #---------------------------------------
        # ROLLING LOOP
        #--------------------------------------- 
        y_test_pred_all = []

        train_start_c = train_start
        test_start_c = test_start

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

            # Fresh model with SAME tuned parameters
            model = self._build_model()

            # fit model
            model.fit(X_train_c, y_train_c)

            y_test_pred = self._predict_model(
                model=model,
                X=X_test_c,
                lags_direct_list=lags_direct_list,
                main_var=main_var,
                block_size=block_size,
                test_or_forecast='Test'
            )

            # append results
            y_test_pred_all.append(y_test_pred)

            # move rolling widnow
            if isinstance(train_period, int):
                train_start_c = train_start_c + pd.Timedelta(days=days_to_retrain)
            test_start_c = test_start_c + pd.Timedelta(days=days_to_retrain)

        y_test_pred_all = pd.concat(y_test_pred_all)

        return y_test_pred_all


    # ------------------------------
    # SAVE
    # ------------------------------
    def save(
        self,
        path: str
    ):
        """
        Save the fitted SklearnPipelineWrapper.

        Directory structure:

            path/
            └── wrapper.pkl
        """

        if self.model_fitted is None:
            raise ValueError(
                "Cannot save an unfitted model. "
                "Please call fit() first."
            )

        os.makedirs(path, exist_ok=True)

        wrapper_path = os.path.join(path,"wrapper.pkl")

        joblib.dump(self, wrapper_path)


    # ------------------------------
    # LOAD
    # ------------------------------
    @classmethod
    def load(
        cls,
        path: str
    ):
        """
        Load a previously saved SklearnPipelineWrapper.

        Expects:

            path/
            └── wrapper.pkl
        """

        wrapper_path = os.path.join(path,"wrapper.pkl")

        if not os.path.exists(wrapper_path):
            raise FileNotFoundError(
                f"Wrapper file not found: {wrapper_path}"
            )

        # --------------------------------------------------
        # Load sklearn wrapper
        # --------------------------------------------------
        wrapper = joblib.load(wrapper_path)

        # Make sure wrapper knows it is fitted
        wrapper.fitted = True

        return wrapper


###-----------------------------------------------------------------------
# END OF FILE



###########################################
#------------------------------------------
# Log model generic
#------------------------------------------
def log_model_generic(
        model,
        model_type:str,
        name:str
    ) -> None:

    """Log model to MLflow based on its type."""
    
    if model_type == "xgboost":
        if isinstance(model, Pipeline):
            import mlflow.sklearn
            mlflow.sklearn.log_model(model, name=name)
        else:
            import mlflow.xgboost
            mlflow.xgboost.log_model(model, name=name)
    elif model_type == "lightgbm":
        if isinstance(model, Pipeline):
            import mlflow.sklearn
            mlflow.sklearn.log_model(model, name=name)
        else:
            import mlflow.lightgbm
            mlflow.lightgbm.log_model(model, name=name)
    elif model_type == "keras":
        #import mlflow.keras
        #mlflow.keras.log_model(model, name=name)
        import mlflow.sklearn
        mlflow.sklearn.log_model(model, name=name)
    elif model_type == "statsmodels":
        import mlflow.pyfunc

        class StatsModelsWrapper(mlflow.pyfunc.PythonModel):
            def __init__(self, model):
                self.model = model

            def predict(self, context, model_input):
                return self.model.predict(model_input)
            
        mlflow.pyfunc.log_model(
            name=name,
            python_model=StatsModelsWrapper(model)
        )

    elif "_wrap" in model_type or model_type == "tft_dart":
        import mlflow
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(tmp_dir)
            mlflow.log_artifacts(tmp_dir, artifact_path=name)

    else:
        import mlflow.sklearn
        mlflow.sklearn.log_model(model, name=name)


#------------------------------------------
# Log metrics, plots, and model to MLflow
#------------------------------------------
def log_mlflow_metrics_plots_and_model(
        model_wrap,
        y_train:pd.DataFrame,
        y_train_pred:pd.DataFrame,
        y_test:pd.DataFrame,
        y_test_pred:pd.DataFrame,
        features_type:str|None=None,
        #features_after_prep:list=[],
        train_start:str|None=None,
        test_start:str|None=None,
        today:str|None=None,
        param_search_space:dict={},
        best_history:dict|None=None,
    )-> None:

    # Extract data from wrapper
    model_type = getattr(model_wrap, "model_type", [])
    features_basic = getattr(model_wrap, "features_basic", [])
    features_after_prep = getattr(model_wrap, "features_after_prep_", [])
    lags_direct_list = getattr(model_wrap, "lags_direct_list", {})
    fwds_direct_list = getattr(model_wrap, "fwds_direct_list", {})
    dummy_for_columns = getattr(model_wrap, "dummy_for_columns", [])
    train_period = getattr(model_wrap, "train_period", None)
    days_to_retrain = getattr(model_wrap, "days_to_retrain", None)

    main_var = getattr(model_wrap, "main_var", None)
    block_size = getattr(model_wrap, "block_size", None)

    model_params = getattr(model_wrap, "model_params", {})
    best_epoch = getattr(model_wrap, "best_epoch_", None)
    formula = getattr(model_wrap, "formula", None)

    #--------------------------------
    # log model config for model registry
    #--------------------------------
    model_config = {
        "main_var":main_var,
        "model_type":model_type,
        "block_size":block_size,
        "model_type": model_type,
        "features":features_basic,
        "lags_direct_list": lags_direct_list,
        "fwds_direct_list": fwds_direct_list,
        "dummy_for_columns": dummy_for_columns,
        "train_period":train_period,
        "train_start":train_start,
        "test_start":test_start,
        "features_after_prepocessing":features_after_prep,
        "days_to_retrain": days_to_retrain
    }

    if formula:
        model_config.update({"formula": formula})

    if model_params:
        model_config.update({"best_params": model_params})

    if best_epoch:
        model_config.update({"n_epoch": best_epoch})

    mlflow.log_dict(model_config, 'model_config.json')

    #--------------------------------
    # log tags for comparing-filter models
    #--------------------------------
    mlflow.set_tag("model_type", model_type) 
    mlflow.set_tag("train_start", train_start) 
    mlflow.set_tag("run_date", today) 
    mlflow.set_tag("features_type", features_type) 

    #--------------------------------
    # log metrics, plots and params
    #--------------------------------
    log_metrics_and_plot(y_train, y_train_pred, y_test, y_test_pred)


    if model_type == "statsmodels_wrap":
        mlflow.log_text(model_wrap.model_fitted.summary().as_text(), "ols_summary.txt")
    elif model_type in ["keras","keras_wrap"]:
        plot_training_history(best_history)
    elif model_type in ["lightgbm", "lightgbm_wrap", "xgboost", "xgboost_wrap"]:
        log_tree_model_feature_importance(
            model=model_wrap.model_fitted.named_steps['model'],
            feature_names=features_after_prep
        )

    #--------------------------------
    # description
    #--------------------------------
    lines = [
            f"Features: {', '.join(features_basic)}",
            f"Training Data Range: {train_start} to {y_train.index[-1]}",
            f"Test Data Range: {y_test.index[0]} to {y_test.index[-1]}"
        ]

    if model_type in ["statsmodels", "statsmodels_wrap"]:
        lines.append(f"Formula: {formula}")
    else:
        lines.append(f"Validation Data Range: 0.2% of train")
        lines.append("Hyperparameter Search Space:")
        lines.append(str(param_search_space))
        lines.append(f"The best hyperparameters found: {model_params}")

    description = "\n".join(lines)

    mlflow.set_tag("mlflow.note.content", description)

    #--------------------------------
    # log model
    #--------------------------------
    log_model_generic(model_wrap, model_type, MODEL_ARTIFACT_NAME)

    return None

###-----------------------------------------------------------------------
# END OF FILE


