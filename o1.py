
#------------------------------------------
# convert pd.Datarfame to pd.Series
#------------------------------------------
def _to_series(
    y: pd.Series|pd.DataFrame,
) -> pd.Series:

    # DataFrame -> Series
    if isinstance(y, pd.DataFrame):

        if y.shape[1] != 1:
            raise ValueError(
                f"{name} must contain exactly one column. "
                f"Got {y.shape[1]} columns."
            )
        y = y.iloc[:, 0]

    # Check Series
    elif not isinstance(y, pd.Series):

        raise TypeError(
            f"{name} must be a pandas Series or DataFrame. "
            f"Got {type(y).__name__}."
        )

    return y

#------------------------------------------
# Log metrics and plot predictions
#------------------------------------------

def log_metrics_and_plot(
        y_train: pd.Series|pd.DataFrame,
        y_train_pred: pd.Series|pd.DataFrame,
        y_test: pd.Series|pd.DataFrame,
        y_test_pred: pd.Series|pd.DataFrame
    )-> None:

    """Log metrics and plot predictions to MLflow."""

    y_train = _to_series(y_train)
    y_train_pred = _to_series(y_train_pred)
    y_test = _to_series(y_test)
    y_test_pred = _to_series(y_test_pred)
    
    # compute metrics
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
#################################



#------------------------------------------
# Log metrics, plots, and model to MLflow
#------------------------------------------
def log_mlflow_metrics_plots_and_model(
        cfg,
        model,
        y_train:pd.DataFrame,
        y_train_pred:pd.DataFrame,
        y_test:pd.DataFrame,
        y_test_pred:pd.DataFrame,
        features_type:str|None=None,
        features_basic:list=[],
        features_after_prep:list=[],
        formula:str|None=None,
        lags_direct_list:dict={},
        fwds_direct_list:dict={},
        dummy_for_columns:list=[],
        train_start:str|None=None,
        test_start:str|None=None,
        today:str|None=None,
        param_search_space:dict={},
        best_params:dict={},
        best_epoch_for_retrain:int|None=None,
        best_history:dict|None=None,
        train_period:str|int|None=None,
        days_to_retrain:int|None=None
    )-> None:

    model_type = cfg["type"]

    #--------------------------------
    # log model config for model registry
    #--------------------------------
    model_config = {
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

    if best_params:
        model_config.update({"best_params": best_params})

    if best_epoch_for_retrain:
        model_config.update({"n_epoch": best_epoch_for_retrain})

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


    if model_type == "statsmodels":
        mlflow.log_text(model.summary().as_text(), "ols_summary.txt")
    elif model_type == "statsmodels_wrap":
        mlflow.log_text(model.model_fitted.summary().as_text(), "ols_summary.txt")
    elif model_type == "keras":
        plot_training_history(best_history)
    elif model_type in ["lightgbm", "lightgbm_wrap", "xgboost", "xgboost_wrap"]:
        log_tree_model_feature_importance(
            model=model.model_fitted.named_steps['model'],
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
        lines.append(f"The best hyperparameters found: {best_params}")

    description = "\n".join(lines)

    mlflow.set_tag("mlflow.note.content", description)

    #--------------------------------
    # log model
    #--------------------------------
    log_model_generic(model, model_type, MODEL_ARTIFACT_NAME)

    return None

####################

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
                horizon_forecast = block_size
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
            days_to_retrain:int=7,
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


###-----------------------------------------------------------------------
# END OF FILE

import numpy as np
import pandas as pd

import os
import joblib

import statsmodels.formula.api as smf

from src.features.features_utils import remove_var_if_data_to_short

from src.modelling.models_utils.predict_with_lags import (
    clean_lag_bloks_for_dependent_var,
    make_ts_with_lags_forecast
)



class StatsmodelWrapper:

    def __init__(self, formula, **kwargs):

        self.formula = formula
        # Fitted estimator
        self.model_fitted = None
        # Fitted flag
        self.fitted = False

        self.main_var = kwargs.get('main_var', None)

        self.lags_direct_list = kwargs.get('lags_direct_list', {}).copy()
        self.train_period = kwargs.get('train_period', None)
        self.block_size = kwargs.get('block_size', 24)
        self.features_basic = kwargs.get('features_basic', None)

        self.fit_params = kwargs.get('fit_params', {}).copy()
        self.seasonal_vars_to_check = kwargs.get('seasonal_vars_to_check', ['week', 'month'])
        self.numbers_obs_allowed_for_seasonal = kwargs.get('numbers_obs_allowed_for_seasonal', None)

    #------------------------------
    # BUID MODEL
    #------------------------------
    def _build_and_fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        fit_params: dict | None = None
    ):
        """
        Build a fresh formula-based model from X/y and fit it.

        If the training set is too short (< numbers_obs_allowed_for_seasonal),
        seasonal variables (e.g. 'week', 'month') are dropped from the
        formula before fitting.
        """
        params = self.fit_params if fit_params is None else fit_params

        # ------------------------------------------------------------
        # Adjust formula for short training windows
        # ------------------------------------------------------------
        formula = self.formula

        if (
            self.numbers_obs_allowed_for_seasonal is not None
            and len(X) <= self.numbers_obs_allowed_for_seasonal
        ):
            formula = remove_var_if_data_to_short(
                self.seasonal_vars_to_check,
                'stat_formula',
                formula=formula
            )

        # Track which formula was actually used for this fit
        self.formula_used_ = formula

        model = smf.ols(
            formula=formula,
            data= pd.concat([X, y], axis=1)
        ).fit(**params)

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

        model = self._build_and_fit(X, y, fit_params=kwargs)
        self.model_fitted = model
        self.fitted = True

        design_info = model.model.data.design_info
        used_features = list({
                factor.name().replace("C(", "").replace(")", "") 
                for term in design_info.terms 
                for factor in term.factors
                if factor.name() != "Intercept"
            })
        self.used_features_ = used_features

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
        Predict with a fitted model, optionally using lags for recursive/direct forecasting.
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
                horizon_forecast = block_size
            )

            if return_X_test:
                return y_test_pred_df, X_test_zeros
            else:
                return y_test_pred_df
            
        else:
            # NORMAL PREDICTION
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
            y_train_fitted,
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
            days_to_retrain:int=7,
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

            # Fresh model, built + fitted from this window's data
            model = self._build_and_fit(X_train_c, y_train_c)

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

import numpy as np
import pandas as pd
from sklearn.base import clone

import os
import joblib
import random

import keras

from sklearn.model_selection import (
    RandomizedSearchCV, GridSearchCV, TimeSeriesSplit, ParameterSampler
)

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.modelling.models_utils.predict_with_lags import (
    clean_lag_bloks_for_dependent_var,
    make_ts_with_lags_forecast
)




class KerasPipelineWrapper:
    """
    Wrapper for sklearn Pipelines containing a SciKeras KerasRegressor.

    """

    def __init__(self, model, **kwargs):
        # model_template is a sklearn pipeline with a SciKeras KerasRegressor as the final step
        self.model_template = model
         # Hyperparameters
        self.model_params = kwargs.get('model_params', {}).copy()
        # Fitted model
        self.model_fitted = None
        # Fitted flag
        self.fitted = False
        # keras history
        self.history_ = None

        # additional attributes
        self.main_var = kwargs.get('main_var', None)
        self.lags_direct_list = kwargs.get('lags_direct_list', {}).copy()
        self.dummy_columns = kwargs.get('dummy_columns', {}).copy()
        self.train_period = kwargs.get('train_period', None)
        self.block_size = kwargs.get('block_size', 24)
        self.features_basic = kwargs.get('features_basic', None)

    #------------------------------
    # BUID MODEL
    #------------------------------
    def _build_model(self, model_params=None):
        """
        Build a completely fresh piepline with Keras/SciKeras estimator.

        If model_template is callable:
            model_template() is called.

        Otherwise:
            sklearn.clone() is used.

        self.model_params are then applied, either from self.model_params or from the supplied model_params argument.
        """

        # Build fresh model
        if callable(self.model_template):
            model = self.model_template()
        else:
            model = clone(self.model_template)

        # Parameters to apply
        params = self.model_params if model_params is None else model_params

        if params:
            model.set_params(**params)

        return model


    #------------------------------
    # FIT
    #------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        **kwargs
    ):
        """
        Build and fit a fresh Keras/SciKeras model.

        kwargs are passed directly to model.fit().

        This allows, parameters in form:
            regressor__model__epochs
            regressor__model__callbacks
            regressor__model__validation_split
            regressor__model__shuffle
            regressor__model__verbose

        'model' here is a sklearn pipeline.
        """

        model = self._build_model()
        model.fit(X,y,**kwargs)

        self.model_fitted = model
        self.fitted = True

        # --------------------------------------------------------
        # Extract Keras history
        # --------------------------------------------------------
        self.history_ = (
            self._get_history(
                self.model_fitted
            )
        )

        return self

    #------------------------------
    # GET KERAS HISTORY
    #------------------------------
    def _get_history(
        self,
        model
    ):
        """
        Extract Keras training history from a fitted
        SciKeras estimator.

        Returns
        -------
        dict or None
        """

        # Direct SciKeras estimator
        if hasattr( model,"history_"):
            return model.history_

        # TransformedTargetRegressor
        if hasattr(model,"regressor_"):
            regressor = model.regressor_

            if hasattr(regressor,"named_steps"):

                # Find final Keras model
                for step_name, step in (regressor.named_steps.items()):

                    if hasattr(step,"history_"):
                        return step.history_

        return None


    #------------------------------
    # PREDICT MODEL
    #------------------------------
    def _predict_model(
        self,
        model,
        X: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        Predict using the supplied fitted model.

        The model is explicitly supplied so that rolling
        evaluation can use a fresh fitted network.

        Supports:
            - normal prediction
            - lag-based recursive/direct forecasting
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
                horizon_forecast = block_size
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
        """
        Predict using self.model_fitted.
        """

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
    # RANDOM PARAMETER SEARCH
    #------------------------------
    def fit_random_search(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        **kwargs
    ):
        """
        Manual randomized parameter search for Keras.

        This is preferable to RandomizedSearchCV when you want:

            EarlyStopping
            validation_split
            best validation epoch
            custom Keras training logic

        Parameters
        ----------
        param_grid : dict
            SciKeras parameter distributions.

        n_iter : int
            Number of parameter combinations.

        fit_kwargs : dict
            Arguments passed to model.fit().

        Example
        -------
        fit_kwargs = {
            "regressor__model__epochs": 300,
            "regressor__model__callbacks": [early_stop],
            "regressor__model__validation_split": 0.2,
            "regressor__model__shuffle": False,
            "regressor__model__verbose": 0,
        }
        """

        # Parameters
        param_grid = kwargs.get('param_grid', {})
        n_iter = kwargs.get('n_iter', 10)
        random_state = kwargs.get('random_state', 42)
        fit_kwargs = kwargs.get('fit_kwargs', {}).copy()
        retrain_on_full_data = kwargs.get('retrain_on_full_data', False)

        early_stopping_kwargs = kwargs.get(
            "early_stopping",
            {
                "monitor": "val_loss",
                "patience": 15,
                "restore_best_weights": True,
            }
        )

        # Reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        random.seed(random_state)

        # Generate parameter combinations
        param_list = list(
            ParameterSampler(
                param_grid,
                n_iter=n_iter,
                random_state=random_state
            )
        )

        # Results
        best_score = np.inf
        best_model = None
        best_params = None
        best_history = None
        best_epoch = None

        #------------------------------------------
        # SEARCH LOOP
        #------------------------------------------
        for params in param_list:

            early_stop = EarlyStopping(
                    **early_stopping_kwargs
                )

            # Build COMPLETELY FRESH model with candidate parameters
            tf.keras.backend.clear_session()
            model = self._build_model(
                model_params=params
            )

            # Fit parameters
            search_fit_kwargs = fit_kwargs.copy()

            # Add EarlyStopping
            search_fit_kwargs["model__callbacks"] = [early_stop]

            # Fit candidate
            model.fit(
                X,
                y,
                **search_fit_kwargs
            )

            # extract history
            history = self._get_history(model)

            if history is None:
                raise ValueError("Could not extract Keras history.")

            if "val_loss" not in history:
                raise ValueError(
                    "Random search requires validation loss. "
                    "Provide validation_split or validation_data."
                )

            val_losses = np.asarray(history["val_loss"])

            candidate_score = float(np.min(val_losses))
            candidate_best_epoch = int(np.argmin(val_losses)) + 1

            # Save best candidate
            if candidate_score < best_score:
                best_score = candidate_score
                best_model = model
                best_params = params.copy()
                best_history = history
                best_epoch = candidate_best_epoch

        # Make sure search succeeded
        # ------------------------------------------------------------
        if best_model is None:
            raise RuntimeError(
                "Random search did not produce a valid model."
            )

        # ------------------------------------------------------------
        # Store best search result
        # ------------------------------------------------------------
        self.model_params = best_params
        self.best_epoch_ = best_epoch
        self.best_score_ = best_score

        if retrain_on_full_data:

            tf.keras.backend.clear_session()

            final_model = self._build_model(
                model_params=best_params
            )

            # Copy fit parameters
            full_fit_kwargs = fit_kwargs.copy()

            # Remove validation-specific arguments
            full_fit_kwargs.pop("model__validation_split", None)
            full_fit_kwargs.pop("model__validation_data", None)
            full_fit_kwargs.pop("model__callbacks", None)

            # Train using epoch selected during search
            full_fit_kwargs["model__epochs"] = best_epoch

            # Train on complete dataset
            final_model.fit(
                X,
                y,
                **full_fit_kwargs
            )

            # Store final model - model i stred during fit
            self.model_fitted = final_model

        else:
            # Use model from search
            self.model_fitted = best_model

        # History  comes from the search
        self.history_ = best_history

        self.fitted = True

        return self

    
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
            days_to_retrain:int=1,
            epochs: int | None = None,
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

        # Keras fit parameters
        fit_kwargs = kwargs.get("fit_kwargs",{})

        if epochs is not None:
            window_epochs = epochs
        elif hasattr(self, "best_epoch_"):
            window_epochs = self.best_epoch_
        else:
            raise ValueError(
                "No epochs specified. Provide `epochs=` or "
                "run fit_random_search() first."
            )

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
            tf.keras.backend.clear_session()
            model = self._build_model()

            # add fit params
            window_fit_kwargs = fit_kwargs.copy()
            window_fit_kwargs["model__epochs"] = window_epochs

            # fit model
            model.fit(X_train_c, y_train_c, **window_fit_kwargs)

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
        Save the fitted KerasPipelineWrapper.

        The fitted sklearn/SciKeras pipeline is saved with joblib,
        while the underlying Keras model is saved separately using
        Keras native serialization.

        Directory structure:

            path/
            ├── wrapper.pkl
            └── keras_model.keras
        """

        if self.model_fitted is None:
            raise ValueError(
                "Cannot save an unfitted model. "
                "Please call fit() or fit_random_search() first."
            )

        os.makedirs(path, exist_ok=True)

        # --------------------------------------------------
        # Locate fitted SciKeras estimator
        # --------------------------------------------------
        try:
            keras_regressor = (
                self.model_fitted
                .regressor_
                .named_steps["model"]
            )
        except AttributeError as e:
            raise TypeError(
                "Expected model_fitted to have the structure: "
                "TransformedTargetRegressor -> Pipeline -> "
                "KerasRegressor."
            ) from e

        # --------------------------------------------------
        # Get underlying Keras model
        # --------------------------------------------------
        if not hasattr(keras_regressor, "model_"):
            raise ValueError(
                "The SciKeras KerasRegressor does not contain "
                "a fitted model_."
            )

        keras_model = keras_regressor.model_

        if keras_model is None:
            raise ValueError(
                "Underlying Keras model is None."
            )

        # --------------------------------------------------
        # Save Keras model separately
        # --------------------------------------------------
        keras_model_path = os.path.join(
            path,
            "keras_model.keras"
        )

        keras_model.save(
            keras_model_path
        )

        # --------------------------------------------------
        # Temporarily remove Keras model from SciKeras
        # --------------------------------------------------
        keras_regressor.model_ = None

        try:
            joblib.dump(
                self,
                os.path.join(
                    path,
                    "wrapper.pkl"
                )
            )

        finally:
            # Restore model in memory
            keras_regressor.model_ = keras_model


    # ------------------------------
    # LOAD
    # ------------------------------
    @classmethod
    def load(
        cls,
        path: str
    ):
        """
        Load a previously saved KerasPipelineWrapper.

        Expects:

            path/
            ├── wrapper.pkl
            └── keras_model.keras
        """

        wrapper_path = os.path.join(
            path,
            "wrapper.pkl"
        )

        keras_model_path = os.path.join(
            path,
            "keras_model.keras"
        )

        if not os.path.exists(wrapper_path):
            raise FileNotFoundError(
                f"Wrapper file not found: {wrapper_path}"
            )

        if not os.path.exists(keras_model_path):
            raise FileNotFoundError(
                f"Keras model file not found: {keras_model_path}"
            )

        # --------------------------------------------------
        # Load sklearn wrapper
        # --------------------------------------------------
        wrapper = joblib.load(
            wrapper_path
        )

        # --------------------------------------------------
        # Load Keras model
        # --------------------------------------------------
        keras_model = keras.models.load_model(
            keras_model_path
        )

        # --------------------------------------------------
        # Locate fitted SciKeras estimator
        # --------------------------------------------------
        try:
            keras_regressor = (
                wrapper.model_fitted
                .regressor_
                .named_steps["model"]
            )
        except AttributeError as e:
            raise TypeError(
                "Loaded wrapper does not have the expected "
                "structure: TransformedTargetRegressor -> "
                "Pipeline -> KerasRegressor."
            ) from e

        # --------------------------------------------------
        # Put Keras model back into SciKeras
        # --------------------------------------------------
        keras_regressor.model_ = keras_model

        # Make sure wrapper knows it is fitted
        wrapper.fitted = True

        return wrapper

###-----------------------------------------------------------------------
# END OF FILE

#---------------------------------------
                # inisiate pipeline
                #--------------------------------------- 
                model_lightgbm = SklearnPipelineWrapper(
                    model = pipe_lightgbm,
                    main_var=main_var,
                    lags_direct_list=lags_direct_list,
                    block_size=24,
                    train_period=start_set_dates['train_period'],
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
                    days_to_retrain = cfg["params"]['retrain_days_nr'],
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
                            train_period = start_set_dates['train_period'],
                            days_to_retrain=cfg["params"]['days_to_retrain']
                        )
