

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
