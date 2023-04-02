
from sklearn.linear_model import Lasso, Ridge, BayesianRidge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor





def create_algorithm(model_type, trial, client, child_run, config_data):

    if model_type == 'Lasso':
        lasso_alpha = trial.suggest_float('lasso_alpha', config_data["lasso_alpha"][0], config_data["lasso_alpha"][1])
        lasso_max_iter = trial.suggest_int('lasso_max_iter', config_data["lasso_max_iter"][0], config_data["lasso_max_iter"][1])

        client.log_param(child_run.info.run_id, "lasso_alpha", lasso_alpha)
        client.log_param(child_run.info.run_id, "lasso_max_iter", lasso_max_iter)
        
        model = Lasso(
            alpha=lasso_alpha,
            max_iter=lasso_max_iter,
            random_state=config_data["RANDOM_STATE"]
        )

        
    if model_type == 'Ridge':
        ridge_alpha = trial.suggest_float('ridge_alpha', config_data["ridge_alpha"][0], config_data["ridge_alpha"][1])
        ridge_max_iter = trial.suggest_int('ridge_max_iter', config_data["ridge_max_iter"][0], config_data["ridge_max_iter"][1])
        ridge_solver =  trial.suggest_categorical('ridge_solver', config_data["ridge_solver"])

        client.log_param(child_run.info.run_id, "ridge_alpha", ridge_alpha)
        client.log_param(child_run.info.run_id, "ridge_max_iter", ridge_max_iter)
        client.log_param(child_run.info.run_id, "ridge_solver", ridge_solver)
        
        model = Ridge(
            alpha=ridge_alpha,
            max_iter=ridge_max_iter,
            solver=ridge_solver,
            random_state=config_data["RANDOM_STATE"]
        )


    if model_type == 'BayesianRidge':
        bayesianRidge_n_iter = trial.suggest_int('bayesianRidge_n_iter', config_data["bayesianRidge_n_iter"][0], config_data["bayesianRidge_n_iter"][1])
        bayesianRidge_alpha_init = trial.suggest_float('bayesianRidge_alpha_init', config_data["bayesianRidge_alpha_init"][0], config_data["bayesianRidge_alpha_init"][1])
        bayesianRidge_alpha_1 = trial.suggest_float('bayesianRidge_alpha_1', config_data["bayesianRidge_alpha_1"][0], config_data["bayesianRidge_alpha_1"][1])
        bayesianRidge_alpha_2 = trial.suggest_float('bayesianRidge_alpha_2', config_data["bayesianRidge_alpha_2"][0], config_data["bayesianRidge_alpha_2"][1])
        bayesianRidge_lambda_init = trial.suggest_float('bayesianRidge_lambda_init', config_data["bayesianRidge_lambda_init"][0], config_data["bayesianRidge_lambda_init"][1])
        bayesianRidge_lambda_1 = trial.suggest_float('bayesianRidge_lambda_1', config_data["bayesianRidge_lambda_1"][0], config_data["bayesianRidge_lambda_1"][1])
        bayesianRidge_lambda_2 = trial.suggest_float('bayesianRidge_lambda_2', config_data["bayesianRidge_lambda_2"][0], config_data["bayesianRidge_lambda_2"][1])
        

        client.log_param(child_run.info.run_id, "bayesianRidge_n_iter", bayesianRidge_n_iter)
        client.log_param(child_run.info.run_id, "bayesianRidge_alpha_init", bayesianRidge_alpha_init)
        client.log_param(child_run.info.run_id, "bayesianRidge_alpha_1", bayesianRidge_alpha_1)
        client.log_param(child_run.info.run_id, "bayesianRidge_alpha_2", bayesianRidge_alpha_2)
        client.log_param(child_run.info.run_id, "bayesianRidge_lambda_init", bayesianRidge_lambda_init)
        client.log_param(child_run.info.run_id, "bayesianRidge_lambda_1", bayesianRidge_lambda_1)
        client.log_param(child_run.info.run_id, "bayesianRidge_lambda_2", bayesianRidge_lambda_2)
        
        model = BayesianRidge(
            n_iter=bayesianRidge_n_iter,
            alpha_init=bayesianRidge_alpha_init,
            alpha_1=bayesianRidge_alpha_1,
            alpha_2=bayesianRidge_alpha_2,
            lambda_init=bayesianRidge_lambda_init,
            lambda_1=bayesianRidge_lambda_1,
            lambda_2=bayesianRidge_lambda_2
        )


    if model_type == 'ElasticNet':
        elasticNet_max_iter = trial.suggest_int('elasticNet_max_iter', config_data["elasticNet_max_iter"][0], config_data["elasticNet_max_iter"][1])
        elasticNet_alpha = trial.suggest_float('elasticNet_alpha', config_data["elasticNet_alpha"][0], config_data["elasticNet_alpha"][1])
        elasticNet_l1_ratio = trial.suggest_float('elasticNet_l1_ratio', config_data["elasticNet_l1_ratio"][0], config_data["elasticNet_l1_ratio"][1])
        
        client.log_param(child_run.info.run_id, "elasticNet_max_iter", elasticNet_max_iter)
        client.log_param(child_run.info.run_id, "elasticNet_alpha", elasticNet_alpha)
        client.log_param(child_run.info.run_id, "elasticNet_l1_ratio", elasticNet_l1_ratio)
        
        model = ElasticNet(
            max_iter=elasticNet_max_iter,
            alpha=elasticNet_alpha,
            l1_ratio = elasticNet_l1_ratio,
            random_state=config_data["RANDOM_STATE"]
        )


    if model_type == 'GradientBoostingRegressor':
        gbr_n_estimators = trial.suggest_int('gbr_n_estimators', config_data["gbr_n_estimators"][0], config_data["gbr_n_estimators"][1])
        gbr_learning_rate = trial.suggest_float('gbr_learning_rate', config_data["gbr_learning_rate"][0], config_data["gbr_learning_rate"][1], log=True)
        gbr_subsample = trial.suggest_float('gbr_subsample', config_data["gbr_subsample"][0], config_data["gbr_subsample"][1])
        gbr_min_samples_split = trial.suggest_float('gbr_min_samples_split', config_data["gbr_min_samples_split"][0], config_data["gbr_min_samples_split"][1])
        gbr_min_samples_leaf = trial.suggest_float('gbr_min_samples_leaf', config_data["gbr_min_samples_leaf"][0], config_data["gbr_min_samples_leaf"][1])
        gbr_max_depth = trial.suggest_int('gbr_max_depth', config_data["gbr_max_depth"][0], config_data["gbr_max_depth"][1])
        gbr_max_features = trial.suggest_float('gbr_max_features', config_data["gbr_max_features"][0], config_data["gbr_max_features"][1])
        gbr_alpha = trial.suggest_float('gbr_alpha', config_data["gbr_alpha"][0], config_data["gbr_alpha"][1])
        
        client.log_param(child_run.info.run_id, "gbr_n_estimators", gbr_n_estimators)
        client.log_param(child_run.info.run_id, "gbr_learning_rate", gbr_learning_rate)
        client.log_param(child_run.info.run_id, "gbr_subsample", gbr_subsample)
        client.log_param(child_run.info.run_id, "gbr_min_samples_split", gbr_min_samples_split)
        client.log_param(child_run.info.run_id, "gbr_min_samples_leaf", gbr_min_samples_leaf)
        client.log_param(child_run.info.run_id, "gbr_max_depth", gbr_max_depth)
        client.log_param(child_run.info.run_id, "gbr_max_features", gbr_max_features)
        client.log_param(child_run.info.run_id, "gbr_alpha", gbr_alpha)
        
        model = GradientBoostingRegressor(
            n_estimators=gbr_n_estimators,
            learning_rate = gbr_learning_rate,
            subsample = gbr_subsample,
            min_samples_split = gbr_min_samples_split,
            min_samples_leaf = gbr_min_samples_leaf,
            max_depth = gbr_max_depth,
            max_features = gbr_max_features,
            alpha = gbr_alpha,
            random_state=config_data["RANDOM_STATE"]
        )


    if model_type == 'RandomForestRegressor':
        rfr_n_estimators = trial.suggest_int('rfr_n_estimators', config_data["rfr_n_estimators"][0], config_data["rfr_n_estimators"][1])
        rfr_min_samples_split = trial.suggest_float('rfr_min_samples_split', config_data["rfr_min_samples_split"][0], config_data["rfr_min_samples_split"][1])
        rfr_min_samples_leaf = trial.suggest_float('rfr_min_samples_leaf', config_data["rfr_min_samples_leaf"][0], config_data["rfr_min_samples_leaf"][1])
        rfr_max_depth = trial.suggest_int('rfr_max_depth', config_data["rfr_max_depth"][0], config_data["rfr_max_depth"][1])
        rfr_max_features = trial.suggest_float('rfr_max_features', config_data["rfr_max_features"][0], config_data["rfr_max_features"][1])
        rfr_max_leaf_nodes = trial.suggest_int('rfr_max_leaf_nodes', config_data["rfr_max_leaf_nodes"][0], config_data["rfr_max_leaf_nodes"][1])
        
        client.log_param(child_run.info.run_id, "rfr_n_estimators", rfr_n_estimators)
        client.log_param(child_run.info.run_id, "rfr_min_samples_split", rfr_min_samples_split)
        client.log_param(child_run.info.run_id, "rfr_min_samples_leaf", rfr_min_samples_leaf)
        client.log_param(child_run.info.run_id, "rfr_max_depth", rfr_max_depth)
        client.log_param(child_run.info.run_id, "rfr_max_features", rfr_max_features)
        client.log_param(child_run.info.run_id, "rfr_max_leaf_nodes", rfr_max_leaf_nodes)
        
        model = RandomForestRegressor(
            n_estimators=rfr_n_estimators,
            min_samples_split = rfr_min_samples_split,
            min_samples_leaf = rfr_min_samples_leaf,
            max_depth = rfr_max_depth,
            max_features = rfr_max_features,
            max_leaf_nodes = rfr_max_leaf_nodes,
            n_jobs = -1,
            random_state=config_data["RANDOM_STATE"]
        )

    client.log_param(child_run.info.run_id, "algo", model.__class__.__name__)
    return model