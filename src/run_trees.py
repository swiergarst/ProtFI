import numpy as np
import argparse
from utils import *
import torch
import datetime
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet


def select_model(model_name, args):
    if model_name == "trees":
        if args['target'] == "mort":
            model = RandomSurvivalForest(n_estimators = args['n_trees'], max_depth = args['max_depth'])
        elif args['target'] == "frailty":
            model = GradientBoostingRegressor(n_estimators = args['n_trees'], max_depth = args['max_depth'])
    elif model_name == "elasticnet":
        if args['target'] == "mort":
            model = CoxnetSurvivalAnalysis(alphas = [args['alpha']], l1_ratio = args['l1_ratio'])
        elif args['target'] == "frailty":
            model = ElasticNet(alpha = args['alpha'], l1_ratio = args['l1_ratio'])
    return model



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "survival forest")
    parser.add_argument("--dset", type = str, default = "cmb", help = "dataset to fit on")
    parser.add_argument("--target", type = str, default = "mort", help = "target to train on")
    parser.add_argument("--model", type = str, default = "trees", help = "which model to run")

    parser.add_argument("--var_t", type=float, default=0, help = "column dropping threshold based on variance")
    
    # tree-specific. will be ignored for other models
    parser.add_argument("--n_trees", type = int, default = 10, help = "amount of trees to fit")
    parser.add_argument("--max_depth", type = int, default = 2, help = "maximum tree depth")

    # elasticnet specific. will be ignored otherwise
    parser.add_argument("--l1_ratio", type = float, default = 0.5, help = "l1_ratio for elasticnet")
    parser.add_argument("--alpha", type = float, default = 1, help = "alpha for elasticnet")

    

    
    args = parser.parse_args()

    config = {
            'dset' : args.dset,
            'target' : args.target,
            'n_trees' : args.n_trees,
            'max_depth' : args.max_depth,
            'l1_ratio' : args.l1_ratio,
            'alpha' : args.alpha,
            'var_t' : args.var_t
            }

    train, test, _, _, columns = get_data(config)
       
    wandb.login(key = '483e19e5215d5e164b4cc3f6c3f85d5c3202eabc', force = True)
    wandb_config = {
        "architecture" : args.model,
        "dataset" : args.dset,
        "columns" : columns
    }


    
    r = wandb.init(
        project = "olink-aging-pers",
        config = {**config, **wandb_config})
    


    X_train = train[0]
    X_test = test[0]

    if args.target == "mort":
        y_train = np.array([(e1, e2) for e1, e2 in np.vstack((train[2], train[1])).T], dtype = [("cens", "?"), ("time", "<f8")])
        y_test = np.array([(e1, e2) for e1, e2 in np.vstack((test[2], test[1])).T], dtype = [("cens", "?"), ("time", "<f8")])
    else:
        y_train = train[1]
        y_test = test[1]
    
    model = select_model(args.model, config)
    model.fit(X_train, y_train)
    
    score_train= model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)

    wandb.log({
        "score_train" : score_train,
        "score_test": score_test })

    wandb.finish()