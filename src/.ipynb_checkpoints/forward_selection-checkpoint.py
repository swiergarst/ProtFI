import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from .utils import *
from model_functions import *
from sklearn.feature_selection import SequentialFeatureSelector
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import check_cv
from sklearn.base import is_classifier, clone
import argparse
from tqdm import tqdm
random.seed(7)


def find_feature_ranking(X, y, fs):

    print(f' number of cpus: {fs.n_jobs}')
    n_features = X.shape[1]
    feature_idx_order = np.zeros(n_features)
    scores = np.zeros(n_features)
    
    # sklearn bookkeeping
    cv = check_cv(fs.cv, y, classifier=is_classifier(fs.estimator))
    cloned_estimator = clone(fs.estimator)
    current_mask = np.zeros(shape=n_features, dtype = bool)

    
   
    for i in tqdm(range(n_features)):
        new_feature_idx, new_score = fs._get_best_new_feature_score(
            cloned_estimator, X, y, cv, current_mask)
        
        feature_idx_order[i] = new_feature_idx
        scores[i] = new_score
        current_mask[new_feature_idx] = True

    return feature_idx_order, scores
        
        





def select_features(dset, target, tol = 0.001, find_ranking = False):
    # get the data
    trainset, set2, eids_train, eids_set2, _ = get_data({'dset': dset, 'target': target})
    X = trainset[0]
    # y = trainset[1]
    X_val = set2[0]
    # y_val = set2[1]

    # select the model to use for feature selection
    if target == "frailty" :
        model = LinearRegression()
        coef_folder = "coefs_frail"
        score_folder = "rsquared"
        cols_folder = "set2frail"
        y = trainset[1]
        y_val = set2[1]

    elif target == "mort":
        model = CoxPHSurvivalAnalysis()
        coef_folder = "coefs_mort"
        score_folder = "concordance"
        cols_folder = "set2mort"
        y = np.array([(e1, e2) for e1, e2 in np.vstack((trainset[2], trainset[1])).T], dtype = [("Status", "?"), ("Survival_in_days", "<f8")])
        y_val = np.array([(e1, e2) for e1, e2 in np.vstack((set2[2], set2[1])).T], dtype = [("Status", "?"), ("Survival_in_days", "<f8")])

    else:
        raise ValueError(f'unknown value for model: {model}')

    # create feature selector
    feature_selector = SequentialFeatureSelector(
        model,
        n_features_to_select = "auto",
        direction = "forward",
        n_jobs = -1,
        cv = 5,
        tol = tol)

    
    if find_ranking:
        feature_order, scores = find_feature_ranking(X, y, feature_selector)
        np.save(f'./output_linear/{score_folder}/ffs_ranking_{dset}.npy', scores)
        np.save(f'./output_linear/{cols_folder}/ffs_feature_order_{dset}.npy', feature_order)
        return feature_selector
    else:    
        # do feature selection
        feature_selector.fit(X, y)

        # create a subset of the data based on the selected features
        X_train_subset = feature_selector.transform(X)
        X_val_subset = feature_selector.transform(X_val)
    
        # fit+test based on the selected features 
        feature_names = get_trainsetnames(dset, target)
        feature_selector.estimator.fit(X_train_subset, y)
        out_score, new_col = validate_model(feature_selector.estimator, X_train_subset, y, X_val_subset, y_val, 'en_forward_frailty')
    
        # save feature names + coefs
        bestcoefs = pd.DataFrame(feature_selector.estimator.coef_, index = feature_names[feature_selector.get_support()], columns = ["coef"])
        bestcoefs.to_csv(f'./output_linear/{coef_folder}/ffs_coefs_{dset}_tol{tol}.csv')
    
        # save corresponding scores
        np.save(f"./output_linear/{score_folder}/ffs_{dset}_tol{tol}.npy", out_score)
        new_col.to_csv(f'./output_linear/{cols_folder}/ffs_set2_tol{tol}.csv')
    
        return feature_selector


def main():
    parser = argparse.ArgumentParser(description='Find the best model based on dataset and target.')
    parser.add_argument("--dset", type = str, default = "cmb", help = "dataset to use")
    parser.add_argument("--target", type = str, default = "frailty", help = "label to train on")
    parser.add_argument("--tol", type = float, default = 0.001, help = "threshold for stopping the forward selection")
    parser.add_argument("--ranking", type = int, default = 0, help = "whether to run feature ranking")

    args = parser.parse_args()

    if args.ranking == 0:
        find_ranking = False
    elif args.ranking == 1:
        find_ranking = True
    
    select_features(args.dset,args.target, args.tol, find_ranking = find_ranking)

if __name__ == "__main__":
    main()
