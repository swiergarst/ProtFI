#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import random
import argparse
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sksurv.linear_model import CoxnetSurvivalAnalysis as CoxPH
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.utils import *

random.seed(7)

def optimize_model(X, y, model, l1_ratios, trainset_names, cv_splits=5):
    best_coefs = pd.DataFrame(0, index=trainset_names, columns=["coefficient"])
    scores = np.zeros((len(l1_ratios), 2))
    best_score = -np.inf if model == CoxPH else np.inf
    best_model = None

    for i, l1_ratio in enumerate(l1_ratios):
        if model == CoxPH:
            run_first = CoxPH(l1_ratio=l1_ratio) #Experiment alpha_min_ratio used to be 0.01
        else:
            run_first = ElasticNetCV(l1_ratio=l1_ratio, cv=cv_splits, max_iter=100000)
        
        run_first.fit(X, y)
        if model == CoxPH:
            alphas = run_first.alphas_
            gcv = GridSearchCV(
                CoxPH(l1_ratio=l1_ratio),
                param_grid={'alphas': [[v] for v in alphas]},
                cv=KFold(n_splits=cv_splits, shuffle=True, random_state=0),
                error_score=0.5,
                n_jobs=1,
            ).fit(X, y)
            score = gcv.cv_results_["mean_test_score"][np.where(gcv.cv_results_["rank_test_score"] == 1)[0][0]]
            scores[i] = [gcv.best_params_['alphas'][0], score]
        else:
            y_pred = run_first.predict(X=X)
            score = mean_squared_error(y_true=y, y_pred=y_pred)
            scores[i] = [run_first.alpha_, score]

        if (model == CoxPH and score > best_score) or (model != CoxPH and score < best_score):
            best_score = score
            best_model = run_first if model != CoxPH else gcv.best_estimator_
            best_coefs = pd.DataFrame(best_model.coef_, index=trainset_names, columns=["coefficient"])

    best_index = np.argmax(scores[:, 1]) if model == CoxPH else np.argmin(scores[:, 1])
    final_model = model()
    if model == CoxPH:
        final_model.set_params(alphas=[scores[best_index, 0]], l1_ratio=l1_ratios[best_index])
    else:
        final_model.set_params(alpha=scores[best_index, 0], l1_ratio=l1_ratios[best_index])
    final_model.fit(X, y)

    return final_model, best_coefs, scores

def validate_model(model, X_train, y_train, X_val, y_val, colname):
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    predictions = pd.DataFrame(model.predict(X_val), columns=[colname])
    return (train_score, val_score), predictions

def find_best_coxph_model(trainset, trainset_names, set2, colname, quick=False, bootstrap=False, dset='cmb', combine=False):
    #make dataset in right format
    X, y_event, y_time = trainset[0], trainset[2], trainset[1]
    y = np.array([(e, t) for e, t in zip(y_event, y_time)], dtype=[("Status", "?"), ("Survival_in_days", "<f8")])

    if quick == True:
        l1_ratios = np.arange(0.1, 1, 0.1)
    else:
        l1_ratios = np.arange(0.1, 1, 0.01)

    if bootstrap | combine:
        scores_cmb = np.load(f"output_linear/scores_mort/scores_mort_{dset}.npy")
        best_index = np.argmax(scores_cmb[:, 1])
        alphas=[scores_cmb[best_index, 0]]
        l1_ratio=l1_ratios[best_index]
        best_model = CoxPH(l1_ratio=l1_ratio, alphas=alphas).fit(X,y)
        best_coefs =  pd.DataFrame(best_model.coef_, index=trainset_names, columns=["coefficient"])
        scores = [alphas, best_model.score(X,y)]
    else:
        best_model, best_coefs, scores = optimize_model(X, y, CoxPH, l1_ratios, trainset_names)
    X_val, y_event_val, y_time_val = set2[0], set2[2], set2[1]
    y_val = np.array([(e, t) for e, t in zip(y_event_val, y_time_val)], dtype=[("Status", "?"), ("Survival_in_days", "<f8")])
    C_save, value = validate_model(best_model, X, y, X_val, y_val, colname)

    return best_model, best_coefs, scores, C_save, value

def find_best_lm_model(trainset, trainset_names, set2, colname, quick=False, bootstrap=False, dset='cmb', combine=False):
    #make dataset
    X, y = trainset[0], trainset[1]
    
    if quick == True:
        l1_ratios = np.arange(0.1, 1, 0.1)
    else:
        l1_ratios = np.arange(0.1, 1, 0.01)

    if bootstrap | combine:
        scores_cmb = np.load(f"output_linear/scores_frail/scores_frailty_{dset}.npy")
        best_index = np.argmin(scores_cmb[:, 1])
        alphas=scores_cmb[best_index, 0]
        l1_ratio=l1_ratios[best_index]
        best_model = ElasticNet(l1_ratio=l1_ratio, alpha=alphas).fit(X,y)
        best_coefs =  pd.DataFrame(best_model.coef_, index=trainset_names, columns=["coefficient"])
        scores = [alphas, best_model.score(X,y)]
    else:
        best_model, best_coefs, scores = optimize_model(X, y, ElasticNet, l1_ratios, trainset_names)
    X_val, y_val = set2[0], set2[1]
    R2_save, value = validate_model(best_model, X, y, X_val, y_val, colname)

    return best_coefs, scores, R2_save, value

def get_trainsetnames(dset, target):
    # Define file paths based on dset and target
    file_paths = {
        'allprot': 'Data/Processed/Full/full_train.csv',
        'cmb': 'Data/Processed/Combined/combined_train_cmb.csv',
        'cmb_sub': 'Data/Processed/Combined/combined_train_cmb.csv',
        'cmb_morefrail': 'Data/morefrail_training.csv',
        'cmb_4050': 'Data/cmb_train_4050.csv',
        'cmb_5060': 'Data/cmb_train_5060.csv',
        'cmb_6070': 'Data/cmb_train_6070.csv',
        'cmb_met': 'Data/Processed/MultiOmics/proteins_metabolites_train.csv',
        'cmb_met_pca': 'Data/Processed/MultiOmics/proteins_metabolites_pca_train.csv',
        'cmb_met_pca20': 'Data/Processed/MultiOmics/proteins_metabolites_pca20_train.csv',
        'cmb_mh': 'Data/Processed/MultiOmics/proteins_mh_train.csv',
        'cmb_met_ajive10': 'Data/Processed/MultiOmics/ajive10_prot_met_train.csv',
        'cmb_met_ajive20': 'Data/Processed/MultiOmics/ajive20_prot_met_train.csv',
        'allprot_ffs_frail': 'output_linear/coefs_frail/ffs_coefs_allprot_tol0.001.csv',
        'allprot_ffs_mort': 'output_linear/coefs_mort/ffs_coefs_allprot_tol0.001.csv',
        'cmb_ffs_frail': 'output_linear/coefs_frail/ffs_coefs_cmb_tol0.001.csv',
        'cmb_ffs_mort': 'output_linear/coefs_mort/ffs_coefs_cmb_tol0.001.csv',
        'cmb_met_ffs_frail': 'output_linear/coefs_frail/ffs_coefs_cmb_met_tol0.001.csv',
        'cmb_met_ffs_mort': 'output_linear/coefs_mort/ffs_coefs_cmb_met_tol0.001.csv'
    }

    # Check if dataset exists in file paths
    if dset not in file_paths and f"{dset}_frail" not in file_paths:
        raise ValueError(f"Check '{dset}' as it is not available.")

    # Load column names based on dset and target
    if 'ffs' in dset:
        # Select the correct file path based on the target
        if target == 'frailty':
            names = pd.read_csv(file_paths[f'{dset}_frail'])['Unnamed: 0'].values
        elif target == 'mort':
            names = pd.read_csv(file_paths[f'{dset}_mort'])['Unnamed: 0'].values
        else:
            raise ValueError(f"Target '{target}' not recognized.")
        
        names = pd.Index(names)
    else:
        # For non-ffs datasets, just take column names
        names = pd.read_csv(file_paths[dset], index_col='eid').columns

    # Add 'age' for the 'frailty' target if it's not selected
    if target == 'frailty' and 'ffs' not in dset:
        if 'age' not in names:
            names_use = names.append(pd.Index(['age']))
        else:
            names_use = names
    else:
        names_use = names

    return names_use

    

def find_best_model(dset, target, bootstrap=False, quickbootstrap=False, combine=False):
    # Retrieve data
    if bootstrap | quickbootstrap: 
        combine=True
    if quickbootstrap:
        bootstrap=True
    if combine==False:
        trainset, set2, set3, eids_train, eids_set2, eids_set3,  _ = get_data({'dset': dset, 'target': target})
    else:
        trainset, set2, eids_train, eids_set2 = get_data({'dset': dset, 'target': target, 'combine_sets':True}) 
    trainset_names = get_trainsetnames(dset, target)
    colname = f'en_{target}_{dset}'
     # Get the number of samples from the 'cmb_met' dataset
   # cmb_morefrail_data, _, _, _, _ ,_, _ = get_data({'dset': 'cmb_morefrail', 'target': target})
   # cmb_met_data, _, _, _, _ ,_,_ = get_data({'dset': 'cmb_met', 'target': target})
    
    #num_samples = cmb_morefrail_data[0].shape[0] if bootstrap_morefrail else cmb_met_data[0].shape[0] # Get the number of participants if bootstrapping on more frail take different number

    # Initialize storage for results if using bootstrap
    combined_coefs = []
    combined_scores = []
    combined_new_col = []
    combined_r2_or_C = []

    # Perform bootstrap sampling if requested
    if bootstrap:
        if quickbootstrap:
            numberrounds=2
        else:
            numberrounds=100
        based_on = 'own_size'
        num_samples = trainset[0].shape[0]
        for i in range(numberrounds):
            # Sample the training data with replacement using the number of samples in full data
            np.random.seed(i)  # Set the seed for reproducibility
            indices = np.random.choice(trainset[0].shape[0], size=num_samples, replace=True)
            sampled_trainset_X = trainset[0][indices]
            sampled_trainset_y = trainset[1][indices]
            if target == 'mort':
                sampled_trainset_event = trainset[2][indices]
                sampled_trainset = (sampled_trainset_X, sampled_trainset_y, sampled_trainset_event)
            else:
                sampled_trainset = (sampled_trainset_X, sampled_trainset_y)
                
            if target == 'frailty':
                best_coefs, scores, R2_save, new_col = find_best_lm_model(sampled_trainset, trainset_names, set2, colname, bootstrap = True)
                combined_r2_or_C.append(R2_save)
                np.save(f"./output_linear/bootstrap/{target}/R2_{dset}.npy", R2_save)
                combined_scores.append(scores)
                
            elif target == 'mort':
                best_model, best_coefs, scores, C_save, new_col = find_best_coxph_model(sampled_trainset, trainset_names, set2, colname, bootstrap = True)
                combined_r2_or_C.append(C_save)
            else:
                raise ValueError(f"Target '{target}' not recognized. Use 'frailty' or 'mort'.")

            combined_coefs.append(best_coefs)
            combined_new_col.append(new_col)
       
        # Combine the results
        combined_scores = np.array(combined_scores)
        np.save(f"./output_linear/bootstrap/{target}/combined_scores_{target}_{dset}_{based_on}.npy", combined_scores)
        combined_coefs = pd.concat(combined_coefs, axis=1)
        combined_new_col = pd.concat(combined_new_col, axis=1)
        combined_r2_or_C = np.array(combined_r2_or_C)
        combined_coefs.to_csv(f"./output_linear/bootstrap/{target}/coefs_{target}_{dset}_{based_on}.csv")
        combined_new_col.to_csv(f"./output_linear/bootstrap/{target}/en_{target}_{dset}_{based_on}_set2.csv")
        np.save(f"./output_linear/bootstrap/{target}/metric_{target}_{dset}_{based_on}.npy", combined_r2_or_C)


    elif bootstrap == False and target == 'frailty':
        # Find best linear model
        best_coefs, scores, R2_save, new_col = find_best_lm_model(trainset, trainset_names, set2, colname, dset, combine)

        if combine:
            best_coefs.to_csv(f'./output_linear/coefs_frail/combine_coefs_frailty_{dset}.csv')
            new_col.to_csv(f'./output_linear/set2frail/combine_en_frailty_{dset}_set2.csv')
            np.save(f"./output_linear/rsquared/combine_R2_{dset}.npy", R2_save)
        else:
            # Save results
            np.save(f"./output_linear/scores_frail/scores_frailty_{dset}.npy", scores)
            best_coefs.to_csv(f'./output_linear/coefs_frail/coefs_frailty_{dset}.csv')
            new_col.to_csv(f'./output_linear/set2frail/en_frailty_{dset}_set2.csv')
            np.save(f"./output_linear/rsquared/R2_{dset}.npy", R2_save)

    elif bootstrap == False and target == 'mort':
        # Find best Cox proportional hazards model
        best_model, best_coefs, scores, C_save, new_col = find_best_coxph_model(trainset, trainset_names, set2, colname, dset, combine)
        if combine:
            best_coefs.to_csv(f'./output_linear/coefs_mort/combine_coefs_mort_{dset}.csv')
            new_col.to_csv(f'./output_linear/set2mort/combine_en_mort_{dset}_set2.csv')
            np.save(f"./output_linear/concordance/combine_C_mort_{dset}.npy", C_save)
        else:
            # Save results
            np.save(f"./output_linear/scores_mort/scores_mort_{dset}.npy", scores)
            best_coefs.to_csv(f'./output_linear/coefs_mort/coefs_mort_{dset}.csv')
            new_col.to_csv(f'./output_linear/set2mort/en_mort_{dset}_set2.csv')
            np.save(f"./output_linear/concordance/C_mort_{dset}.npy", C_save)
            np.save(f"./output_linear/bestmodel_mort/best_model_mort_{dset}.npy", best_model)

    else:
        raise ValueError(f"Target '{target}' not recognized. Use 'frailty' or 'mort'.")





def main():
    parser = argparse.ArgumentParser(description='Find the best model based on dataset and target.')
    parser.add_argument('dset', type=str, help='Dataset name (e.g., allprot, cmb, cmb_sub, etc.)')
    parser.add_argument('target', type=str, help='Target variable (frailty orqu mort)')
    parser.add_argument('--bootstrap', action='store_true', help='Set this flag to enable bootstrapping')
    parser.add_argument('--combine', action='store_true', help='Set this flag to combine train and set2 into big train set')
    parser.add_argument('--quickbootstrap', action='store_true', help='Set this flag to enable faster bootstrapping')
    

    args = parser.parse_args()
    find_best_model(args.dset, args.target, bootstrap=args.bootstrap, quickbootstrap=args.quickbootstrap, combine=args.combine)

if __name__ == "__main__":
    main()

        
