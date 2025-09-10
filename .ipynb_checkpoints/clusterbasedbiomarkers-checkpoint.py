import numpy as np
import pandas as pd
import random
import argparse
from model_functions import *
from utils import *
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sksurv.linear_model import CoxPHSurvivalAnalysis

random.seed(7)

def get_strongest_features(dset, target, level):
    # Get data and trainset names
    trainset, set2, eids_train, eids_set2, _ = get_data({'dset': dset, 'target': target})
    trainset_names = get_trainsetnames(dset, target)
    
    # Make list otherwise won't work
    columnnames = list(trainset_names)
    
    # Transpose the data to get features instead of samples
    transposed_data = trainset[0].T

    if target == 'frailty':
        transposed_data = np.delete(transposed_data, -1, 0)
    
    # Perform hierarchical clustering using Ward's method
    linkage_matrix = sch.linkage(transposed_data, method='ward', metric='euclidean')
    
    # Assign clusters
    clusters = sch.fcluster(linkage_matrix, level, criterion='distance')

    if target == 'frailty':
        clusters = np.append(clusters, max(clusters)+1)
    
    # Create dataframe of column names and cluster assignments
    clustered_columns = pd.DataFrame({'Column Name': columnnames, 'Cluster': clusters})
        
    # Group by cluster
    grouped_columns = clustered_columns.groupby('Cluster')['Column Name'].apply(list)
    
    # Initialize list to store
    strongest_features = []
    
    # Start for loop
    for cluster, names in grouped_columns.items():
        best_feature = None
        best_score = -np.inf
        
        # Associate each feature in cluster with outcome
        for name in names:
            feature_idx = columnnames.index(name)
            X = trainset[0][:, feature_idx].reshape(-1, 1)  # Extract feature by index
            
            if target == 'frailty':
                y = trainset[1]
                model = LinearRegression().fit(X, y)
                score = model.score(X, y)  # R-squared value
            
            elif target == 'mort': #Still need to fix
                y = np.array([(e, t) for e, t in zip(trainset[2], trainset[1])], dtype=[("Status", "?"), ("Survival_in_days", "<f8")])
                model = CoxPHSurvivalAnalysis().fit(X, y)
                score = model.score(X, y)  # Concordance index
            
            # Keep track of the feature with the strongest association
            if score > best_score:
                best_score = score
                best_feature = name
        
        # Store the strongest feature from the cluster
        strongest_features.append(best_feature)
        
    # Get the indices of the strongest features per cluster
    strongest_feature_indices = [columnnames.index(f) for f in strongest_features]
    
    # Save column name for set2
    colname = f'en_cluster_{level}_{target}_{dset}'
    
    if target == 'mort':
        strongest_trainset = (trainset[0][:, strongest_feature_indices], trainset[1], trainset[2])
        strongest_set2 = (set2[0][:, strongest_feature_indices], set2[1], set2[2])
        best_model, best_coefs, scores, C_save, new_col = find_best_coxph_model(trainset, trainset_names, set2, colname)

        # Save results
        np.save(f"./output_linear/scores_mort/scores_cluster_{level}_{target}_{dset}.npy", scores)
        best_coefs.to_csv(f'./output_linear/coefs_mort/coefs_cluster_{level}_{target}_{dset}.csv')
        new_col.to_csv(f'./output_linear/set2mort/en_cluster_{level}_{target}_{dset}_set2.csv')
        np.save(f"./output_linear/concordance/C_cluster_{level}_{target}{dset}.npy", C_save)
        np.save(f"./output_linear/bestmodel_mort/best_model_cluster_{level}_{target}{dset}.npy", best_model)

    else:
        
        strongest_trainset = (trainset[0][:, strongest_feature_indices], trainset[1])
        strongest_set2 = (set2[0][:, strongest_feature_indices], set2[1])
        best_coefs, scores, R2_save, new_col = find_best_lm_model(strongest_trainset, strongest_features, strongest_set2, colname)
        
        # Save results
        np.save(f"./output_linear/scores_frail/scores_cluster_{level}_{target}_{dset}.npy", scores)
        best_coefs.to_csv(f'./output_linear/coefs_frail/coefs_cluster_{level}_{target}_{dset}.csv')
        new_col.to_csv(f'./output_linear/set2frail/en_cluster_{level}_{target}_{dset}_set2.csv')
        np.save(f"./output_linear/rsquared/R2_cluster_{level}_{target}{dset}.npy", R2_save)

def main():
    parser = argparse.ArgumentParser(description='Find the best model per cluster based on dataset and target.')
    parser.add_argument("--dset", type = str, default = "cmb_met", help = "dataset to use")
    parser.add_argument("--target", type = str, default = "frailty", help = "outcome either mort or frailty")
    parser.add_argument("--level", type = int, default = 16, help = "number of clusters")

    args = parser.parse_args()
    get_strongest_features(args.dset, args.target, args.level)

if __name__ == "__main__":
    main()
