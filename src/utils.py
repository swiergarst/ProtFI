import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils._param_validation import StrOptions
import torch
import wandb
from tqdm import tqdm
import math
from sksurv.metrics import concordance_index_censored

import os
import sys
wd = os.getcwd()
sys.path.append(wd + '/jointomicscomp/')


import scipy as sc
from .constants import *




def preprocess(data_df: pd.DataFrame, target_df: pd.DataFrame,target:str='mort', etime_name:str = 'censorage', eindicator_name:str = 'died' , id_name:str = 'eid', frailty_name:str = 'FI_0', do_pca:bool = False, n_components:int = 2, normalize:bool = True, var_t = 0,cols = None):
    ''' 
    tranforms a dataframe into a split between data, event time and event indicator (X, T, E, respectively).
    input:
        data_df: a dataframe  with all the data. included an 'eid' column with participant id's.
        target_df: dataframe with the event times and indicators, or frailty indexes
        etime_name: column name for the event time
        eindicator_name: column name for the event indicator
        id_name: column name for the id (used to merge the dataframes)
        do_pca: whether to perform pca on the data
        n_components: the amount of components to use in pca (unused if do_pca= False)
    output:
        X: (n,d) numpy array with data
        T: (n,) numpy array with event times
        E: (n,) numpy array with event indicators
        output is sorted based on event time
    '''

    if cols is not None:
        data_df = data_df[cols]
    else:
        # drop proteins under variance treshold
        data_df_noid = data_df.drop(columns=[id_name])
        data_df_noid = data_df_noid.loc[:, data_df_noid.var() > var_t]
        data_df_noid[id_name] = data_df[id_name]
        data_df = data_df_noid # bit of beun but w/e
    
    if target == 'mort':
        # merge based on merge column (just to ensure everything is sorted fully)
        data_df = data_df.drop(columns = [etime_name, eindicator_name], errors = "ignore")
        full_df = data_df.merge( target_df[[etime_name, eindicator_name]], on= id_name)
    
        # create the X, T, E split
        X = full_df.drop(columns =[etime_name, eindicator_name, id_name]).values
        T = full_df[etime_name].values
        E = full_df[eindicator_name].values

    elif target == 'frailty':
        # frailty_name = 'FI_0' # hardcoding this for now, could do something more elegant in the future
        # data_df = data_df.drop(columns = [etime_name, eindicator_name], errors = "ignore")
        full_df = data_df.merge(target_df[frailty_name], on = id_name) 
        if normalize == False: # still normalize age if this is the case
            scaler = StandardScaler()
            full_df['age_center.0.0'] = scaler.fit_transform(full_df['age_center.0.0'].values.reshape(-1, 1))

        X = full_df.drop(columns = [frailty_name, id_name]).values
        Y = full_df[frailty_name].values

    elif target == "ft_mort":
        data_df = data_df.drop(columns = [etime_name, eindicator_name, frailty_name], errors = "ignore")
        full_df = data_df.merge(target_df, on = id_name)   
        # print(f'full data shape: {full_df.shape}')
        if normalize == False: # still normalize age if this is the case
            scaler = StandardScaler()
            full_df['age_center.0.0'] = scaler.fit_transform(full_df['age_center.0.0'].values.reshape(-1, 1))
        
        X = full_df.drop(columns = [frailty_name, etime_name, eindicator_name, id_name]).values
        Y = full_df[frailty_name].values
        T = full_df[etime_name].values
        E = full_df[eindicator_name].values

    elif target == "ft_components":
        full_df = data_df.merge(target_df, on = id_name)

        X = full_df.drop(columns = FT_COMPONENTS).values
        Y = full_df[FT_COMPONENTS].values
    
    if normalize == True:
        # normalize the protein values
        scaler = StandardScaler()      
        X = scaler.fit_transform(X)


    if target == 'mort':
        return (X, T, E), full_df[id_name].values, data_df.columns
    elif (target == 'frailty') or (target == 'ft_components'):
        return (X, Y), full_df[id_name].values, data_df.columns
    elif target == "ft_mort":
        return (X, Y, T, E), full_df[id_name].values, data_df.columns
    #return sort_sets(X, T, E)

# sorts proteins, event times, and events according to the event times (DESCENDING)
def sort_sets(X, T, E):
    sort_idx = np.argsort(T)[::-1]
    X = X[sort_idx]
    T = T[sort_idx]
    E = E[sort_idx]
    return X, T, E	
	

def load_config(n_epochs = 100, lr = 5e-6, weight_decay=1e-5, batch_size=1000, gamma=0.99, beta = 0.1, n_runs=1, dset="cmb", net="ds_default", target = 'mort', log_wandb = False, init= "normal", var_t = 0, in_dim = 344, add_age = False, nested_batch_size = 0, combine_sets = False, seed = 7, bootstrap = False):
### loads (default_ settings into the config dictionary, which is used to set hyperparameters/settings in the main training loop.
### 
    config_options = {
        "n_epochs" : np.array([1, 1000]), # number of epochs (upper limit is arbitrary)
        "lr" : np.array([0, 1]), # learning rate
        "weight_decay" : np.array([0, 1]), # l2 regularization parameter
        "batch_size" : np.array([1, 100000]), # batch size
        "beta" : np.array([0, 1]), # loss function mixing for auto encoder models
        "gamma" : np.array([0, 1]), # learning rate scheduler parameter (coefficient)
        "dset" : ["allprot", "cmb", 'cmb_met', 'cmb_mh', 'cmb_sub', "cmb_met_pca", "cmb_met_pca20", "cmb_met_ajive", "cmb_met_ajive20", "cmb_ffs", "cmb_met_ffs", "allprot_ffs"], # protein subset to use
        "net" : ["ds_default", "ds_lognet","ds_deep", "ae_default", "ae_double_out", "ae_combined", "ae_prior", "bnn", "PoE"], # network architecture
        "runs" : np.array([1, 100]), # number of runs (upper limit is arbitrary)
        "target" : ['mort', 'frailty', 'ft_mort', 'ft_components'], # whether to train on mortality or frailty, or both
        "log_wandb" : [True, False], # whether to log the runs on weights and biases or not
        "init" : ['normal', 'uniform', 'kaiming_n', 'kaiming_u', 'xavier_n', 'xavier_u'], # different weight initializations
        "var_t": np.array([0,1]), # threshold to drop based on variance
        "in_dim": np.array([1,1500]), #dimensionality of the input
        "add_age" : [True, False],# whether to add age (explicitly) to the input data (implicitly done when target == frailty)
        "nested_batch_size": np.array([1, 10000]), # inner batch size
        "combine_sets": [True, False], # to train on the combined train/val split
        "seed": np.array([1, 100]), # random seed
        "bootstrap" : [True, False]
    }
    if nested_batch_size == 0:
        nested_batch_size = batch_size
    config = {
        "n_epochs": n_epochs,
        "lr" : lr,
        "weight_decay" : weight_decay,
        "batch_size" : batch_size,
        "gamma" : gamma,
        "beta" : beta,
        "dset": dset, # cmb, full, pca_cmb, pca_full
        "net" : net, # deepSurv options: ds_default, ds_lognet. auto encoder options: ae_lognet, ae_double_out, ae_
        "runs" : n_runs,
        "target" : target,
        "log_wandb" : log_wandb,
        "init" : init,
        "var_t": var_t,
        "in_dim": in_dim,
        "add_age" : add_age,
        "nested_batch_size": nested_batch_size,
        "seed" : seed,
        "combine_sets" : combine_sets,
        "bootstrap" : bootstrap
        }

    # assert whether dictionary is correct
    for key in config_options.keys():
        if isinstance(config_options[key], list):
            error_message = f'config {key} value not in options. value: {config[key]}. options: {config_options[key]}'
            assert config[key] in config_options[key], error_message
        elif isinstance(config_options[key], np.ndarray):
            error_message = f'config {key} value not within limits. value: {config[key]}. limits: [{config_options[key].min()}, {config_options[key].max()}]'
            assert (config[key] >= config_options[key].min()) & (config[key] <= config_options[key].max()), error_message

    if target == "frailty":
        config['add_age'] = True
    return config

# splits a dataframe into train/validation/test
def create_ttv_split(df, train_ratio: int = 0.7, val_ratio: int = 0.2) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    train_sub_idx  = math.floor(df.shape[0] * train_ratio)
    val_sub_idx = math.floor(df.shape[0] * (train_ratio + val_ratio))

    train_df = df.iloc[:train_sub_idx, :]
    val_df = df.iloc[train_sub_idx:val_sub_idx, :]
    test_df = df.iloc[val_sub_idx:,:]

    return train_df, val_df, test_df









def combine_sets(prots_train_df, prots_val_df, prots_test_df, target_train, target_val, target_test, target_name):
    prots_trainval_df = pd.concat((prots_train_df, prots_val_df))
    target_trainval = pd.concat((target_train, target_val))

    full_train, train_eids, train_cols = preprocess(prots_trainval_df, target_trainval, target = target_name)

    if prots_test_df is not None:
        full_test, test_eids, test_cols = preprocess(prots_test_df, target_test, target = target_name)
    else:
        full_test = None
        test_eids = None
        
    return full_train, full_test, train_eids, test_eids

def select_columns(df, dset, target):

    if target == 'mort':
        folder = 'coefs_mort'
    elif target == 'frailty':
        folder = 'coefs_frail'

    if dset == "cmb_ffs":
        cols_to_select = pd.read_csv(f'output_linear/{folder}/ffs_coefs_cmb_tol0.001.csv')['Unnamed: 0'].values

    elif dset == "cmb_met_ffs":
        cols_to_select = pd.read_csv(f'output_linear/{folder}/ffs_coefs_cmb_met_tol0.001.csv')['Unnamed: 0'].values

    elif dset == "allprot_ffs":
        cols_to_select = pd.read_csv(f'output_linear/{folder}/ffs_coefs_allprot_tol0.001.csv')['Unnamed: 0'].values
        
    ## This doesn't work
    # elif dset == "cmb_clust":
    #     cols_to_select = np.load(f'output_linear/bootstrap/{target}/{target}samedirection.npy')
    # remove age if it's selected (we add it later)
    age_id = np.where(cols_to_select == 'age')[0]
    if len(age_id) > 0:
        cols_to_select = np.delete(cols_to_select, age_id)

    cols_to_select = np.append(cols_to_select,'eid')
    # create subset
    # print(f'df shape: {df[cols_to_select].shape}')
    return df[cols_to_select]        


def get_data(config, age_col_name = None):
    prefix_data = "Data/Processed"

    normalize = True
    prots_test_df = None

    # default from UKB
    if age_col_name is None:
        age_col_name = 'age_center.0.0'
        
    # fallback if add_age is not defined
    if 'add_age' not in config.keys():
        if config['target'] == 'mort':
            config['add_age'] = False
        elif config['target'] == 'frailty':
            config['add_age'] = True
        else:
            config['add_age'] = False

    if (config['dset'] == "allprot") or (config['dset'] == "allprot_ffs"):
        prots_train_df = pd.read_csv(f'{prefix_data}/Full/full_train.csv')
        prots_val_df = pd.read_csv(f'{prefix_data}/Full/full_val.csv')
        prots_test_df = pd.read_csv(f'{prefix_data}/Full/full_test.csv')
        
    elif (config['dset'] == 'cmb') or (config['dset'] == 'cmb_ffs'):
        prots_train_df = pd.read_csv(f'{prefix_data}/Full/full_train_cmb.csv')
        prots_val_df = pd.read_csv(f'{prefix_data}/Full/full_val_cmb.csv')
        prots_test_df = pd.read_csv(f'{prefix_data}/Full/full_test_cmb.csv')

    elif (config['dset'] == 'cmb_met') or (config['dset'] == 'cmb_met_ffs') or (config['dset'] == 'cmb_clust'):
        prots_train_df = pd.read_csv(f'{prefix_data}/MultiOmics/proteins_metabolites_train.csv')
        prots_val_df = pd.read_csv(f'{prefix_data}/MultiOmics/proteins_metabolites_set2.csv')
        prots_test_df = pd.read_csv(f'{prefix_data}/MultiOmics/proteins_metabolites_set3.csv')

    elif config['dset'] == "cmb_met_pca":
        normalize = False # don't normalize the pca'd data
        prots_train_df = pd.read_csv(f'{prefix_data}/MultiOmics/proteins_metabolites_pca_train.csv')
        prots_val_df = pd.read_csv(f'{prefix_data}/MultiOmics/proteins_metabolites_pca_set2.csv')

    elif config['dset'] == "cmb_met_pca20":
        normalize = False
        prots_train_df = pd.read_csv(f'{prefix_data}/MultiOmics/proteins_metabolites_pca20_train.csv')
        prots_val_df = pd.read_csv(f'{prefix_data}/MultiOmics/proteins_metabolites_pca20_set2.csv')

    elif config['dset'] == "cmb_met_ajive":
        prots_train_df = pd.read_csv(f'{prefix_data}/MultiOmics/ajive10_prot_met_train.csv')
        prots_val_df = pd.read_csv(f'{prefix_data}/MultiOmics/ajive10_prot_met_set2.csv')

    elif config['dset'] == "cmb_met_ajive20":
        prots_train_df = pd.read_csv(f'{prefix_data}/MultiOmics/ajive20_prot_met_train.csv')
        prots_val_df = pd.read_csv(f'{prefix_data}/MultiOmics/ajive20_prot_met_set2.csv')
    
    elif config['dset'] == 'cmb_mh':
        prots_train_df = pd.read_csv(f'{prefix_data}/MultiOmics/proteins_mh_train.csv')
        prots_val_df = pd.read_csv(f'{prefix_data}/MultiOmics/proteins_mh_set2.csv')
        
    elif config['dset'] == 'cmb_sub':
        prots_train_df = pd.read_csv(f'{prefix_data}/MultiOmics/proteins_mh_train.csv')
        prots_val_df = pd.read_csv(f'{prefix_data}/MultiOmics/proteins_mh_set2.csv')

        prots_train_df = prots_train_df.drop(columns = ['mortScore'])
        prots_val_df = prots_val_df.drop(columns = ['mortScore'])

    ## this doesn't work
    # elif config['dset'] == 'cmb_morefrail': # subset of people who are more frail
    #     prots_train_df = pd.read_csv("Data" + "/morefrail_training.csv")
    #     prots_val_df = pd.read_csv("Data" + "/morefrail_set2.csv")

    # elif config['dset'] == 'cmb_4050' : # subset of people ages 40-50
    #     prots_train_df = pd.read_csv("Data" + "/cmb_train_4050.csv")
    #     prots_val_df = pd.read_csv("Data" + "/cmb_set2_4050.csv")

    # elif config['dset'] == 'cmb_5060' : # subset of people ages 50-60
    #     prots_train_df = pd.read_csv("Data" + "/cmb_train_5060.csv")
    #     prots_val_df = pd.read_csv("Data" + "/cmb_set2_5060.csv")
            
    # elif config['dset'] == 'cmb_6070' : # subset of people ages 60-70
    #     prots_train_df = pd.read_csv("Data" + "/cmb_train_6070.csv")
    #     prots_val_df = pd.read_csv("Data" + "/cmb_set2_6070.csv")

    
    else:
        raise ValueError(f'unknown value for dset: {config["dset"]}')
    
    prefix_endpoints = "Data/endpoints"

    # select columns from feature selection
    if config['dset'] in ('cmb_ffs', 'cmb_met_ffs','allprot_ffs'):
        prots_train_df = select_columns(prots_train_df, config['dset'], config['target'])
        prots_val_df = select_columns(prots_val_df, config['dset'], config['target'])
        if prots_test_df is not None:
            prots_test_df = select_columns(prots_test_df, config['dset'], config['target'])
    
    if config['target'] == "mort":
        target_train = pd.read_csv(f'{prefix_data}/Full/mort_full_train.csv', index_col = 'eid')
        target_val = pd.read_csv(f'{prefix_data}/Full/mort_full_test.csv', index_col = 'eid')
        if prots_test_df is not None: # failsafe for the datasets for which we haven't made a test set (set 3) yet
            target_test = pd.read_csv(f'{prefix_data}/Full/mort_full_val.csv', index_col = "eid")
        #mort_test = pd.read_csv("Data/Processed/Full/mort_full_val.csv", index_col = 'eid')

    elif config['target'] == "frailty":
        target_train = pd.read_csv(f'{prefix_endpoints}/Frailty/frailty_clean_train.csv', index_col = 'eid')
        target_val = pd.read_csv(f'{prefix_endpoints}/Frailty/frailty_clean_set2.csv', index_col = 'eid')

        if prots_test_df is not None:
            target_test = pd.read_csv(f'{prefix_endpoints}/Frailty/frailty_clean_set3.csv', index_col = "eid")


    elif config['target'] == 'ft_mort':
        # load both mortality and frailty information
        mort_train = pd.read_csv(f'{prefix_data}/Full/mort_full_train.csv', index_col = 'eid')
        mort_val = pd.read_csv(f'{prefix_data}/Full/mort_full_test.csv', index_col = 'eid')
        ft_train = pd.read_csv(f'{prefix_endpoints}/Frailty/frailty_clean_train.csv', index_col = 'eid')
        ft_val = pd.read_csv(f'{prefix_endpoints}/Frailty/frailty_clean_set2.csv', index_col = 'eid')

        target_train = mort_train.merge(ft_train, on = 'eid')
        target_val = mort_val.merge(ft_val, on = 'eid')
        if prots_test_df is not None:
            ft_test = pd.read_csv(f'{prefix_endpoints}/Frailty/frailty_clean_set3.csv', index_col = "eid")
            mort_test = pd.read_csv(f'{prefix_data}/Full/mort_full_val.csv', index_col = "eid")
            target_test = mort_test.merge(ft_test, on = 'eid')

    
    elif config['target'] == 'ft_components':
        target_train = pd.read_csv("Data/endpoints/Frailty/frailtyComponents_processed.csv")
        target_test = target_train # bit dirty but this works
        target_val = target_train

    if config['add_age'] == True: # add age to input data
        basicinfo = pd.read_csv("Data/covar/basicinfo_instance_0.csv", index_col = "eid")
        prots_train_df = prots_train_df.merge(basicinfo[age_col_name], on = 'eid')
        prots_val_df = prots_val_df.merge(basicinfo[age_col_name], on = 'eid')

        if prots_test_df is not None:
            prots_test_df = prots_test_df.merge(basicinfo[age_col_name], on = "eid")


    
    if "var_t" in config.keys():
        var_t = config['var_t']
    else:
        var_t = 0

    # combining set 1 and 2 for training
    if config['combine_sets'] == True:
        return combine_sets(prots_train_df, prots_val_df, prots_test_df, target_train, target_val, target_test, config['target'])
    
    full_train, train_eids, train_cols = preprocess(prots_train_df, target_train, target = config['target'] , normalize = normalize, var_t=var_t)
    full_val, val_eids, val_cols = preprocess(prots_val_df, target_val, target = config['target'], normalize = normalize,cols=train_cols)

    if prots_test_df is not None:
        full_test, test_eids, test_cols = preprocess(prots_test_df, target_test, target = config['target'], normalize = normalize,cols=train_cols)


    return full_train, full_val, full_test, train_eids, val_eids, test_eids, train_cols
                                                   


## TODO: remove most of these, deprecated
def get_net(dset,  net_cfg, target, in_dim = None):
    from .AutoEncoder_models import auto_encoder, AE_combined, AE_double_out, AE_prior, PoE
    from .DeepSurv_models import DeepSurv, LogNet, DeepDeepSurv, bioInspiredNN
    from .multitask_models import ft_mort_model
    if target == "ft_mort":
        net = ft_mort_model(in_dim = in_dim)
        # net = net.double()
        # return net
    elif net_cfg == 'ae_default':
        net = auto_encoder(dset = dset)
    elif net_cfg == 'ae_combined':
        net = AE_combined(dset = dset)
    elif net_cfg == 'ae_double_out':
        net = AE_double_out(dset = dset)
    elif net_cfg == 'ae_prior':
        net = AE_prior(dset = dset)
    elif net_cfg == 'ds_default':
        net = DeepSurv(dset = dset, target = target, in_dim = in_dim)
    elif net_cfg == 'ds_lognet':
        net = LogNet(dset = dset)
    elif net_cfg == "ds_deep":
        net = DeepDeepSurv(dset = dset)
    elif net_cfg == "bnn":
        net = bioInspiredNN(target = target)
    elif net_cfg == "PoE":
        net = PoE(dset = dset, target = target)
    else:
        raise ValueError(f'unkown net: {config["net"]}')

    net = net.double()
    return net

# convienence function to also call get_data to get the data for nn predictions
def get_pred(get_data_cfg, net_cfg, target, weights_path, set_num = 2):

    if combine_sets == False:
        train, val, test, train_eids, val_eids, test_eids, _ = get_data(get_data_cfg)
    else:
        set_num = 3
        train, test, train_eids, test_eids = get_data(get_data_cfg)
    
    in_dim = train[0].shape[1]
    net = get_net(dset, net_cfg, target, in_dim = in_dim)
    net.load_state_dict(torch.load(weights_path))
    
    
    if set_num == 1:
        in_data = torch.Tensor(train[0]).double()
        in_eids = train_eids
    elif set_num == 2:
        in_data = torch.Tensor(val[0]).double()
        in_eids =val_eids
    elif set_num == 3:
        in_data = torch.Tensor(test[0]).double()
        in_eids = test_eids
    else:
        raise valueError(f'unknown value for set_num: {set_num}')


    
    return get_nn_output(in_data, weights_path, get_data_cfg['dset'], get_data_cfg['target'], data_index = in_eids)


def get_nn_output(data_in, model_path, dset_type, target, net_type = "ds_default", data_index = None):
    tx_data = torch.Tensor(data_in).double()
    in_dim = data_in.shape[1]
    net = get_net(dset_type, net_type, target, in_dim = in_dim)
    net.load_state_dict(torch.load(model_path))

    output = net.forward(tx_data)

    if data_index is None:
        data_index = np.arange(data_in.shape[0]) # numerical index if no eids available
    out_df = pd.DataFrame(output.detach().numpy(), index = data_index)

    return out_df
    
                                

