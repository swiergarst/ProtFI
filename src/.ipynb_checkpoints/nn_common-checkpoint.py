import numpy as np
import torch
import scipy as sc
import pandas as pd
from .utils import get_net, get_data
from .constants import *
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import wandb
from tqdm import tqdm
import math
from sksurv.metrics import concordance_index_censored

def init_weights(n, init):
    if init == "normal":
        f = torch.nn.init.normal_
    elif init == "uniform":
        f = torch.nn.init.uniform_
    elif init == "xavier_n":
        f = torch.nn.init.xavier_normal_
    elif init == "xavier_u":
        f = torch.nn.init.xavier_uniform_
    elif init == "kaiming_n":
        f = torch.nn.init.kaiming_normal_
    elif init == "kaiming_u":
        f = torch.nn.init.kaiming_uniform_
            
    if isinstance(n, nn.Linear):
        f(n.weight)


def make_bootstrap(data, seed):
    np.random.seed(seed)
    ids = np.random.choice(data[0].shape[0], size=data[0].shape[0], replace=True)
    new_data = []
    for arr in data:
        new_arr = arr[ids,...]
        new_data.append(new_arr)

    return new_data



def neg_log_likelihood(out, T, E):
    # out: h(xi) for i in set of observations with an event
    # X: covariate data
    # T: right-censored event times
    # E: whether an event has occurred or not
    # print(f'T shape: {T.shape}')
    
    # we sort the input (and output) based on event time, such that we can use cumsum (and make our lives easier)
    sort_idx = np.argsort(T.detach().numpy())[::-1]
    # X = X[sort_idx.copy(), :]
    T = T[sort_idx.copy()]
    E = E[sort_idx.copy()]
    out = out[sort_idx.copy(), ...]

    
    cum_risk = torch.log(torch.cumsum(torch.exp(out), 0))

    
    likelihood = (out - cum_risk)[:,0]* E # multiply by E to keep only the events

    neg_likelihood = - torch.sum(likelihood) / torch.sum(E)
    return neg_likelihood



def rank_loss(out, X, T, E):
    X_np = X.detach().numpy()
    dist = sc.spatial.distance_matrix(X_np, X_np) # there is a better way to do this
    sort_idx = np.argsort(X_np, axis = 1)

    out_new = np.zeros_like(X_np)
    for i in range(X_np.shape[0]):
        out_new[i,:] = out[i, sort_idx[i,:]]

    Tout_new = torch.Tensor(out_new)

    crit = nn.MSELoss()
    loss = crit(Tout_new, out)
    
    return loss


def preprocess_ft_comps(frailty_df, eids):
    sel_ft_df = frailty_df.loc[frailty_df['eid'].isin(eids)]
    sel_ft_df.index = sel_ft_df['eid']
    sel_ft_df = sel_ft_df.reindex(index = eids)
    # sel_ft_components_df = sel_ft_df[FT_COMPONENTS]
    sel_ft_components_df = sel_ft_df[BASE_FT_COMPONENTS]
    sel_ft_components_arr = sel_ft_components_df.values
    ft_cols = sel_ft_components_df.columns

    return sel_ft_components_arr, ft_cols, sel_ft_df['FI_0']


def ft_components_loop(config):
    from .multitask_models import ft_components_model

    # just hardcode the data for now, could mess around with a get_data later if needed
    # still get the protein data from get_data
    full_train, full_val, full_test, train_eids, val_eids, test_eids, _ = get_data(config)
    # frailty_df = pd.read_csv("Data/endpoints/Frailty/frailtyComponents_processed.csv")
    frailty_df = pd.read_csv("Data/endpoints/Frailty/UKB_WithFI0Codes.csv")

    # some dirty preprocessing
    train_ft_arr, train_ft_cols, train_ft = preprocess_ft_comps(frailty_df, train_eids)
    val_ft_arr, val_ft_cols, val_ft = preprocess_ft_comps(frailty_df, val_eids)
    test_ft_arr, test_ft_cols, test_ft = preprocess_ft_comps(frailty_df, test_eids)

    tX_train = torch.Tensor(full_train[0]).double()
    tY_train = torch.Tensor(train_ft_arr)
    
    tX_val = torch.Tensor(full_val[0]).double()
    tY_val = torch.Tensor(val_ft_arr)

    net = ft_components_model(in_dim = config['in_dim'])
    net = net.double()
    opt = torch.optim.Adam(net.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])


    n_epochs = config['n_epochs']
    loss_train = np.zeros(n_epochs)
    loss_val = np.zeros_like(loss_train)
    acc_comps_train = np.zeros_like(loss_train)
    r2_train = np.zeros_like(loss_train)
    r2_val = np.zeros_like(loss_train)

    train_loader = DataLoader(TensorDataset(tX_train, tY_train), batch_size = config['batch_size'], shuffle = True)
    for e in tqdm(range(n_epochs)):
        with torch.no_grad():
            loss_train[e], _, r2_train[e]  = net.test_net(tX_train, train_ft, tY_train, train_ft_cols)
            loss_val[e], _, r2_val[e] = net.test_net(tX_val, val_ft, tY_val, val_ft_cols)
            
        for tX_batch, tY_batch in train_loader:
            net.train_net(opt, tX_batch, tY_batch, train_ft_cols)

        if config['log_wandb'] == 1:
            wandb.log({
                "training_loss" : loss_train[e].item(),
                "validation_loss" : loss_val[e].item(),
                # "concordance_train" : c_train[e].item(),
                # "concordance_val" : c_val[e].item(),
                "r2_train" : r2_train[e].item(),
                "r2_val" : r2_val[e].item()
            })



def training_loop(train_full, val_full, config, net = None):

    if config['bootstrap'] == True:
        train_full = make_bootstrap(train_full, config['seed'])
    
    tX_train = torch.Tensor(train_full[0]).double()
    tX_val = torch.Tensor(val_full[0]).double()

    if config['target'] == 'mort' :
        tT_train = torch.Tensor(train_full[1]).double()
        tE_train = torch.Tensor(train_full[2])
    
        tT_val = torch.Tensor(val_full[1]).double()
        tE_val = torch.Tensor(val_full[2])
        train_loader = DataLoader(TensorDataset(tX_train, tT_train, tE_train), batch_size = config['batch_size'], shuffle=True)

    elif config['target'] == 'frailty':
        tY_train = torch.Tensor(train_full[1][:,np.newaxis]).double()
        tY_val = torch.Tensor(val_full[1][:,np.newaxis]).double()
        train_loader = DataLoader(TensorDataset(tX_train, tY_train), batch_size = config['batch_size'], shuffle=True)

    elif config['target'] == "ft_mort":
        tY_train = torch.Tensor(train_full[1][:,np.newaxis]).double()
        tY_val = torch.Tensor(val_full[1][:,np.newaxis]).double()
        
        tT_train = torch.Tensor(train_full[2]).double()
        tE_train = torch.Tensor(train_full[3])

        tT_val = torch.Tensor(val_full[2]).double()
        tE_val = torch.Tensor(val_full[3])

        train_loader = DataLoader(TensorDataset(tX_train, tY_train, tT_train, tE_train), batch_size = config['batch_size'], shuffle = True)
        mort_train_loader = DataLoader(TensorDataset(tX_train, tT_train, tE_train), batch_size = config['batch_size'], shuffle = True)
        ft_train_loader = DataLoader(TensorDataset(tX_train, tY_train), batch_size = config['nested_batch_size'], shuffle = True)

    if net is None:
        net = get_net(config['dset'], config['net'], config['target'], config['in_dim'])

    
    #net.apply(lambda net: init_weights(n = net, init = config['init']))
    
    opt = torch.optim.Adam(net.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])
    scheduler = ExponentialLR(opt, gamma = config['gamma'])

    n_epochs = config['n_epochs']
    loss_train = np.zeros((n_epochs))
    loss_val = np.zeros(n_epochs)
    c_train = np.zeros(n_epochs)
    cos_train = np.zeros_like(c_train)
    c_val = np.zeros_like(c_train)
    cos_val = np.zeros_like(c_train)
    param_log = np.zeros_like(c_train)
    r2_train = np.zeros_like(c_train)
    r2_val = np.zeros_like(c_train)
    
    for e in tqdm(range(n_epochs)):
        if config['target'] == "mort":
            if e != 0:
                with torch.no_grad():
                    loss_train[e], c_train[e], cos_train[e]  = net.test_mort(tX_train, tT_train, tE_train, config['beta'])
                    loss_val[e], c_val[e], cos_val[e] = net.test_mort(tX_val, tT_val, tE_val, config['beta'])
            for tX_batch, tT_batch, tE_batch in train_loader:
                if config['net'] == "PoE":
                    config['beta'] = len(tX_batch) / len(train_loader.dataset)
                    # print(f'beta: {config['beta']}')
                opt.zero_grad()
                net.train_net_mort(opt, tX_batch, tT_batch, tE_batch, config['beta'],  X_full = tX_train, T_full = tT_train, E_full = tE_train)           
            scheduler.step()
    
        elif config['target'] == 'frailty':
            if e != 0:
                with torch.no_grad():
                    loss_train[e], r2_train[e] = net.test_frailty(tX_train, tY_train, config['beta'])
                    loss_val[e], r2_val[e] = net.test_frailty(tX_val, tY_val, config['beta'])
            for tX_batch, tY_batch in train_loader:
                opt.zero_grad()
                net.train_net_frailty(opt, tX_batch, tY_batch, config['beta'],  X_full = tX_train, Y_full = tY_train)
            scheduler.step()
            

        elif config['target'] == "ft_mort":
            if e != 0:
                with torch.no_grad():
                    loss_train[e], r2_train[e], c_train[e] = net.test(tX_train, tY_train, tT_train, tE_train, config['beta'])
                    loss_val[e], r2_val[e], c_val[e] = net.test(tX_val, tY_val, tT_val, tE_val, config['beta'])
            for tX_batch, tY_batch, tT_batch, tE_batch in train_loader:
                opt.zero_grad()
                net.train_net(opt, tX_batch, tY_batch, tT_batch, tE_batch, config['beta'])
            # net.train_net_ft_mort(opt, mort_train_loader, ft_train_loader)
                # net.train_net_seq(opt, tX_batch, tY_batch, tT_batch, tE_batch, config['nested_batch_size'])
            scheduler.step()
        
        else:
            raise ValueError(f'unknown value for target: {config["target"]}')

        

        if config['log_wandb'] == 1:
            if e != 0:
                wandb.log({
                    "training_loss" : loss_train[e].item(),
                    "validation_loss" : loss_val[e].item(),
                    "concordance_train" : c_train[e].item(),
                    "concordance_val" : c_val[e].item(),
                    "cosim_train" : cos_train[e].item(),
                    "cosim_val" : cos_val[e].item(),
                    "r2_train" : r2_train[e].item(),
                    "r2_val" : r2_val[e].item()
                })
    return net