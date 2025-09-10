import torch
import torch.nn as nn
import numpy as np
from .nn_common import neg_log_likelihood
# from utils import *
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import math
from torch.utils.data import DataLoader, TensorDataset

# from MetaboNet.model_dev_workflow.MetaboNet_model import PriorKnowledgeLayer as PKL

            

class DeepSurv(nn.Module):
    # the default network for biomarker creation. simple linear network with some dropout
    # last architecture changes made on 14-10-2024 : added last reLU layer, modified the allprot architecture
    def __init__(self, dset = 'cmb', target = 'mort', in_dim = None):
        super().__init__()


        self.out_dim = 1
        if dset == "cmb":
            self.net = nn.Sequential(
                nn.Linear(in_dim, 200),
                nn.Dropout(p= 0.2),
                nn.ReLU(),
                nn.Linear(200, 100),
                nn.Dropout(p = 0.1),
                nn.ReLU(),
                nn.Linear(100, 10),
                nn.ReLU(), # added this one on 14th oct.
                nn.Linear(10, self.out_dim))
            
        elif dset == "allprot":
            self.net = nn.Sequential(
                nn.Linear(in_dim, 750),
                nn.Dropout(p= 0.2),
                nn.ReLU(),
                nn.Linear(750,400),
                nn.Dropout(p= 0.1),
                nn.ReLU(),
                nn.Linear(400,100),
                nn.Dropout(p=0.1),
                nn.ReLU(),
                nn.Linear(100,self.out_dim))
            
        elif dset == 'cmb_met':
            self.net = nn.Sequential(
                nn.Linear(in_dim, 342),
                nn.SELU(),
                nn.Linear(342, 253),
                nn.SELU(),
                nn.Linear(253, 164),
                nn.SELU(),
                nn.Linear(164, 95),
                nn.SELU(),
                nn.Linear(95, 10),
                nn.SELU(),
                nn.Linear(10,self.out_dim)
            )

        elif dset == "cmb_ffs":
            self.net = nn.Sequential(
                nn.Linear(in_dim, 15),
                # nn.Dropout(p=0.1),
                nn.ReLU(),
                nn.Linear(15, 10),
                nn.ReLU(),
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5,self.out_dim))
        
        elif dset == "cmb_mh":
            self.net = nn.Sequential(
                nn.Linear(in_dim, 253),
                nn.SELU(),
                nn.Dropout(p = 0.2),
                nn.Linear(253, 164),
                nn.SELU(),
                nn.Dropout(p = 0.1),
                nn.Linear(164, 95),
                nn.SELU(),
                nn.Linear(95, 10),
                nn.SELU(),
                nn.Linear(10,self.out_dim)
            )
            
        else:
            raise ValueError(f'unknown value for dataset: {dset}')

    def forward(self, x):
        return self.net(x)

    def test_mort(self, X, T, E, beta = None):
        if self.out_dim > 1:
            out = (self.forward(X)[:,0]).reshape(-1, 1)
        else:
            out = self.forward(X)
        
        surv_loss = neg_log_likelihood(out, X, T, E)
        conc = concordance_index_censored(E.detach().numpy().astype(bool), T.detach().numpy(), out.detach().numpy()[:,0])[0]

        return surv_loss.detach().numpy(), conc, None
    
    def train_net_mort(self, opt, X_train, T_train, E_train, beta = None, X_full = None, T_full = None, E_full = None):
            
        out = self.forward(X_train)
        loss = neg_log_likelihood(out, X_train, T_train, E_train)
        loss.backward()
        opt.step()

    def test_frailty(self, X, Y, beta = None):
        if self.out_dim > 1:
            out = (self.forward(X)[:,1]).reshape(-1, 1)
        else:
            out = self.forward(X)
        crit = nn.MSELoss()
        loss = crit(out, Y)
        
        score = r2_score(Y.detach().numpy(), out.detach().numpy())
        return loss.detach().numpy(), score

    def train_net_frailty(self, opt,  X, Y, beta = None, X_full = None, Y_full = None):
        crit = nn.MSELoss()

        out = self.forward(X)
        loss = crit(out, Y)
        loss.backward()
        opt.step()