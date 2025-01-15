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

        # if in_dim is None:
        #     # we need to define this before because of frailty
        #     if dset == 'allprot':
        #         in_dim = 1428
        #     elif dset == 'cmb' or dset == 'cmb_sub':
        #         in_dim = 344
        #     elif dset == "cmb_met":
        #         in_dim = 451
        #     elif dset == "cmb_mh":
        #         in_dim = 345
        #     elif dset == "cmb_met_pca":
        #         in_dim = 20
        #     elif dset == "cmb_met_pca20":
        #         in_dim = 40
        #     elif dset == "cmb_met_ajive":
        #         in_dim = 19
        #     elif dset == "cmb_met_ajive20":
        #         in_dim = 35
        #     else:
        #         raise valueError(f'unkown value for dataset: {dset}')


        # if target == 'frailty':
        #     in_dim += 1

        self.out_dim = 1
        if dset in["cmb", "cmb_sub", "cmb_4050", "cmb_5060", "cmb_6070", "cmb_morefrail"]:
            # self.net = nn.Sequential(
            #     nn.BatchNorm1d(in_dim),
            #     nn.Linear(in_dim, 253),
            #     nn.Dropout(p = 0.2),
            #     nn.SELU(),
            #     nn.BatchNorm1d(253),
            #     nn.Linear(253, 164),
            #     nn.Dropout(p = 0.1),
            #     nn.SELU(),
            #     nn.BatchNorm1d(164),
            #     nn.Linear(164, 95),
            #     nn.SELU(),
            #     nn.BatchNorm1d(95),
            #     nn.Linear(95, 10),
            #     nn.SELU(),
            #     nn.BatchNorm1d(10),
            #     nn.Linear(10,1)
            # )
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

        elif dset in ["cmb_met_pca", "cmb_met_pca20", "cmb_met_ajive", "cmb_met_ajive20", "cmb_ffs", "cmb_met_ffs", "allprot_ffs", "cmb_clust"]:
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

   
                
    def train_net_ft_components(self, opt, X, Y):
        pass
        


class bioInspiredNN(DeepSurv):
    def __init__(self, target = "mort", adjacency_matrices = None, nodes_per_pathway = [3, 3, 3]):
        super().__init__(dset = "cmb", target = target)
        if adjacency_matrices == None:
            adjacency_matrices = self.build_default_adjacency(target, nodes_per_pathway = nodes_per_pathway)
            print(f'matrix sizes: {[mat.shape for mat in adjacency_matrices]}') 
        PKL_list = [PKL(A) for A in adjacency_matrices]

        bn_shapes = [adj.shape[0] for adj in adjacency_matrices]
        bn_list = [nn.BatchNorm1d(shape) for shape in bn_shapes]
        self.bioLayers = nn.ModuleList([value for triplet in zip(bn_list, PKL_list, [nn.ReLU()] * len(PKL_list)) for value in triplet])
        self.bn_out = nn.BatchNorm1d(adjacency_matrices[-1].shape[1])
        self.out = nn.Linear(adjacency_matrices[-1].shape[1], 1)

    def forward(self, X):
        for layer in self.bioLayers:
            X = layer(X)
        X = self.bn_out(X)
        return self.out(X)

    def build_default_adjacency(self, target, nodes_per_pathway = [1, 1, 1]):
        pathways_df = pd.read_csv("KEGG_pathways.csv", delimiter = "\t")
        cmb_data_df = pd.read_csv("Data/Processed/Full/full_train_cmb.csv")
        cmb_proteins= cmb_data_df.columns[1:].values

        # create adjacency matrix for pathways
        ## determine amt of pathways
        n_p = pathways_df.shape[0]

        n_in = len(cmb_proteins)



        ## determine out size
        n_out = n_p * nodes_per_pathway[0]

        ## create empty matrix
        A_pw = np.zeros((n_in, n_out))

            

            # ## create a new n_in for the next iteration
            # n_in = n_out
            
        ## loop over all matching protein sets
        for i, matching_prots in enumerate(pathways_df['matching proteins in your network (labels)'].values):
            matching_prot_idx = np.nonzero(np.in1d(cmb_proteins,matching_prots.split(",")))
            starting_node = i*nodes_per_pathway[0]
            A_pw[matching_prot_idx, starting_node: starting_node + nodes_per_pathway[0] ] = 1


        ## add age if we are looking at frailty
        if target == "frailty":
            Age_vec = np.ones((1,n_out))
            A_pw = np.concatenate((A_pw, Age_vec))

        
        # create adjacency matrix for super-pathways
        ## determine unique super-pathways in df
        n_p2 = len(pathways_df['Super-pathway'].unique())
        n_out2 = n_p2 * nodes_per_pathway[1]
        
        ## create empty matrix
        A_pw2 = np.zeros((n_out, n_out2))
        
        ## loop over all super-pathways
        for i, spw in enumerate(pathways_df['Super-pathway'].unique()):
            # select all rows in df with that super pathway
            spw_df = pathways_df.loc[pathways_df['Super-pathway'] == spw]

            converted_indexes = np.concatenate([spw_df.index.values * nodes_per_pathway[0] + i for i in range(nodes_per_pathway[0])]) # bit of a funky generator to deal with the fact that we use more nodes per pathway
            starting_node = i*nodes_per_pathway[1]
            # put those values at 1
            A_pw2[converted_indexes, starting_node: starting_node + nodes_per_pathway[1] ] = 1
        
        # optional third layer: super-pathways to categories
        spws = pathways_df['Super-pathway'].unique()
        
        # define the categories
        spws_cats = np.array([math.floor(spw) for spw in spws])
        
        # determine unique categories
        un_cats = np.unique(spws_cats)
        n_out3 = len(un_cats) * nodes_per_pathway[2]
        
        
        A_pw3 = np.zeros((n_out2, n_out3))
        # fill in the adjacency matrix
        for i, spw_cat in enumerate(np.unique(spws_cats)):
            cat_idx = np.where(spws_cats == spw_cat)[0]

            converted_indexes = np.concatenate([cat_idx * nodes_per_pathway[1] + i for i in range(nodes_per_pathway[1])]) # bit of a funky generator to deal with the fact that we use more nodes per pathway
            starting_node = i*nodes_per_pathway[2]
            A_pw3[converted_indexes, starting_node: starting_node + nodes_per_pathway[2] ] = 1

        As = [torch.Tensor(A_pw), torch.Tensor(A_pw2), torch.Tensor(A_pw3)]
        return As


class DeepDeepSurv(DeepSurv):
    def __init__(self, dset = "cmb", target = 'mort'):
        super().__init__(dset = dset,target = target)

        if dset == "cmb": 
            self.net = nn.Sequential(
                nn.Linear(344, 300),
                nn.SELU(),
                nn.Linear(300,300),
                nn.SELU(),
                nn.Linear(300,300),
                nn.SELU(),
                nn.Linear(300,300),
                nn.SELU(),
                nn.Linear(300,300),
                nn.SELU(),
                nn.Linear(300,300),
                nn.SELU(),
                nn.Linear(300, 150),
                nn.SELU(),
                nn.Linear(150, 42),
                nn.SELU(),
                nn.Linear(42,1))

    def forward(self, X):
        return self.net(X)



class ResidualBlock(nn.Module):
    def __init__(self, size, bottleneck = False):
        super().__init__()
        in_size = size
        if bottleneck:
            bneck_size = int(size / 2)
        else:
            bneck_size = size
        self.layers = nn.Sequential(
            nn.Linear(size, bneck_size),
            nn.ReLU(),
            nn.Linear(bneck_size, size))
        self.rl_out = nn.ReLU()

    def forward(self, X):
        res = X
        out = self.layers(X)
        out += res
        return self.rl_out(out)

class ResNet(DeepSurv):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(1428, bottleneck = False),
            nn.Linear(1428, 600),
            nn.ReLU(),
            ResidualBlock(600, bottleneck = False),
            nn.Linear(600, 200),
            nn.ReLU(),
            ResidualBlock(200, bottleneck = True),
            nn.Linear(200,1))
    def forward(self, X):
        return self.net(X)


class LogNet(DeepSurv):
    # options to consider:
    # - make the out-layer larger than 2
    # - make the first couple layers of the logspace same in/out dimension
    # - use of activation functions in the linspace (and logspace? spicy)
    def __init__(self, dset = "cmb"):
        super().__init__()
        if dset == "cmb":
            self.linspace = nn.Sequential(
                nn.Linear(344, 200),
                # nn.Dropout(p = 0.1),
                nn.Linear(200,100),
                # nn.Dropout(p = 0.1),
                nn.Linear(100, 10),
                nn.Linear(10,1))
    
            self.logspace = nn.Sequential(
                nn.Linear(344, 200),
                # nn.Dropout(p = 0.1),
                nn.Linear(200,100),
                # nn.Dropout(p = 0.1),
                nn.Linear(100, 10),
                nn.Linear(10,1))
    
            self.out = nn.Linear(2,1)
            
        elif dset == "allprot":
            self.linspace = nn.Sequential(
                nn.Linear(1428, 700),
                # nn.Dropout(p = 0.1),
                nn.Linear(700,350),
                # nn.Dropout(p = 0.1),
                nn.Linear(350, 100),
                nn.Linear(100,1))
    
            self.logspace = nn.Sequential(
                nn.Linear(1428, 700),
                # nn.Dropout(p = 0.1),
                nn.Linear(700,350),
                # nn.Dropout(p = 0.1),
                nn.Linear(350, 100),
                nn.Linear(100,1))
    
            self.out = nn.Linear(2,1)

        else:
            raise ValueError(f'unknown dataset: {dset}')
        # self.linspace = nn.Sequential(
        #         nn.Linear(1428, 750),
        #         nn.SELU(),
        #         nn.Dropout(p= 0.2),
        #         nn.Linear(750,400),
        #         nn.SELU(),
        #         nn.Dropout(p= 0.1),
        #         nn.Linear(400,100),
        #         nn.SELU(),
        #         # nn.Dropout(p=0.1),
        #         nn.Linear(100,1))

    def forward(self, x):
        X_np = x.detach().numpy()

        # standardize the data (between 0 and 1)
        X_scaled = MinMaxScaler(feature_range = (0.001, 1)).fit_transform(X_np)
        x_lin = torch.Tensor(X_scaled).double()
        x_log = torch.log(torch.Tensor(X_scaled).double())


        lin_out = self.linspace(x_lin)
        log_out = torch.exp(self.logspace(x_log))
        full_out = torch.concat((lin_out, log_out), axis = 1)
        return self.out(full_out)
        
        
