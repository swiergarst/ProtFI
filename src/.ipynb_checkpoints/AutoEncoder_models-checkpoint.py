import torch
import torch.nn as nn
import numpy as np
from .nn_common import neg_log_likelihood, rank_loss
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from jointomicscomp.src.PoE.model import PoE as model
from jointomicscomp.src.PoE.train import loss_function

# from utils import *


class PoE(model):
    def __init__(self, dset = "cmb", target = "mort"):
        n_feat2 =  451 - 344
        if target == "frailty": 
            n_feat2 += 1
            self.lr = LinearRegression()
        else:
            self.coxph = CoxnetSurvivalAnalysis() # use default vals for now
         # args['dropout_probability'], args['use_batch_norm'], args['log_inputs']
        args = {
            "latent_dim" : "128-64",
            "num_features1" : 344,
            "num_features2" : n_feat2,
            'dropout_probability': 0.1,
            'use_batch_norm' : True,
            'log_inputs' : False,
            'likelihood1': 'normal',
            'likelihood2' : 'normal',
            'data2' : "not_ATAC",
            'cuda' : False
        }
        
        super().__init__(args)

    def calc_loss(self, X_train, beta):
        # split up X_train into the two modalities
        X_t1 = X_train[:,:344]
        X_t2 = X_train[:,344:]

        # forward pass using both omics
        joint_rec1, joint_rec2, joint_mu, joint_logvar = self.forward(X_t1, X_t2)

        # forward pass with single omics
        prot_rec1, prot_rec2, prot_mu, prot_logvar = self.forward(omic1 = X_t1)
        met_rec1, met_rec2, met_mu, met_logvar = self.forward(omic2 = X_t2)

        # calculate loss function for all passes
        joint_loss = loss_function(joint_rec1, X_t1,
                                   joint_rec2, X_t2, 
                                   joint_mu, joint_logvar, beta) 
        prot_loss = loss_function(prot_rec1, X_t1,
                                  prot_rec2, X_t2,
                                  prot_mu, prot_logvar, beta)
        met_loss = loss_function(met_rec1, X_t1, 
                                 met_rec2, X_t2,
                                 met_mu, met_logvar, beta)
        
        tot_loss = joint_loss['loss'] + prot_loss['loss'] + met_loss['loss']
        tot_kld_loss = joint_loss['KLD'] + prot_loss['KLD'] + met_loss['KLD']
        return tot_loss, tot_kld_loss
        

    def train_net_mort(self, opt, X_train, T_train, E_train, beta, X_full = None, T_full = None, E_full = None):
        loss, kld_loss = self.calc_loss(X_train, beta)
        loss.backward()
        opt.step()
        
        y = np.array([(e1, e2) for e1, e2 in np.vstack((E_full.detach().numpy(), T_full.detach().numpy())).T], dtype = [("Status", "?"), ("Survival_in_days", "<f8")])

        X_f1 = X_full[:,:344]
        X_f2 = X_full[:,344:]
        z_joint, _, _, _, _, _, _= self.embedAndReconstruct(X_f1, X_f2)
        self.coxph.fit(z_joint.detach().numpy(), y)
        
        return loss.detach().numpy()

    def train_net_frailty(self, opt, tX_batch, tY_batch, beta,  X_full = None, Y_full = None):
        loss, kld_loss = self.calc_loss(tX_batch, beta)
        loss.backward()
        opt.step()

        X_f1 = X_full[:,:344]
        X_f2 = X_full[:,344:]
        z_joint, _, _, _, _, _, _ = self.embedAndReconstruct(X_f1, X_f2)
        
        self.lr.fit(z_joint, Y_full)

        return loss.detach().numpy()
    
    def test_mort(self, X, T, E, beta):
        X_f1 = X[:,:344]
        X_f2 = X[:,344:]
        z_joint, _, _, _, _, _, _ = self.embedAndReconstruct(X_f1, X_f2)

        loss, kld_loss = self.calc_loss(X, beta)


        y = np.array([(e1, e2) for e1, e2 in np.vstack((E.detach().numpy(), T.detach().numpy())).T], dtype = [("Status", "?"), ("Survival_in_days", "<f8")])
        conc = self.coxph.score(z_joint.detach().numpy(), y)
        
        # forward pass using both omics
        joint_rec1, joint_rec2, joint_mu, joint_logvar = self.forward(omic1 = X_f1, omic2 = X_f2)

        print(f'tensor shapes: {joint_rec1.shape}, {joint_rec2.shape}')
        joint_rec_full = torch.cat((joint_rec1, joint_rec2), dim = 1)
        cos = nn.CosineSimilarity(dim=1)

        cos_res = cos(joint_rec_full, X)
        sim = torch.mean(cos_res)
        
        return loss.detach().numpy(), conc, sim.detach().numpy()

    def test_frailty(self, X, Y, beta):
        X_f1 = X[:,:344]
        X_f2 = X[:,344:]
        z_joint, _, _, _, _, _, _ = self.embedAndReconstruct(X_f1, X_f2)

        tot_loss, kld_loss = self.calc_loss(X, beta)

        pred = self.lr.predict(z_joint.detach().numpy())
        pred_loss = mean_squared_error(y, pred) 
        
        r2 = self.lr.score(z_joint.detach().numpy(), y)

        # cos_res = cos(dec, X)
        # sim = torch.mean(cos_res).detach().numpy()

        return tot_loss.detach().numpy(), r2




class auto_encoder(nn.Module):
    # default auto encoder. only reconstruction loss is used
    def __init__(self,latent_dim = 50, dset = 'cmb'):
        super().__init__()
        self.latent_dim = latent_dim
        if dset == "allprot":
            self.encoder = nn.Sequential(
                nn.Linear(1428, 1000),
                # nn.Sigmoid(),
                nn.ReLU(),
                nn.Dropout(p= 0.1),
                nn.Linear(1000, 750),
                # nn.Sigmoid(),
                nn.ReLU(),
                nn.Linear(750, 500),
                # nn.Sigmoid(),
                nn.ReLU(),
                nn.Dropout(p = 0.1),
                nn.Linear(500,250),
                nn.ReLU(),
                nn.Linear(250, self.latent_dim))
                # nn.ReLU())
                # nn.Sigmoid())
    
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 250),
                nn.ReLU(),
                nn.Linear(250, 500),
                nn.ReLU(),
                nn.Dropout(p = 0.1),
                nn.Linear(500, 750),
                # nn.Sigmoid(),
                nn.ReLU(),
                nn.Linear(750, 1000),
                # nn.Sigmoid(),
                nn.ReLU(),
                nn.Dropout(p = 0.1),
                nn.Linear(1000, 1428))
        elif dset == 'cmb':
            self.encoder = nn.Sequential(
                nn.Linear(344, 200),
                nn.Dropout(p= 0.2),
                nn.ReLU(),
                nn.Linear(200, 100),
                nn.Dropout(p = 0.1),
                nn.ReLU(),
                nn.Linear(100, self.latent_dim))
                # nn.ReLU())
                # nn.Sigmoid())
    
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 100),
                nn.Dropout(p = 0.2),
                nn.ReLU(),
                nn.Linear(100, 200),
                nn.Dropout(p = 0.1),
                nn.ReLU(),
                nn.Linear(200, 344))
            

    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)
    def encode(self, x):
        return self.encoder(x)
    def decode(self, x):
        return self.decoder(x)

    def test_mort(self, x, T = None, E = None, beta = None):
        # we do cosine similarity as well as test loss (MSE)
        cos = nn.CosineSimilarity(dim=1)
        mse = nn.MSELoss()

        with torch.no_grad():
            out = self.forward(x)

            loss = mse(out, x)
            cos_res = cos(out, x)
            # print(cos_res.shape)
            sim = torch.mean(cos_res)

        return loss, None, sim


    def train_net_mort(self, opt, X_train,T_train = None, E_train = None, beta = None ):
        out = self.forward(X_train)
        
        crit = nn.MSELoss()

        loss = crit(out, X_train)
        loss.backward()
        opt.step()
        return loss.detach().numpy()

    def test_frailty(self, X, Y, beta = None):
        loss, _ , _ = self.test_mort(X)
        return loss

    def train_net_frailty(self,opt, X, Y, beta = None):
        return self.train_net_mort(opt, X)

class AE_prior(auto_encoder):
    def __init__(self, dset='cmb'):
        super().__init__(dset = dset)
        # to think about: should this be an elastic net or not?
        self.coxph = CoxnetSurvivalAnalysis() # use default values for now  
        
    def forward(self, x):
        x = self.encode(x)
        return x, self.decode(x)
    
    def train_net_mort(self, opt, X_train, T_train, E_train, beta):
        enc, dec = self.forward(X_train)

        out_crit = nn.MSELoss()
        
        prior_loss = rank_loss(dec, X_train, T_train, E_train)
        rec_loss = out_crit(dec, X_train)
        
        tl = self.total_loss(enc, dec, X_train, T_train, E_train, beta)
        tl.backward()
        opt.step()
        
        y = np.array([(e1, e2) for e1, e2 in np.vstack((E_train.detach().numpy(), T_train.detach().numpy())).T], dtype = [("Status", "?"), ("Survival_in_days", "<f8")])
        self.coxph.fit(X_train, y)
        
        return total_loss.detach().numpy()

    def total_loss(self, enc, dec, X, T, E, beta):
        crit = nn.MSELoss()
        rec_loss = crit(dec, X)
        prior_loss = rank_loss(dec, X, T, E)
        return (1-beta) * rec_loss + beta * rank_loss
        
    def test_mort(self, X, T, E, beta):
        enc, dec = self.forward(X)

        loss = self.total_loss(enc, dec, X, T, E, beta).detach().numpy()

        cos = nn.CosineSimilarity(dim=1)

        y = np.array([(e1, e2) for e1, e2 in np.vstack((E.detach().numpy(), T.detach().numpy())).T], dtype = [("Status", "?"), ("Survival_in_days", "<f8")])
        conc = self.coxph.score(X.detach().numpy(), y)
        
        cos_res = cos(dec, X)
        sim = torch.mean(cos_res).detach().numpy()

        return loss, conc, sim
        





class AE_double_out(auto_encoder):
    # autoencoder where the final output neuron encodes the survival output
    def __init__(self, dset = 'cmb'):
        super().__init__(dset = dset)
        self.latent_dim = 50

        if dset == 'allprot':
            self.decoder = nn.Sequential(
                    nn.Linear(self.latent_dim, 250),
                    nn.ReLU(),
                    nn.Linear(250, 500),
                    # nn.Sigmoid(),
                    nn.ReLU(),
                    nn.Dropout(p = 0.1),
                    nn.Linear(500, 750),
                    # nn.Sigmoid(),
                    nn.ReLU(),
                    nn.Linear(750, 1000),
                    # nn.Sigmoid(),
                    nn.ReLU(),
                    nn.Dropout(p = 0.1),
                    nn.Linear(1000, 1429))
        elif dset == 'cmb':
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 100),
                nn.ReLU(),
                nn.Dropout(p = 0.1),
                nn.Linear(100, 200),
                # nn.Sigmoid(),
                nn.ReLU(),
                nn.Linear(200, 345))

    def test_mort(self, x, T, E, beta):
        # we do cosine similarity as well as test loss (MSE)
        cos = nn.CosineSimilarity(dim=1)
        mse = nn.MSELoss()

        with torch.no_grad():
            out = self.forward(x)
            # print(f'out shape: {out.shape}')
            rec_loss = mse(out[:,:-1], x)
            surv_loss =  neg_log_likelihood(out[:,-1].reshape(-1, 1), x, T, E)
            
            cos_res = cos(out[:,:-1], x)
            # print(cos_res.shape)
            sim = torch.mean(cos_res)
            conc = concordance_index_censored(E.detach().numpy().astype(bool), T.detach().numpy(), out[:,-1].detach().numpy())[0]
            
            full_loss = (1 - beta) *rec_loss + beta * surv_loss

        
        return full_loss.detach().numpy(),  conc, sim.detach().numpy()

    def train_net_mort(self, opt, X_train, T_train, E_train, beta):
        out = self.forward(X_train)

        rec_loss = nn.MSELoss()

        loss1 = rec_loss(out[:,:-1], X_train)
        loss2 = neg_log_likelihood(out[:,-1].reshape(-1, 1), X_train, T_train, E_train)
        full_loss = (1 - beta) *loss1 + beta * loss2

        full_loss.backward()
        opt.step()

        return loss1.detach().numpy(), loss2.detach().numpy()

    def test_frailty(self, X, Y, beta):
        out = self.forward(X)

        rec_loss_ = nn.MSELoss()
        acc_loss_ = nn.MSELoss()

        rec_loss = rec_loss_(out[:,:-1], X)
        acc_loss = acc_loss_(out[:,-1].reshape(-1, 1), Y)
        full_loss = (1 - beta) * rec_loss + beta * acc_loss
        return full_loss.detach().numpy()

    def train_net_frailty(self, opt, X, Y, beta):
        out = self.forward(X)
        rec_loss_ = nn.MSELoss()
        acc_loss_ = nn.MSELoss()
        
        rec_loss = rec_loss_(out[:,:-1], X)
        acc_loss = acc_loss_(out[:,-1].reshape(-1, 1),Y)
        full_loss = (1 - beta) *rec_loss + beta * acc_loss

        full_loss.backward()
        opt.step()


class AE_combined(auto_encoder):
    #trained on a combination of survival and reconstruction loss 
# class AE_combined(nn.Module):

    def __init__(self, dset = 'cmb'):

        super().__init__(latent_dim = 10, dset = dset)
        self.regressor  = nn.Sequential(
            nn.Linear(self.latent_dim, 1))

    
    def forward(self, x):
        lspace = self.encoder(x)
        surv_out = self.regressor(lspace)
        # surv_out = lspace
        dec_out = self.decoder(lspace)

        return surv_out, dec_out
        # return dec_out

    def test_mort(self, X, T, E, beta):
        # we do cosine similarity as well as test loss 
        cos = nn.CosineSimilarity(dim=1)
        mse = nn.MSELoss()

        with torch.no_grad():
            surv_out, dec_out = self.forward(X)
            # dec_out = self.forward(X)

            surv_loss = neg_log_likelihood(surv_out, X, T, E)

            rec_loss = mse(dec_out, X)
            
            cos_res = cos(dec_out, X)
            sim = torch.mean(cos_res)
            
            conc = concordance_index_censored(E.detach().numpy().astype(bool), T.detach().numpy(), surv_out.detach().numpy()[:,0])[0]
            full_loss = (1 - beta) *rec_loss + beta * surv_loss

        return full_loss.detach().numpy(), conc, sim.detach().numpy()


    def train_net_mort(self, opt,  X_train, T_train, E_train, beta):
        rec_loss = nn.MSELoss()

        surv_out, dec_out = self.forward(X_train)

        
        loss1 = rec_loss(dec_out, X_train)
        loss2 = neg_log_likelihood(surv_out, X_train, T_train, E_train)
        full_loss = (1 - beta) *loss1 + beta * loss2

        full_loss.backward()
            # loss1.backward()
        opt.step()
        return loss1.detach().numpy(), loss2.detach().numpy()

    def test_frailty(self, X, Y, beta):
        frail_out, dec_out = self.forward(X)

        rec_loss_ = nn.MSELoss()
        acc_loss_ = nn.MSELoss()

        rec_loss = rec_loss_(dec_out, X)
        acc_loss = acc_loss_(frail_out, Y)
        full_loss = (1 - beta) * rec_loss + beta * acc_loss
        return full_loss.detach().numpy()

    def train_net_frailty(self, opt, X, Y, beta):
        frail_out, dec_out = self.forward(X)

        rec_loss_ = nn.MSELoss()
        acc_loss_ = nn.MSELoss()
        
        rec_loss = rec_loss_(dec_out, X)
        acc_loss = acc_loss_(frail_out,Y)
        full_loss = (1 - beta) *rec_loss + beta * acc_loss

        full_loss.backward()
        opt.step()

