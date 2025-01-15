import torch
import torch.nn as nn
import numpy as np
from .nn_common import neg_log_likelihood
from .constants import *
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import r2_score




class ft_mort_model(nn.Module):
    def __init__(self, in_dim = 344):
        super().__init__()
        # self.target = target
        self.in_dim = in_dim

        # ## define architecture
        # self.shared_layers = nn.Sequential(
        #     nn.Linear(in_dim, 200),
        #     nn.Dropout(p = 0.2),
        #     nn.ReLU(),
        #     nn.Linear(200,100),
        #     nn.Dropout(p = 0.1),
        #     nn.ReLU(),
        #     nn.Linear(100,10),
        #     nn.ReLU())
        # self.ft_head = nn.Linear(10,1)
        # self.mort_head = nn.Linear(10,1)

        ## alternative arch
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 200),
            nn.Dropout(p = 0.2),
            nn.ReLU(),
            nn.Linear(200,100),
            nn.Dropout(p = 0.1),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.ReLU(),
            nn.Linear(10,2))
        # self.ft_head = nn.Linear(10,1)
        # self.mort_head = nn.Linear(10,1)
    

    def forward(self, x):
        # shared_out = self.shared_layers(x)
        # out_ft = self.ft_head(shared_out)
        # out_mort = self.mort_head(shared_out)
        # return torch.cat((out_mort, out_ft), dim = 1)
        return self.layers(x)

    def train_net(self, opt, X, Y, T, E, beta):

        out = self.forward(X)
        loss = self.get_loss(out, Y, T, E, beta)
        # loss = ft_loss + beta * mort_loss

        loss.backward()
        opt.step()

    def test(self, X, Y, T, E, beta):
        out = self.forward(X)
        ft_out = out[:,1].reshape(-1, 1)
        mort_out = out[:,0].reshape(-1, 1)

        loss = self.get_loss(out, Y, T, E, beta)
         
        r2score = r2_score(Y.detach().numpy(), ft_out.detach().numpy())
        conc = concordance_index_censored(E.detach().numpy().astype(bool), T.detach().numpy(), mort_out.detach().numpy()[:,0])[0]
        return loss, r2score, conc


    def get_loss(self, out, Y, T, E, beta):
        mort_out = out[:,0].reshape(-1, 1)
        ft_out = out[:,1].reshape(-1, 1)

        ft_loss_fn = nn.MSELoss()
        ft_loss = ft_loss_fn(ft_out, Y)

        # ignore mortality loss if there are no deaths in batch
        if torch.sum(E) == 0:
            mort_loss = 0
        else:
            mort_loss = neg_log_likelihood(mort_out, T, E)
        loss = (1-beta)* ft_loss + beta * mort_loss       
        # loss = ft_loss + beta * mort_loss       

        return loss


class ft_components_model(nn.Module):
    def __init__(self, in_dim=344):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 250),
            nn.Dropout(p = 0.2),
            nn.ReLU(),
            nn.Linear(250,150),
            nn.Dropout(p = 0.1),
            nn.ReLU(),
            nn.Linear(150,100),
            nn.ReLU(),
            nn.Linear(100, len(FT_COMPONENTS)))

        # heads_list = []
        # self.heads_order = []
        # for frailty_component in BASE_FT_COMPONENTS:
        #     output_dim = sum(subcomponent.startswith(frailty_component) for subcomponent in FT_COMPONENTS) 
        #     head = nn.Linear(10, output_dim)
        #     heads_list.append(head)
        #     self.heads_order.append(frailty_component)
    
        # self.heads = nn.ModuleList(heads_list)

    def forward(self, X):
        # shared_out = self.shared_layers(X)

        # output = [head(shared_out) for head in self.heads]
        # return torch.cat(output)
        return self.layers(X)

    def get_component_vals(self, net_out, components, all_component_names, base_component_name):
        # print(f'base component name: {base_component_name}')
        # print(f'all component names:{all_component_names}')
        # find all the values associated with the same output

        # idx's of nn output layer
        output_component_idx = np.where([component.startswith(base_component_name) for component in FT_COMPONENTS])[0]
        # idx of frailty components dataframe
        Y_idx = np.where(all_component_names == base_component_name)[0]

        # print(f'bla: { np.where([component.startswith(base_component_name) for component in all_component_names])}')
        # select those outputs/label vectors
        Y_component = components[:,Y_idx]

        print(f'output component: {base_component_name}, values: {Y_component.unique()}, shape: {Y_component.shape}')
        out_component = net_out[:,output_component_idx]

        tY_component= torch.Tensor(list(map(self.frailty_to_output, Y_component.detach().numpy()))).long()        


        # print(f'Y component shape: {Y_component.shape}')

        return out_component, tY_component
    
    def train_net(self,opt, X, components, component_names):
        full_out = self.forward(X)

        loss = 0
        for i, base_component in enumerate(BASE_FT_COMPONENTS):
            self.current_base_component = base_component
            out_component, Y_component = self.get_component_vals(full_out, components, component_names, base_component)
            
            # calculate loss
            crit = nn.CrossEntropyLoss()
            component_loss = crit(out_component, Y_component)

            loss += component_loss
            
        loss.backward()
        opt.step()          
        # for i, head_name in enumerate(self.heads_order):
        #     crit = nn.CrossEntropyLoss()
        #     label_idx = np.where(component_names == head_name)[0]
        #     head_out = self.heads[label_idx](emb)
        #     component_idx = np.where([subcomponent.startswith(head_name) for subcomponent in component_names]== 1)[0]
        #     ft_component = components[:, component_idx]
        #     head_loss = crit(head_out, ft_component)
        #     loss += head_loss
        


  

    def test_net(self, X, frailty, components, component_names):
        per_task_accuracy = np.zeros(len(BASE_FT_COMPONENTS))
        emb = self.forward(X)

        loss = 0
        ft_est = np.zeros(X.shape[0])
        for i, base_component in enumerate(BASE_FT_COMPONENTS):
            self.current_base_component = base_component
            out_component, Y_component = self.get_component_vals(emb, components, component_names, base_component)

            # calculate loss
            crit = nn.CrossEntropyLoss()
            component_loss = crit(out_component, Y_component)

            loss += component_loss

            # print(f'out_component shape: {out_component.shape}')
            _, pred = torch.max(out_component.data, 1) # get prediction
            pred_list = pred.detach().numpy().tolist()
    
            ft_component_est = list(map(self.output_to_frailty, pred_list))
            # ft_component_est = self.convert_output_to_frailty_score(pred, base_component)
            # print(f'pred shape: {pred.shape}')

            
            ft_est += ft_component_est
            
            # calculate accuracy of the specific task
            correct = (pred == Y_component).sum().item()
            task_acc = correct/ X.size()[0]
            per_task_accuracy[i] = task_acc
            
        
        ft_est /= len(BASE_FT_COMPONENTS)

        r2 = r2_score(ft_est, frailty)
            
        return loss, per_task_accuracy, r2
        

    def mapping_logic(self):
        component_name = self.current_base_component
        if (component_name == "srhealth_0") or component_name == "fatigue_0":
            mapping = [0., 0.25, 0.5, 1.]
        elif (component_name == "insomnia_0") or component_name == "falls_0":
            mapping = [0., 0.5, 1.]
        elif (component_name == "depressed_0"):
            mapping = [0., 0.5, 0.75, 1.]
        else:
            mapping = [0., 1.]

        return mapping
    # conversion between predicted output neuron idx and specific frailty component score
    def output_to_frailty(self, mapping_idx):
        mapping = self.mapping_logic()
        return mapping[mapping_idx]
        

    def frailty_to_output(self, pred_val):
        mapping = self.mapping_logic()
        # print(f'mapping: {mapping}')
        # print(f'current component name: {self.current_base_component}')
        return mapping.index(pred_val[0])
            # label_idx = np.where(component_names == head_name)[0]
            # component_idx = np.where([subcomponent.startswith(head_name) for subcomponent in component_names]== 1)[0]
            # ft_component = components[:, component_idx]

            # head_out = self.heads[label_idx](emb)     
        

## deprecated, here for reference
 # def mt_test_net_frailty(self, X, Y):
 #        ft_out = self.mt_forward(X, 'frailty')
 #        crit = nn.MSELoss()
 #        loss = crit(ft_out, Y)
 #        score = r2_score(Y.detach().numpy(), ft_out.detach().numpy())

 #        return loss.detach().numpy(), score

 #    def mt_test_net_mort(self, X, T, E):
 #        mort_out = self.mt_forward(X, 'mort')
        
 #        surv_loss = neg_log_likelihood(mort_out, X, T, E)
 #        conc = concordance_index_censored(E.detach().numpy().astype(bool), T.detach().numpy(), mort_out.detach().numpy()[:,0])[0]
 #        return surv_loss.detach().numpy(), conc, None
    
 #    def mt_train_net_frailty(self, opt, X, Y):
 #        out = self.mt_forward(X, 'frailty')
 #        ft_loss_fn = nn.MSELoss()
 #        ft_loss = ft_loss_fn(out, Y)
 #        ft_loss.backward()
 #        opt.step()
    
 #    def mt_train_net_mort(self, opt, X, T, E):
 #        out = self.mt_forward(X, 'mort')
 #        mort_loss = neg_log_likelihood(out, X, T, E)
 #        mort_loss.backward()
 #        opt.step()
        
 #    def train_net_ft_mort_naive(self, opt, X, Y, T, E, beta = 0):
 #        out = self.forward(X)
 #        mort_out = out[:,0].reshape(-1, 1)
 #        ft_out = out[:,1].reshape(-1, 1)

 #        ft_loss_fn = nn.MSELoss()
        
 #        ft_loss = ft_loss_fn(ft_out, Y)

 #        # ignore mortality loss if there are no deaths in batch
 #        if torch.sum(E) == 0:
 #            mort_loss = 0
 #        else:
 #            mort_loss = neg_log_likelihood(mort_out, X, T, E)
 #        # loss = (1-beta)* ft_loss + beta * mort_loss
 #        loss = ft_loss + beta * mort_loss

 #        loss.backward()
 #        opt.step()

 #    def train_net_seq(self,opt, X, Y, T, E, nested_bs):
 #        # update for mortality
 #        self.mt_train_net_mort(opt, X, T, E)

 #        ft_batch_loader = DataLoader(TensorDataset(X, Y), batch_size = nested_bs, shuffle=False)
 #        for X_b, Y_b in ft_batch_loader:
 #            self.mt_train_net_frailty(opt, X_b, Y_b)

 #    def train_net_ft_mort(self, opt,  mort_train_loader, ft_train_loader):
 #        ratio = int(len(ft_train_loader) / len(mort_train_loader))
 #        for X_mort, T, E in mort_train_loader:
 #            opt.zero_grad()
 #            self.mt_train_net_mort(opt, X_mort, T, E)
 #            for _ in range(ratio):
 #                X_ft, Y = next(iter(ft_train_loader))
 #                opt.zero_grad()
 #                self.mt_train_net_frailty(opt, X_ft, Y)