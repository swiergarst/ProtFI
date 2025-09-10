import numpy as np
import argparse
from src.utils import *
from src.nn_common import *
import torch
import datetime
import random



def nn_main(config):
    
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])


    
    if config['combine_sets'] == True:
        full_train, full_val, _, _ = get_data(config)
    else:
        full_train, full_val, _ , _, _ , _,cols = get_data(config)

    config['in_dim'] = full_train[0].shape[1]


    return training_loop(full_train, full_val, config)

    # return trained_net


def convert_truefalse(var):
    if var == 1:
        return True
    elif var == 0 :
        return False
    else:
        raise ValueError(f'unknown value to convert to true/false: {var}')

if __name__ == "__main__":

    # parse all config parameters from command line
    parser = argparse.ArgumentParser(description = "running neural network experiments")
    parser.add_argument("--epochs", type=int, default = 100, help = "number of epochs")
    parser.add_argument("--lr", type=float, default = 5e-6, help = "learning rate")
    parser.add_argument("--wd", type=float, default = 1e-5, help = "weight decay (l2 regularization parameter)")
    parser.add_argument("--bs", type=int, default = 1000, help = "batch size")
    parser.add_argument("--gamma", type= float, default = 0.99, help = "learning rate scheduler parameter")
    parser.add_argument("--dset", type=str, default = "cmb", help = 'which dataset to use (allprot/cmb/ffs)')
    parser.add_argument("--target", type = str, default = "frailty", help = "whether to train on mortality or frailty")
    parser.add_argument("--log", type = int, default = 0, help = "whether to log on weights and biases")
    parser.add_argument("--add_age", type=int, default = 0, help = "whether to add age to the input (automatically done for frailty predictions, here for extra testing)")
    parser.add_argument("--combine_sets", type = int, default = 1, help = "whether to combine train and test sets")
    parser.add_argument("--seed", type = int, default = 7, help = "random seed")
    parser.add_argument("--bootstrap", type=int, default = 0, help = "whether to run bootstrap or not")

    args = parser.parse_args()


    args.add_age = convert_truefalse(args.add_age)
    args.combine_sets = convert_truefalse(args.combine_sets)
    args.bootstrap = convert_truefalse(args.bootstrap)
    
    config = load_config(
        n_epochs = args.epochs,
        lr = args.lr,
        weight_decay = args.wd,
        batch_size = args.bs,
        gamma = args.gamma,
        dset = args.dset,
        target = args.target,
        log_wandb = args.log,
        add_age = args.add_age,
        combine_sets = args.combine_sets,
        seed = args.seed,
        bootstrap = args.bootstrap)
    

    # initialize logging to the weights and biases plaform: wandb.ai
    if config['log_wandb'] == 1:
        wandb.login(key = 'wandb_key', force = True) # change 'wand_key' to the API key of your wandb project
        wandb_config = {
            "architecture" : config['net'],
            "dataset" : config["dset"]
        }
        r = wandb.init(
            project = "wandb_project_name", # change to the name of your wandb project
            config = {**config, **wandb_config})


    
    net = nn_main(config)

    if config['log_wandb'] == 1:
        # date = datetime.date
        # date = datetime.date.today().isoformat()
        path = f'nn_models/model_{r.id}.pt'
        torch.save(net.state_dict(), path)
        wandb.finish()






