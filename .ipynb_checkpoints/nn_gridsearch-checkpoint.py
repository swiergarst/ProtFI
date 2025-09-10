import wandb
from run_nn import nn_main
from src.utils import *
from src.nn_common import *

import argparse
from multiprocessing import Process

# parameters kept constant
parser = argparse.ArgumentParser(description = "running neural network experiments")
parser.add_argument("--epochs", type=int, default = 10, help = "number of epochs")
parser.add_argument("--gamma", type= float, default = 0.99, help = "learning rate scheduler parameter")


#parameters to be sweeped
parser.add_argument("--lrmin", type=float, default = -8, help = "minimum learning rate exponent for sweep")
parser.add_argument("--lrmax", type=float, default = -4, help = "maximum learning rate exponent for sweep")

parser.add_argument("--bsmin", type=int, default = 32, help = "minimum batch size for sweep")
parser.add_argument("--bsmax", type=int, default = 2000, help = "maximum batch size for sweep")

parser.add_argument("--wdmin", type=float, default = -5, help = "minimum weight decay exponent(l2 regularization parameter) for sweep")
parser.add_argument("--wdmax", type=float, default = -3, help = "maximum weight decay exponent(l2 regularization parameter) for sweep")


# these we won't sweep, but will try some configurations
parser.add_argument("--dset", type=str, default = "cmb", help = 'which dataset to use (allprot/cmb/cmb_ffs)')
parser.add_argument("--target", type = str, default = "mort", help = "whether to train on mortality or frailty (mort/frailty)")

parser.add_argument("--counts", type = int, default = 1000, help = "the amount of configurations to try in the hyperparameter search")
parser.add_argument("--sweep_pid", type = str, default ="0", help = "wandb sweep pid")
args = parser.parse_args()




def load_sweep_config(args):
    gs_params = ['lr', 'bs', 'wd']
    
    n_params = len(gs_params)
    n_counts_per_param = int(args.counts ** (1/n_params))
    
    
    cfg = {
        "lr" : {"values" : np.logspace(args.lrmin, args.lrmax, num = n_counts_per_param).tolist()},
        "bs" : {"values" : np.linspace(args.bsmin, args.bsmax, num = n_counts_per_param, dtype = int).tolist()},
        "wd" : {"values" : np.logspace(args.wdmin, args.wdmax, num = n_counts_per_param).tolist() },
    }
    return cfg
                

def convert_config(run_config, wandb_config):
    print(f'wandb config: {wandb_config}')
    run_config['lr'] = wandb_config.lr
    run_config['batch_size'] = wandb_config.bs
    run_config['weight_decay'] = wandb_config.wd


    return run_config


if __name__ == "__main__":


    # run_config = load_config( # these values don't matter
    #     n_epochs = args.epochs,
    #     lr = 0,
    #     weight_decay = 0,
    #     batch_size = 1,
    #     gamma = args.gamma,
    #     dset = args.dset,
    #     net = args.net,
    #     target = args.target,
    #     log_wandb = 1,
    #     add_age = args.add_age)
    
    sweep_params = load_sweep_config(args)
    sweep_configuration = {
        "method" : "grid",
        "metric" : {"goal" : "minimize", "name" : "validation_loss"},
        "parameters" : sweep_params}

    def gs_main():
        wandb_project_name = "my_wandb_project" # change this to your project name
        run_config = load_config( # hard-coded values are set by convert_config()
            n_epochs = args.epochs,
            lr = 0,
            weight_decay = 0,
            batch_size = 1,
            gamma = args.gamma,
            dset = args.dset,
            target = args.target,
            log_wandb = 1)

        r = wandb.init(project = wandb_project name)
        # print(f'running main with parameters:')
        # print(f'batch size: {wandb.config.bs}')
        # print(f'learning rate: {wandb.config.lr}')
        # print(f'l2 regularization: {wandb.config.wd}')
        run_config = convert_config(run_config, wandb.config)
        nn_main(run_config)
    



    wandb.login(key = 'wanb_key') # change this to your wandb project key
    wandb_config = {
        "architecture" : run_config['net'],
        "dataset" : run_config["dset"]
        }

    if args.sweep_pid == "0":
        sweep_id = wandb.sweep(sweep = sweep_configuration, project = wandb_project_name)
    else:
        sweep_id = args.sweep_pid

    print(f'starting agent with sweep_id: {sweep_id}')
    wandb.agent(sweep_id, function = gs_main, project = wandb_project_name) # change project to your wandb project name

