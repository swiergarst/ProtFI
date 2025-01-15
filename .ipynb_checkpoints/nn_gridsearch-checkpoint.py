import wandb
from run_nn import nn_main
from src.utils import *
from src.nn_common import *

import argparse
from multiprocessing import Process
# import math

# parameters kept constant
parser = argparse.ArgumentParser(description = "running neural network experiments")
parser.add_argument("--epochs", type=int, default = 10, help = "number of epochs")
parser.add_argument("--gamma", type= float, default = 0.99, help = "learning rate scheduler parameter")
parser.add_argument("--beta", type= float, default = 0.1, help = "auto encoder loss function mixing weight")
parser.add_argument("--nruns", type = int, default = 1, help = 'how many runs to do')

#parameters to be sweeped
# parser.add_argument("--lr", type=float, default = 5e-6, help = "learning rate")
# parser.add_argument("--bs", type=int, default = 1000, help = "batch size")
# parser.add_argument("--wd", type=float, default = 1e-5, help = "weight decay (l2 regularization parameter)")

parser.add_argument("--lrmin", type=float, default = -8, help = "minimum learning rate exponent for sweep")
parser.add_argument("--lrmax", type=float, default = -4, help = "maximum learning rate exponent for sweep")

parser.add_argument("--bsmin", type=int, default = 32, help = "minimum batch size for sweep")
parser.add_argument("--bsmax", type=int, default = 2000, help = "maximum batch size for sweep")

parser.add_argument("--bsinnermin", type = int, default = 32, help = "minimum inner batch size")
parser.add_argument("--bsinnermax", type = int, default = 1000, help = "maximum inner batch size")

parser.add_argument("--wdmin", type=float, default = -5, help = "minimum weight decay exponent(l2 regularization parameter) for sweep")
parser.add_argument("--wdmax", type=float, default = -3, help = "maximum weight decay exponent(l2 regularization parameter) for sweep")

parser.add_argument("--betamin", type = float, default = 1, help = "minimum beta")
parser.add_argument("--betamax", type = float, default = 1, help = "maximum beta")


# these we won't sweep, but will try some configurations
parser.add_argument("--net", type = str, default = "ds_default", help = 'which network to use')
parser.add_argument("--dset", type=str, default = "allprot", help = 'which dataset to use (allprot or cmb)')
parser.add_argument("--target", type = str, default = "mort", help = "whether to train on mortality or frailty")

parser.add_argument("--counts", type = int, default = 1000, help = "the amount of configurations to try in the hyperparameter search")
parser.add_argument("--n_workers", type = int, default = 1, help = "amount of agents to use")
parser.add_argument("--sweep_pid", type = str, default ="0", help = "wandb sweep pid")
args = parser.parse_args()




def load_sweep_config(args):
    gs_params = ['lr', 'bs', 'wd','beta']# "nested_batch_size"]
    
    n_params = len(gs_params)
    n_counts_per_param = int(args.counts ** (1/n_params))
    
    
    cfg = {
        "lr" : {"values" : np.logspace(args.lrmin, args.lrmax, num = n_counts_per_param).tolist()},
        "bs" : {"values" : np.linspace(args.bsmin, args.bsmax, num = n_counts_per_param, dtype = int).tolist()},
        "wd" : {"values" : np.logspace(args.wdmin, args.wdmax, num = n_counts_per_param).tolist() },
        "beta": {"values": np.linspace(args.betamin, args.betamax, num = n_counts_per_param).tolist()}
        # "nested_batch_size": {"values" : np.linspace(args.bsinnermin, args.bsinnermax, num = n_counts_per_param).tolist()}
    }
    return cfg
                

def convert_config(run_config, wandb_config):
    print(f'wandb config: {wandb_config}')
    run_config['lr'] = wandb_config.lr
    run_config['batch_size'] = wandb_config.bs
    run_config['wd'] = wandb_config.wd

    return run_config


if __name__ == "__main__":


    run_config = load_config(
        n_epochs = args.epochs,
        lr = 0,
        weight_decay = 0,
        batch_size = 1,
        gamma = args.gamma,
        beta = args.beta,
        n_runs = 1,
        dset = args.dset,
        net = args.net,
        target = args.target,
        log_wandb = 1)
    
    sweep_params = load_sweep_config(args)
    sweep_configuration = {
        "method" : "grid",
        "metric" : {"goal" : "minimize", "name" : "validation_loss"},
        "parameters" : sweep_params}

    def gs_main():
        run_config = load_config(
            n_epochs = args.epochs,
            lr = 0,
            weight_decay = 0,
            batch_size = 1,
            gamma = args.gamma,
            beta = args.beta,
            n_runs = 1,
            dset = args.dset,
            net = args.net,
            target = args.target,
            log_wandb = 1)

        r = wandb.init(project = "olink-aging")
        # print(f'running main with parameters:')
        # print(f'batch size: {wandb.config.bs}')
        # print(f'learning rate: {wandb.config.lr}')
        # print(f'l2 regularization: {wandb.config.wd}')
        run_config = convert_config(run_config, wandb.config)
        nn_main(run_config)
    



    wandb.login(key = '483e19e5215d5e164b4cc3f6c3f85d5c3202eabc')
    wandb_config = {
        "architecture" : run_config['net'],
        "dataset" : run_config["dset"]
        }

    #config = {**run_config, **wandb_config})
    if args.sweep_pid == "0":
        sweep_id = wandb.sweep(sweep = sweep_configuration, project = "olink-aging")
    else:
        sweep_id = args.sweep_pid

    print(f'starting agent with sweep_id: {sweep_id}')
    wandb.agent(sweep_id, function = gs_main, project = "olink-aging")


    # def run_agent():
    #     wandb.agent(sweep_id, function = gs_main)

    # ps = []
    # for _ in range(args.n_workers):
    #     p = Process(target = run_agent)
    #     p.start()
    #     ps.append(p)

    # for p in ps:
    #     p.join()
    # wandb.agent(sweep_id, function = gs_main)