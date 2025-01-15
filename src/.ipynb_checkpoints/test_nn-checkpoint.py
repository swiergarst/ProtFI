import pytest
from utils import *

@pytest.mark.parametrize("dset", ['cmb', 'allprot'])
@pytest.mark.parametrize("net", ["ds_default", "ds_lognet", "ae_default", "ae_double_out", "ae_combined"])
@pytest.mark.parametrize("target", ['mort', 'frailty'])
def test_main_loop( dset, net, target, n_epochs = 3, lr = 5e-6, weight_decay=1e-5, batch_size=1000, gamma=0.99, beta = 0.1, n_runs=1, log_wandb = False):
    
    config = load_config(
        n_epochs = n_epochs,
        lr = lr,
        weight_decay = weight_decay,
        batch_size = batch_size,
        gamma = gamma,
        beta = beta,
        n_runs = n_runs,
        dset = dset,
        net = net,
        target = target,
        log_wandb = log_wandb)

    full_train, full_val = get_data(config)

    training_loop(full_train, full_val, config)