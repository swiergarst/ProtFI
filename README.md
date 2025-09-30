This repository holds the code to reproduce the figures and analyses presented in [ProtFI, an efficient frailty-trained proteomics-based biomarker of aging, robustly predicts age-related decline](https://www.medrxiv.org/content/10.1101/2025.09.19.25336152v1). For using the developed biomarkers (protFI / protMort), we refer to [RS_association_analysis.ipynb](RS_association_analysis.ipynb), which holds examples of association analyses on an independent test set. The rest of this readme will elaborate on how to reproduce the model training procedure presented in the paper.

You can also calculate ProtFI and ProtMort via the R-package ProtBiom. 
Therefore, you first need to install `devtools` (if not already installed) via
```
install.packages('devtools')
```
To install the package you need to run

```
devtools::install_github("swiergarst/ProtFI", subdir = "ProtBiom")
```
To load the package the command is
```
library('ProtBiom')
```
A more in depth README is available in the ProtBiom folder.

# Installation

A conda environment can be installed using the `env_file.txt`, by running 
```
    conda create --name protFI --file env_file.txt
```

# ElasticNet models

For development of the ElasticNet models the Python packages sklearn and sksurv were used. We created a function find_best_model in model_functions.py. 

```
find_best_model(dset, target, bootstrap=False, quickbootstrap=False, combine=False)
```

find_best_model **needs** input on:
* the dataset name (e.g., 'allprot' for the set of 1428 proteins, cmb for the Cardiometabolic panel, etc.),
* the target variable ('frailty' for the (Rockwood) frailty index or 'mort' for all-cause mortality)

Additionally it is possible to give input on:
* whether we want to run a bootstrap analysis
* whether we want a quick bootstrap
* whether we want to combine the train and test set into a larger set

## Hyperparameter optimization
Running `python model_functions.py 'dset' 'target'` will run the `find_best_model` from bash on the training set. The target, either the frailty index or all-cause mortality, determines whether either `find_best_lm_model` runs an ElasticNetCV or `find_best_coxph_model` runs a CoxnetSurvivalAnalysis.

The functions `find_best_lm_model` (for frailty index) and `find_best_coxph_model` (for all-cause mortality) both receive the trainset and run model optimization with 5-fold cross-validation and an l1_ratio grid ranging from 0.1 to 1 in steps of 0.01. For each l1_ratio, the corresponding alpha and performance score are stored: mean squared error for the linear model (`ElasticNetCV`) and concordance index for the survival model (`CoxPH` with `GridSearchCV` and `KFold`). The best model is defined as the one with the lowest mean squared error (frailty index) or highest concordance index (mortality). The selected l1_ratio, alpha, and coefficients are saved, and the function returns the best model. Using validate_model, this model is then applied to set2 to assess performance in both train and test sets (R² for frailty index, concordance index for mortality), and to obtain biomarker values for the test set. All outputs are stored in the folder output_linear.



## Model training
Based on the performance metrics (either R² or concordance index) in the test set we selected our final models. Next, for these selected models we used the `--combine` flag to retrieve the final coefficients, using the hyperparameters as determined on the 70% training data, on the 90% set (train + test set). Based on the target, `find_best_model` calls for  `find_best_lm_model` (for frailty) or `find_best_coxph_model` (for all-cause mortality). Now, no nested cross-validation is run but the final hyperparameters are the inputs for the model. Using `validate_model` the performance metrics of the the 90% set, and 2) 10% set (validation set), will be returned. Furthermore, `validate_model` returns the biomarker values for the validation set. 

## Bootstrap
To check the stability of the resulting biomarker, we use bootstrapping with replacement on the 90% training set.  
When the `--bootstrap` flag is set, `find_best_model` calls either `find_best_lm_model` (for the frailty index) or `find_best_coxph_model` (for all-cause mortality). In each run, the hyperparameters chosen on the training set are reused.  

By default, 100 bootstrap cycles are performed.  
For quick testing, a `--quickbootstrap` flag exists that runs only 2 cycles.

The bootstrap procedure saves the coefficients, biomarker scores for the individuals in the 10%-set and the model performance metrics for each bootstrap cycle in: `./output_linear/bootstrap/{target}/`, where `{target}` is either `frailty` or `mort`.  

# deep learning models

## using weights and biases
for development of the deep learning models, the [Weights and Biases](wandb.ai) (wandb) platform was used. In order to use wandb, modify line 81 in `run_nn.py` :
```
    wandb.login(key = 'wandb_key', force = True) # change 'wand_key' to the API key of your wandb project
```
to use the API key of your own wandb project instead of 'wandb_key'. Furthermore, change line 87 (same script, `run_nn.py`):
```
 project = "wandb_project_name", # change to the name of your wandb project
```
to use the project name of your wandb project instead of 'wandb_project_name'. Finally, make sure you run `run_nn.py` with the `log_wandb` option enabled, e.g.

```
    python run_nn.py --log_wandb 1
```

If you wish to run similar gridsearches as presented in the paper, equivalent modifications have to be made to `nn_gridsearch.py`. Running this script will create a weights and biases sweep, so it has to be configured to properly use weights and biases beforehand by setting the correct wandb project and API key. in order to do this, modify line 85:
```
        wandb_project_name = "my_wandb_project" # change this to your project name
```
to use your own wandb project, and line 107: 
```
    wandb.login(key = 'wanb_key') # change this to your wandb project key
```
to use the API key of your own respective project.

## Hyperparameter optimization
The gridsearches are preformed by running `nn_gridsearch.py`. This script creates a grid over the learning rate, batch size and l2 regularization parameters.

the minima and maxima of each parameter are defined by their respective command line argument, e.g. `--lrmin` defines the minimum learning rate used in the sweep. The grid uses linear intervals for batch size, and logarithmic intervals for the learning rate and l2 regularization penalty. This means that an input using `--lrmin -8 --lrmax -4` will create a grid with a minimum learning rate of 1 * 10^-8, and a maximum learning rate of 1 * 10^-4. 

The total size of the grid is defined by the `--counts` command line input. The script evenly divides this number over the three parameters, _rounded down_ . This means that the total grid size can be lower than the value given with the `--counts` argument. 

*example*:
```
    python nn_gridsearch.py --lrmin -8 --lrmax -4 --wdmin -5 --wdmax -3 --bsmin 32 --bsmax 2000 --counts 10000
```

creates a sweep with a learning rate between 1 * 10^-8 and 1 * 10^-4, l2 regularization between 1 * 10^-5 and 1 * 10^-3 and a batch size between 32 and 2000. for each parameter, there are `floor($\sqrt[3]{10000}$)` options that are tried.

### Other parameters
`epochs` defines the number of training epochs used. There are two command line inputs that define the dataset (input size) and training target; `--dset` can be 'allprot', 'cmb' or 'cmb_ffs'.'allprot' uses all 1428 proteins, 'cmb' only uses the cardiometabolic olink panel, and 'cmb_ffs' uses the proteins selected from this panel using forward feature selection. `--target` can be `mort` or `frailty` to train on mortality or frailty, respectively.

Finally, weights and biases has the option to run multiple _sweep agents_ in parallel. This allows speedup of the gridsearch, given enough computational resources. the command line argument `--sweep_pid` allows for this funtionality; First, start a sweep by keeping this value 0 (default). Then, once the sweep has started, find the sweep pid on the wandb sweeps page of your wandb project. This should note the sweep id of the specific sweep you are running.

when launching another sweep agent on the same sweep, simply run
```
    python nn_grisearch.py --sweep_pid wandb_sweep_pid
```
with the specific sweep pid of your sweep.

## Model training

To train a deep learning model with a given set of parameters, run `run_nn.py`. There are various command line arguments to set hyperparameters and run configurations, which are defined in `run_nn.py` itself (line 43-55). if ran without any parameters (i.e. `python run_nn.py`), default parameters will be used, which will result in training the reported model on the cardiometabolic proteins, predicting frailty.

### Bootstrap
In order to run the bootstrapping procedure as reported in figure 1 of the paper, use the `--bootstrap` argument in `run_.py`, i.e. `python run_nn.py --bootstrap 1` will run a bootstrap on the cardiometabolic proteins, predicting frailty. This will create one instance of the bootstrap, so this will have to be rerun for as many bootstraps as is desired (e.g. 100 times for what was reported in the paper). This is easiest done using a bash script such as `[nn_bootstrap.sh](batchfiles/nn_bootstrap.sh)`, though `sbatch` should be replaced by `python` when not using a SLURM scheduler.



# Forward Feature Selection (FFS)
The forward feature selection (FFS) procedure is implemented in `forward_selection.py`. By simply running `python forward_selection.py`, the forward feature selection procedure for protFI is replicated. for protMort, change the `--target` argument to 'mort', i.e. by running `python forward_selection.py --target mort`. Finally, the performance increase threshold can be changed by modifying the `--tol` flag. The proteins used for protFI are found [here](output_linear/coefs_frail/ffs_coefs_cmb_tol0.001.csv), and for protMort [here](output_linear/coefs_mort/ffs_coefs_cmb_tol0.001.csv).
