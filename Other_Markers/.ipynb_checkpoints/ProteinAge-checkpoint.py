#!/usr/bin/env python
# coding: utf-8

#from proteinage.utils import *
#from proteinage.predict import ProtAge_predict
#import random
import miceforest as mf
import pandas as pd
import datetime as dt

random_seed = 3456

# Load protein data
data_path = '../Data/Olink_UKBB_packageDataFreeze_19Jan2024/67864_olink_0.csv'
data = pd.read_csv(data_path)
data = data[data.columns[data.columns != 'X']]

# Create a dictionary of variables use to impute
exclude = ['eid', 'olink_batch', 'olink_plate', 'glipr1', 'npm1', 'pcolce']
dont_impute = ['eid', 'olink_batch', 'olink_plate']
column_dict = {col: [other_col for other_col in data.columns if other_col != col and other_col not in exclude] for col in data.columns if col not in dont_impute}

# run miceforest imputation on multiple cores
kds = mf.ImputationKernel(
  data,
  num_datasets=1,
  variable_schema=column_dict,
  random_state=random_seed
)

# run
kds.mice(
  iterations=5,
  n_jobs=-1, 
  verbose=True
)

# get the completed dataframe from the miceforest object
olink_data_imputed = kds.complete_data()

name = "../Data/Other_Biomarkers/OlinkImputedProteinAge.csv"
olink_data_imputed.to_csv(name, index=False)

# calculate ProtAge
#model_file = 'ProteinAge_Model_Files/ProtAge_model.p'
#predictions = ProtAge_predict(data, model='ProtAge', model_file=model_file)

# calculate ProtAge20
#model_file20 = 'ProteinAge_Model_Files/ProtAge20_model.p'
#predictions = ProtAge_predict(data, model='ProtAge20', model_file=model_file)

# calculate ProtAge and normalize proteomic data within-sample as we use UK Biobank data
#predictions = ProtAge_predict(
#    olink_data_imputed, 
#    model='ProtAge', 
#    model_path=model_file,
#    normalize=True,
#    normalize_ref='None' 
#)

# calculate ProtAge20 and normalize proteomic data within-sample as we use UK Biobank data
#predictions20 = ProtAge_predict(
#    olink_data_imputed, 
#    model='ProtAge20', 
#    model_path=model_file20,
#    normalize=True,
#    normalize_ref='None' 
#)

#proteinage = pd.DataFrame({'ProteinAge':predictions, 'ProteinAge20': predictions20}, index = data.index)
#proteinage.to_csv('../Data/Other_Biomarkers/ProteinAge.csv')
