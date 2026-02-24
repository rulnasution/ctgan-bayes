import warnings
warnings.filterwarnings("ignore")

import math
import os
import numpy as np
import pandas as pd
import torch
import delu

from torch.optim import SGD, Adam
## if there are error of ctgan due to nonetype split, upgrade threadpoolctl library

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state

# for i in os.listdir('models/bnn_layers_based_model'):
#     if '.py' in i: exec(open('models/bnn_layers_based_model/'+i).read())

import matplotlib.pyplot as plt, seaborn as sns

### column names
import xgboost
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from pathlib import Path

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

init_dir = os.getcwd()

if not torch.cuda.is_available():
    device = 'cpu'
else:
    device = 'cuda'
device

data_dir = '/'.join(init_dir.split('/')[:-3])+'/tab-ddpm/'
print('device used: '+device)
print(init_dir.split('/'), data_dir,init_dir)
dfs = [] 
country_names = ['Canada','Fiji','UK','Rwanda','Indonesia','Adult','Churn','Insurance','Credit']
for i in range(9):
    dfs.append(pd.read_csv(data_dir+'census-data2/'+country_names[i]+'.csv',dtype=str))
    print(dfs[i].info())

cont_cols_canada = ['HRSWK','INCTOT','WKSWORK']
cat_cols_canada = [i for i in dfs[0].columns if i not in cont_cols_canada]
dfs[0][cont_cols_canada] = dfs[0][cont_cols_canada].astype(float)

iimp = IterativeImputer(
    estimator = xgboost.XGBRegressor(),
    random_state = 42,
    verbose = 2,
)

dfs[0][cont_cols_canada] = iimp.fit_transform(dfs[0][cont_cols_canada]) ## imputation to remove NA

cont_col_adult = ['age','fnlwgt','capital-gain','capital-loss','hours-per-week']
cont_col_churn = ['CreditScore', 'Age', 'Balance','EstimatedSalary']
cont_col_insurance = ['age','bmi','charges'] ## y = regression
cont_col_credit = ['months_loan_duration','amount','age']

## adult

cont_col_noncensus = [cont_col_adult, cont_col_churn, cont_col_insurance, cont_col_credit]
discrete_col_noncensus = [[j for j in dfs[i+5].columns if j not in cont_col_noncensus[i]] for i in range(4)]

discrete_columns = [cat_cols_canada,dfs[1].columns.tolist(),dfs[2].columns.tolist(),
                    dfs[3].columns.tolist(),dfs[4].columns.tolist()] + discrete_col_noncensus
cont_columns = [cont_cols_canada,None,None,None, None] + cont_col_noncensus

for i in range(5,9):
    dfs[i][cont_columns[i]] = dfs[i][cont_columns[i]].astype(float)
    
y_cols = ['TENURE','TENURE','TENURE','MARST','MARST',
          'income','Exited','charges','checking_balance']


dset_folder = [init_dir+"/data/census/"+i+'/'for i in ['canada', 'fiji', 'uk', 'rwanda', 'indonesia', 'adult', 'churn', 'insurance', 'credit']]
dset_folder[5:] = [i.replace('census','noncensus') for i in dset_folder[5:]]
dset_name = ['canada', 'fiji', 'uk', 'rwanda', 'indonesia', 'adult', 'churn', 'insurance', 'credit']
n_classes = [len(dfs[i][y_cols[i]].unique()) for i in range(9)]

init_dir2 = '/'.join(init_dir.split('/')[:-2])+'/ctgan-bbb/code/'
exec(open(init_dir+'/models.py').read())
exec(open(init_dir+'/CTGAN-swa.py').read())
exec(open(init_dir2+'/optimiser_mcmc.py').read())
exec(open(init_dir2+'/eval.py').read())

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swa_rank", type=int, default=100)
    args = parser.parse_args()

    final_res = pd.DataFrame()
    for i in range(9):
        for loss in ['wasserstein','vanilla']: # ,'wasserstein','ls'
            try:
                print(country_names[i],loss, args.seed, args.swa_rank)
                delu.random.seed(args.seed)
                ctgan = BayesCTGANSWA(epochs=500,loss_discriminator=loss,
                            verbose=False, pac=10, full_eval_every = 10,
                            bma_steps = [1, 2], cov_scales = [0, 0.25, 0.5, 1],
                            n_test_eval = None, country = country_names[i],
                            swa_start = 50, swa_lr = 2e-3, cyclic_lr = True, swa_collect = 1, 
                            swa_cov_mat = True, swag = True,
                            swa_max_models_save = args.swa_rank, save_results = True,
                            save_folder = f'{init_dir}/simulation/{args.swa_rank}-every10/{args.seed}')
                ctgan.fit(dfs[i], discrete_columns[i])

            #     for bma_step in [1,2,5,10]:
            #         for scale in [0.0, 0.5, 1.0]:
            #             for cov1 in [True, False]:
            #                 for block in [True, False]:
            #                     for bnu in [True, False]:
            #                         for s in range(5):
            #                             delu.random.seed(s)
            #                             dt1 = ctgan.sample(len(dfs[i]), bma_step, scale, cov1, block, bnu)
            #                             roc_val = cal_mean_roc(country_names[i],dfs[i],dt1)
            #                             cio_val = cal_mean_cio(country_names[i],dfs[i],dt1)
            #                             tcap_val = cal_mean_tcap(country_names[i],dfs[i],dt1)
            #                             final_res = pd.concat([final_res,
            #                                                     pd.DataFrame([[country_names[i],
            #                                     'GAN',loss,'SWAG',bma_step,scale,cov1,block,bnu,
            #                                     s,
            #                                     roc_val[0], roc_val[1], cio_val,
            #                                     (roc_val[0]+roc_val[1]+cio_val)/3,tcap_val]])],axis=0)
            #                     final_res.to_csv(f'results_gan_swag_30.csv', index=False)
            except Exception as e:
                print('GAN','SWAG',loss,e)
