import pdb
import pickle
import sys
import os
import os.path
import collections
import torch
import argparse
import pandas as pd
from tqdm import tqdm
import itertools
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sparse_gp import SparseGP
import scipy.stats as sps
import numpy as np
import scipy.io
from scipy.io import loadmat
from scipy.stats import pearsonr
import time
from shutil import copy
from copy import deepcopy
sys.path.append('%s/../software/enas' % os.path.dirname(os.path.realpath(__file__))) 
sys.path.append('%s/..' % os.path.dirname(os.path.realpath(__file__))) 
sys.path.insert(0, '../')
from utils import *
from layers.models_ig import CktGNN, DVAE
from layers.dagnn_pyg import DAGNN
from layers.constants import *

'''Experiment settings'''
parser = argparse.ArgumentParser(description='SGP regression on Ckt-Bench-101.')
# must specify
parser.add_argument('--data-fold-name', default='CktBench101', help='dataset fold name')
parser.add_argument('--data-name', default='ckt_bench_101', help='circuit benchmark dataset name')
parser.add_argument('--save-appendix', default='_cktgnn', help='identifuy the encoder')
parser.add_argument('--checkpoint', type=int, default=300, help="load which epoch's model checkpoint")
parser.add_argument('--res-dir', default='res/', 
                    help='where to save the Bayesian optimization results')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data instead of using prestored .pkl data')


# Model configuration
parser.add_argument('--model', default='CktGNN', help='model to use: DVAE, SVAE, \
                    DVAE_fast, DVAE_BN, SVAE_oneshot, DVAE_GCN')
parser.add_argument('--hs', type=int, default=301, metavar='N',
                    help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=66, metavar='N',
                    help='number of dimensions of latent vectors z')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='whether to use bidirectional encoding')
parser.add_argument('--nvt', type=int, default=26, help='number of different node (subgraph) types')
parser.add_argument('--max_n', type=int, default=8, help='number of different node (subgraph) types')
parser.add_argument('--subg_nvt', type=int, default=10, help='number of subgraph nodes')
parser.add_argument('--subn_nvt', type=int, default=10, help='number of subgraph feat')
parser.add_argument('--ng', type=int, default=10000, help='number of circuits in the dataset')
parser.add_argument('--node_feat_type', type=str, default='discrete', help='node feature type: discrete or continuous')
parser.add_argument('--emb_dim', type=int, default=24, metavar='N', help='embdedding dimension')
parser.add_argument('--feat_emb_dim', type=int, default=8, metavar='N', help='embdedding dimension')


parser.add_argument('--dagnn_layers', type=int, default=2)
parser.add_argument('--dagnn_agg', type=str, default=NA_ATTN_H)
parser.add_argument('--dagnn_out_wx', type=int, default=0, choices=[0, 1])
parser.add_argument('--dagnn_out_pool_all', type=int, default=0, choices=[0, 1])
parser.add_argument('--dagnn_out_pool', type=str, default=P_MAX, choices=[P_ATTN, P_MAX, P_MEAN, P_ADD])
parser.add_argument('--dagnn_dropout', type=float, default=0.0)


# device setting
parser.add_argument('--cuda_id', type=int, default=1, metavar='N',
                    help='id of GPU')
parser.add_argument('--infer-batch-size', type=int, default=128, metavar='N',
                    help='batch size during inference')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--all-gpus', action='store_true', default=False,
                    help='use all available GPUs')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if not args.cuda:
    device = torch.device("cpu")
else:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:{}".format(args.cuda_id))
np.random.seed(args.seed)
random.seed(args.seed)

# file dirs
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}{}'.format(args.data_name,args.save_appendix))
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 
test_results_name = os.path.join(args.res_dir, 'sgp_results.txt')


# loading data
args.data_dir = os.path.join(args.file_dir, 'OCB/{}'.format(args.data_fold_name))

data_name = args.data_name
if args.model.startswith('SVAE'):
    data_type = 'tensor'
    data_name += '_tensor'
elif args.model.startswith('DAGNN'):
    data_type = 'pygraph'
    data_name += '_pygraph'
else:
    data_type = 'igraph'

pkl_name = os.path.join(args.data_dir, data_name + '.pkl')
with open(pkl_name, 'rb') as f:
    all_datasets =  pickle.load(f)
train_dataset = all_datasets[0]
test_dataset = all_datasets[1]

# determine data formats according to models, DVAE: igraph, SVAE: string (as tensors)
if args.model.startswith('CktGNN'):
    train_data = [train_dataset[i][0] for i in range(len(train_dataset))]
    test_data = [test_dataset[i][0] for i in range(len(test_dataset))]
elif args.model.startswith('DAGNN'):
    train_data = [train_dataset[i] for i in range(len(train_dataset))]
    test_data = [test_dataset[i] for i in range(len(test_dataset))]
else:
    train_data = [train_dataset[i][1] for i in range(len(train_dataset))]
    test_data = [test_dataset[i][1] for i in range(len(test_dataset))]


if args.model.startswith('CktGNN'):
     nvt = 26
     START_TYPE = 0
     END_TYPE = 1
     max_n = 8
     max_pos = 8
     subn_nvt = 40
     subg = True
else:
     nvt = 10
     START_TYPE = 8
     END_TYPE = 9
     max_n = 8
     subn_nvt=103
     subg = False



def performance_readout(num_graphs, file_dir='circuit', name = 'ckt_simulation_summary_10000.txt'):
    num_graphs = 10000
    pbar = tqdm(range(num_graphs))
    gain = []
    bw = []
    pm = []
    fom = []
    valid = []
    #with open('ckt_simulation_summary_10000.txt', 'r') as f:
    file_name = os.path.join(file_dir, name)
    with open(file_name, 'r') as f:
        for i in pbar:
            row = f.readline().strip().split()
            if not row[1] == 'Simulation':
                g = float(row[1])/100.0
                p = float(row[2])/-90.0
                b = float(row[3])/1e9
                gain.append(g)
                pm.append(p)
                bw.append(b)
                fo = 1.2 * np.abs(g) + 1.6 * p + 10 * b
                fom.append(fo)
                valid.append(1)
            else:
                gain.append(0)
                pm.append(0)
                bw.append(0)
                fom.append(0)
                valid.append(0)
    gain = np.array(gain) - np.min(gain) + 0.00001
    pm = np.array(pm) - np.min(pm) + 0.00001
    perform = {'valid':valid, 'gain':gain, 'pm':pm, 'bw':bw, 'fom':fom}
    perform_df = pd.DataFrame(perform)
    out_name = os.path.join(file_dir, 'perform.csv')
    perform_df.to_csv(out_name)
    return perform_df


def extract_latent(data, perform_df, start_idx=0):
    model.eval()
    Z = []
    Y = []
    Gain = []
    BW = []
    PM = []
    g_batch = []
    for i, g  in enumerate(tqdm(data)):
        if args.model.startswith('SVAE'):
            g_ = g.to(device)
        elif args.model.startswith('DAGNN'):
            g_ = deepcopy(g)
        else:
            g_ = g.copy()  
        #if perform_df['valid'][start_idx + i] == 1: 
        g_batch.append(g_)
        if len(g_batch) == args.infer_batch_size or i == len(data) - 1:

            g_batch = model._collate_fn(g_batch)
            mu, _ = model.encode(g_batch)
            mu = mu.cpu().detach().numpy()
            Z.append(mu)
            g_batch = []
        #if perform_df['valid'][start_idx + i] == 1: 
        y = perform_df['fom'][start_idx + i]
        gain = perform_df['gain'][start_idx + i]
        bw = perform_df['bw'][start_idx + i]
        pm = perform_df['pm'][start_idx + i]
        Y.append(y)
        Gain.append(gain)
        BW.append(bw)
        PM.append(pm)
    return np.concatenate(Z, 0), np.array(Y), np.array(Gain), np.array(BW), np.array(PM)


'''Extract latent representations Z'''
def save_latent_representations(epoch, perform_df):
    Z_train, Y_train, Gain_train, BW_train, PM_train = extract_latent(train_data, perform_df, 0)
    Z_test, Y_test, Gain_test, BW_test, PM_test = extract_latent(test_data, perform_df, 9000)
    latent_pkl_name = os.path.join(args.res_dir, args.data_name +
                                   '_latent_epoch{}.pkl'.format(epoch))
    latent_mat_name = os.path.join(args.res_dir, args.data_name + 
                                   '_latent_epoch{}.mat'.format(epoch))
    with open(latent_pkl_name, 'wb') as f:
        pickle.dump((Z_train, Y_train, Z_test, Y_test), f)
    print('Saved latent representations to ' + latent_pkl_name)
    scipy.io.savemat(latent_mat_name, 
                     mdict={
                         'Z_train': Z_train, 
                         'Z_test': Z_test, 
                         'Y_train': Y_train, 
                         'Y_test': Y_test,
                         'Gain_train': Gain_train,
                         'Gain_test': Gain_test,
                         'BW_train':BW_train,
                         'BW_test':BW_test,
                         'PM_train':PM_train,
                         'PM_test':PM_test
                         }
                     )



# other BO hyperparameters
lr = 0.0005  # the learning rate to train the SGP model 0.0005
max_iter = 100  # how many iterations to optimize the SGP each time

#perform_df = performance_readout(args.ng, file_dir=args.data_dir)
perf_name = os.path.join(args.data_dir, 'perform101.csv')
perform_df = pd.read_csv(perf_name)

for rand_idx in range(1,10):
     print('START FOLD: {}'.format(rand_idx))

     save_dir = os.path.join(args.res_dir,'sgp_reg_{}_{}/'.format(save_appendix, rand_idx))
     # set seed
     random_seed = rand_idx
     torch.manual_seed(random_seed)
     torch.cuda.manual_seed(random_seed)
     np.random.seed(random_seed)

     if args.model.startswith('CktGNN'):
         model = CktGNN_sep(
            max_n = max_n, 
            max_pos = max_pos,
            nvt = nvt, 
            subn_nvt = subn_nvt,
            START_TYPE = START_TYPE, 
            END_TYPE = END_TYPE, 
            emb_dim = args.emb_dim, 
            feat_emb_dim = args.feat_emb_dim,
            hs=args.hs, 
            nz=args.nz,
            pos=True
            )
     elif args.model.startswith('DAGNN'):
         model = eval(args.model)(
            emb_dim = 10, 
            hidden_dim = args.hs, 
            out_dim = args.hs,
            max_n = max_n, 
            nvt = nvt, 
            START_TYPE = START_TYPE, 
            END_TYPE = END_TYPE,  
            hs=args.hs, 
            nz=args.nz,
            num_nodes=nvt+2,
            agg=args.dagnn_agg,
            num_layers=args.dagnn_layers, 
            bidirectional=args.bidirectional,
            out_wx=args.dagnn_out_wx > 0, 
            out_pool_all=args.dagnn_out_pool_all, 
            out_pool=args.dagnn_out_pool,
            dropout=args.dagnn_dropout
         )
     else:
         model = eval(args.model)(
         max_n = max_n, 
         nvt = nvt, 
         feat_nvt = subn_nvt, 
         START_TYPE = START_TYPE, 
         END_TYPE = END_TYPE,  
         hs=args.hs, 
         nz=args.nz
         )
    
     model.to(device)
     load_module_state(model, os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(checkpoint)), device=device)
     X_train, Y_train, Gain_train, BW_train, PM_train = extract_latent(train_data, perform_df, 0)
     X_test, Y_test, Gain_test, BW_test, PM_test = extract_latent(test_data, perform_df, 9000)

     y_train = -Y_train.reshape((-1,1))
     gain_train = -Gain_train.reshape((-1,1))
     bw_train = -BW_train.reshape((-1,1))
     pm_train = -PM_train.reshape((-1,1))
    

     mean_y_train, std_y_train = np.mean(y_train), np.std(y_train)
     mean_gain_train, std_gain_train = np.mean(gain_train), np.std(gain_train)
     mean_bw_train, std_bw_train = np.mean(bw_train), np.std(bw_train)
     mean_pm_train, std_pm_train = np.mean(pm_train), np.std(pm_train)

     #print('Mean, std of y_train is ', mean_y_train, std_y_train)
     y_train = (y_train - mean_y_train) / std_y_train
     gain_train = (gain_train - mean_gain_train) / std_gain_train
     bw_train = (bw_train - mean_bw_train) / std_bw_train
     pm_train = (pm_train - mean_pm_train) / std_pm_train


     y_test = -Y_test.reshape((-1,1))
     y_test = (y_test - mean_y_train) / std_y_train
     gain_test = -Gain_test.reshape((-1,1))
     gain_test = (gain_test - mean_gain_train) / std_gain_train
     bw_test = -BW_test.reshape((-1,1))
     bw_test = (bw_test - mean_bw_train) / std_bw_train
     pm_test = -PM_test.reshape((-1,1))
     pm_test = (pm_test - mean_pm_train) / std_pm_train
    

     '''SGP regression begins here'''
     print("Average pairwise distance between train points = {}".format(np.mean(pdist(X_train))))
     print("Average pairwise distance between test points = {}".format(np.mean(pdist(X_test))))


     M = 500
     ### Predicting FoM
     sgp_fom = SparseGP(X_train, 0 * X_train, y_train, M)
     sgp_fom.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0,  y_test, minibatch_size = 2 * M, max_iterations = max_iter, learning_rate = lr)
     pred_fom, uncert_fom = sgp_fom.predict(X_test, 0 * X_test)
     error_fom= np.sqrt(np.mean((pred_fom - y_test)**2))
     testll_fom = np.mean(sps.norm.logpdf(pred_fom - y_test, scale = np.sqrt(uncert_fom)))
     pearson_fom = float(pearsonr(pred_fom.reshape(-1,), y_test.reshape(-1,))[0])  
     print('Fom RMSE: ', error_fom)
     print('Fom Pearson r: ', pearson_fom)
     
     ### Predicting Gain
     sgp_gain = SparseGP(X_train, 0 * X_train, gain_train, M)
     sgp_gain.train_via_ADAM(X_train, 0 * X_train, gain_train, X_test, X_test * 0,  gain_test, minibatch_size = 2 * M, max_iterations = max_iter, learning_rate = lr)
     pred_gain, uncert_gain = sgp_gain.predict(X_test, 0 * X_test)
     error_gain= np.sqrt(np.mean((pred_gain - gain_test)**2))
     testll_gain = np.mean(sps.norm.logpdf(pred_gain - gain_test, scale = np.sqrt(uncert_gain)))
     pearson_gain = float(pearsonr(pred_gain.reshape(-1,), gain_test.reshape(-1,))[0])  
     print('Gain RMSE: ', error_gain)
     print('Gain Pearson r: ', pearson_gain)
    
     ### Predicting bw
     sgp_bw = SparseGP(X_train, 0 * X_train, bw_train, M)
     sgp_bw.train_via_ADAM(X_train, 0 * X_train, bw_train, X_test, X_test * 0,  bw_test, minibatch_size = 2 * M, max_iterations = max_iter, learning_rate = lr)
     pred_bw, uncert_bw = sgp_bw.predict(X_test, 0 * X_test)
     error_bw= np.sqrt(np.mean((pred_bw - bw_test)**2))
     testll_bw = np.mean(sps.norm.logpdf(pred_bw - bw_test, scale = np.sqrt(uncert_bw)))
     pearson_bw = float(pearsonr(pred_bw.reshape(-1,), bw_test.reshape(-1,))[0])  
     print('BW RMSE: ', error_bw)
     print('BW Pearson r: ', pearson_bw)
    
     ### Predicting pm
     sgp_pm = SparseGP(X_train, 0 * X_train, pm_train, M)
     sgp_pm.train_via_ADAM(X_train, 0 * X_train, pm_train, X_test, X_test * 0,  pm_test, minibatch_size = 2 * M, max_iterations = max_iter, learning_rate = lr)
     pred_pm, uncert_pm = sgp_pm.predict(X_test, 0 * X_test)
     error_pm= np.sqrt(np.mean((pred_pm - pm_test)**2))
     testll_pm = np.mean(sps.norm.logpdf(pred_pm - pm_test, scale = np.sqrt(uncert_pm)))
     pearson_pm = float(pearsonr(pred_pm.reshape(-1,), pm_test.reshape(-1,))[0])  
     print('PM RMSE: ', error_pm)
     print('PM Pearson r: ', pearson_pm)

     with open(test_results_name, 'a') as result_file:
         result_file.write(" Run: {} Fom rmse: {:.4f} Fom pearson: {:.4f} Gain rmse: {:.4f} Gain pearson: {:.4f} Bw rmse: {:.4f} Bw pearson: {:.4f} Pm rmse: {:.4f} Pm pearson: {:.4f} \n".format(rand_idx,
            error_fom, pearson_fom, error_gain, pearson_gain, error_bw, pearson_bw, error_pm, pearson_pm))
























