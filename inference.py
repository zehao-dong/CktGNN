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
import scipy.stats as sps
import numpy as np
import scipy.io
from scipy.io import loadmat
from scipy.stats import pearsonr
sys.path.append('%s/../software/enas' % os.path.dirname(os.path.realpath(__file__))) 
sys.path.append('%s/..' % os.path.dirname(os.path.realpath(__file__))) 
sys.path.insert(0, '../')
from utils import *
from layers.models_ig import CktGNN, DVAE
from layers.dagnn_pyg import DAGNN
from layers.constants import *
from copy import deepcopy

'''Experiment settings'''
parser = argparse.ArgumentParser(description='Generating embeddings in the latent space for Bayesian optimization on dataset Ckt-Bench-301.')
# must specify
parser.add_argument('--data_fold_name', default='CktBench301', help='graph dataset name')
parser.add_argument('--data-name', default='ckt_bench_301', help='circuit benchmark dataset name')
parser.add_argument('--load-model-name', default='ckt_bench_101', help='graph dataset name')
parser.add_argument('--save-appendix', default='', 
                    help='what is appended to data-name as save-name for results')
parser.add_argument('--checkpoint', type=int, default=300, 
                    help="load which epoch's model checkpoint")
parser.add_argument('--res-dir', default='res/', 
                    help='where to save the Bayesian optimization results')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data instead of using prestored .pkl data')


# can be inferred from the cmd_input.txt file, no need to specify
parser.add_argument('--data-type', default='ipython',
                    help='ipython, tensor, pygraph')
parser.add_argument('--model', default='CktGNN', help='model to use: CktGNN, PACE, DAGNN, DVAE, SVAE, DVAE_GCN')
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
parser.add_argument('--ng', type=int, default=80000, help='number of circuits in the dataset')
parser.add_argument('--node_feat_type', type=str, default='discrete', help='node feature type: discrete or continuous')
parser.add_argument('--emb_dim', type=int, default=24, metavar='N', help='embdedding dimension')
parser.add_argument('--feat_emb_dim', type=int, default=8, metavar='N', help='embdedding dimension')

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
args.model_dir = os.path.join(args.file_dir, 'results/{}{}'.format(args.load_model_name,args.save_appendix))
args.data_dir = os.path.join(args.file_dir, 'OCB/{}'.format(args.data_fold_name))
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 

if args.model.startswith('CktGNN'):
     nvt = 26
     START_TYPE = 0
     END_TYPE = 1
else:
     nvt = 10
     START_TYPE = 8
     END_TYPE = 9

data_name = args.data_name
if args.model.startswith('SVAE'):
    data_type = 'tensor'
    data_name += '_tensor'
elif args.model.startswith('DAGNN'):
    data_type = 'pygraph'
    data_name += '_pygraph'
else:
    data_type = 'igraph'

# cLoad dataset
pkl_name = os.path.join(args.data_dir, data_name + '.pkl')
with open(pkl_name, 'rb') as f:
    all_dataset =  pickle.load(f)

if args.model.startswith('CktGNN'):
    data = [all_dataset[i][0] for i in range(len(all_dataset))]
elif args.model.startswith('DAGNN'):
    data = [all_dataset[i] for i in range(len(all_dataset))]
else:
    data = [all_dataset[i][1] for i in range(len(all_dataset))]


# model construction

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

if args.model.startswith('CktGNN'):
    model = CktGNN(
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
    model = DAGNN(
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


load_module_state(model, os.path.join(args.model_dir, 'model_checkpoint{}.pth'.format(args.checkpoint)), device)

# load performance of circuits
perf_name = os.path.join(args.data_dir, 'perform301.csv')
perform_df = pd.read_csv(perf_name)


def inference(data, perform_df):
    model.eval()
    embeddings = []
    foms = []
    
    g_batch = []
    for i, g  in enumerate(tqdm(data)):
        if data_type== 'tensor':
            g_ = g.to(device)
        elif data_type== 'pygraph':
            g_ = deepcopy(g)
        else:
            g_ = g.copy() 
 
        g_batch.append(g_)
        if len(g_batch) == args.infer_batch_size or i == len(data) - 1:
            g_batch = model._collate_fn(g_batch)
            mu, _ = model.encode(g_batch)
            mu = mu.cpu().detach()
            embeddings.append(mu)
            g_batch = []
        #if perform_df['valid'][start_idx + i] == 1: 
        y = list(perform_df['fom'])[i]
        foms.append(y)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings, np.array(foms)

embeddings, foms = inference(data, perform_df)
pretrained_embeddings = {'embeddings': embeddings, 'foms': foms}
embed_name = args.model + '_embeddings' + '.pt'
save_dir = os.path.join(args.res_dir, embed_name)
torch.save(pretrained_embeddings, save_dir)






