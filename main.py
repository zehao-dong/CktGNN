from __future__ import print_function
import os
import sys
import math
import pickle
import pdb
import argparse
import random
from tqdm import tqdm
from shutil import copy
import numpy as np
import scipy.io
from scipy.linalg import qr 
import igraph
from random import shuffle
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import *
from layers.models_ig import CktGNN, DVAE
from layers.dagnn_pyg import DAGNN
from layers.constants import *
import time

parser = argparse.ArgumentParser(description='VAE experiments on Ckt-Bench-101')
# general settings
parser.add_argument('--data-fold-name', default='CktBench101', help='dataset fold name')
parser.add_argument('--data-name', default='ckt_bench_101', help='circuit benchmark dataset name')
parser.add_argument('--nvt', type=int, default=26, help='number of different node (subgraph) types')
parser.add_argument('--subg_nvt', type=int, default=10, help='number of subgraph types')
parser.add_argument('--subn_nvt', type=int, default=103, help='number of subgraph feats if discrete')
parser.add_argument('--ng', type=int, default=10000, help='number of circuits in the dataset')
parser.add_argument('--node_feat_type', type=str, default='discrete', help='subg feature type: discrete or continuous')

parser.add_argument('--save-appendix', default='_cktgnn', help='identifuy the encoder')
parser.add_argument('--save-interval', type=int, default=100, metavar='N',
                    help='how many epochs to wait each time to save model states')
parser.add_argument('--sample-number', type=int, default=20, metavar='N',
                    help='how many samples to generate each time')
parser.add_argument('--reprocess', action='store_true', default=False,
                    help='if True, reprocess data instead of using prestored .pkl data')
parser.add_argument('--no-test', action='store_true', default=False,
                    help='if True, merge test with train, i.e., no held-out set')

# model settings
parser.add_argument('--model', default='CktGNN', help='model to use: CKTGNN, PACE, DAGNN, DVAE...')
parser.add_argument('--load-latest-model', action='store_true', default=False,
                    help='whether to load latest_model.pth')
parser.add_argument('--continue-from', type=int, default=None, 
                    help="from which epoch's checkpoint to continue training")
parser.add_argument('--emb_dim', type=int, default=24, metavar='N', help='embdedding dimension')
parser.add_argument('--feat_emb_dim', type=int, default=8, metavar='N', help='embedding dimension of subg feats')
parser.add_argument('--hs', type=int, default=301, metavar='N',help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=66, metavar='N',help='embedding dimension of latent space')
parser.add_argument('--bidirectional', action='store_true', default=False,help='whether to use bidirectional encoding')

#dagnn specific 
parser.add_argument('--dagnn_layers', type=int, default=2)
parser.add_argument('--dagnn_agg', type=str, default=NA_ATTN_H)
parser.add_argument('--dagnn_out_wx', type=int, default=0, choices=[0, 1])
parser.add_argument('--dagnn_out_pool_all', type=int, default=0, choices=[0, 1])
parser.add_argument('--dagnn_out_pool', type=str, default=P_MAX, choices=[P_ATTN, P_MAX, P_MEAN, P_ADD])
parser.add_argument('--dagnn_dropout', type=float, default=0.0)

# training  settings
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='batch size during training')
parser.add_argument('--cuda_id', type=int, default=0, metavar='N',
                    help='id of GPU')
parser.add_argument('--infer-batch-size', type=int, default=128, metavar='N',
                    help='batch size during inference')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--all-gpus', action='store_true', default=False,
                    help='use all available GPUs')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--predictor', action='store_true', default=False, help='whether to train a performance predictor from latent encodings and a VAE at the same time')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:{}".format(args.cuda_id))
else:
    device = torch.device("cpu")
np.random.seed(args.seed)
random.seed(args.seed)
print(args)

args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}{}'.format(args.data_name,args.save_appendix))
args.data_dir = os.path.join(args.file_dir, 'OCB/{}'.format(args.data_fold_name))

if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)  

# Loading datasets:
# 1. igraph dataset:  dataset[0] = train set, dataset[1] = test set, each item is a pair (DAG of subgraphs for CktGNN, original igraph DAG)
# 2. pygraph datasets: dataset[0] = train set, dataset[1] = test set, each item is a pygraph Data
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

if args.model.startswith('CktGNN'):
    train_data = [train_dataset[i][0] for i in range(len(train_dataset))]
    test_data = [test_dataset[i][0] for i in range(len(test_dataset))]
elif args.model.startswith('DAGNN'):
    train_data = [train_dataset[i] for i in range(len(train_dataset))]
    test_data = [test_dataset[i] for i in range(len(test_dataset))]
else:
    train_data = [train_dataset[i][1] for i in range(len(train_dataset))]
    test_data = [test_dataset[i][1] for i in range(len(test_dataset))]
    
# delete old files in the result directory
remove_list = [f for f in os.listdir(args.res_dir) if not f.endswith(".pkl") and 
        not f.startswith('train_graph') and not f.startswith('test_graph') and
        not f.endswith('.pth')]

for f in remove_list:
    tmp = os.path.join(args.res_dir, f)
    if not os.path.isdir(tmp) and not args.keep_old:
        os.remove(tmp)

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')


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

# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

model.to(device)

if args.load_latest_model:
    load_module_state(model, os.path.join(args.res_dir, 'latest_model.pth'))
else:
    if args.continue_from is not None:
        epoch = args.continue_from
        load_module_state(model, os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch)), device)
        load_module_state(optimizer, os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch)), device)
        load_module_state(scheduler, os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch)), device)

# training function
def train(epoch):
    model.train()
    train_loss = 0
    recon_loss = 0
    kld_loss = 0
    shuffle(train_data)
    pbar = tqdm(train_data)
    g_batch = []
    for i, g in enumerate(pbar):
        if args.model.startswith('SVAE'):  # for SVAE, g is tensor
            g = g.to(device)
        g_batch.append(g)
        #y_batch.append(y)
        if len(g_batch) == args.batch_size or i == len(train_data) - 1:
            optimizer.zero_grad()
            g_batch = model._collate_fn(g_batch)
            if args.all_gpus:  # does not support predictor yet
                loss = net(g_batch).sum()
                pbar.set_description('Epoch: %d, loss: %0.4f' % (epoch, loss.item()/len(g_batch)))
                recon, kld = 0, 0
            else:
                mu, logvar = model.encode(g_batch)
                loss, recon, kld, type_l, pos_l, df_l= model.loss(mu, logvar, g_batch)
                pbar.set_description('Epoch: %d, loss: %0.4f, recon: %0.4f, kld: %0.4f, type loss: %0.4f, pos loss: %0.4f,all df_loss: %0.4f' % (
                                epoch, loss.item()/len(g_batch), recon.item()/len(g_batch), 
                                kld.item()/len(g_batch), -type_l.item()/len(g_batch), -pos_l.item()/len(g_batch), 
                                df_l.item()/len(g_batch)))
            loss.backward()
            
            train_loss += float(loss)
            recon_loss += float(recon)
            kld_loss += float(kld)
            type_loss -= float(type_l)
            pos_loss -= float(pos_l)
            if args.predictor:
                pred_loss += float(pred)
            optimizer.step()
            g_batch = []
            y_batch = []
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_data)))
    return train_loss, recon_loss, kld_loss, type_loss, pos_loss

def test():
    # test recon accuracy
    model.eval()
    encode_times = 10
    decode_times = 10
    Nll = 0
    pred_loss = 0
    n_perfect = 0
    print('Testing begins...')
    pbar = tqdm(test_data)
    g_batch = []
    #y_batch = []
    #time_start=time.time()
    for i, g in enumerate(pbar):
        if args.model.startswith('SVAE'):
            g = g.to(device)
        g_batch.append(g)
        #y_batch.append(y)
        if len(g_batch) == args.infer_batch_size or i == len(test_data) - 1:
            g = model._collate_fn(g_batch)
            mu, logvar = model.encode(g)
            _, nll, _, _, _, _ = model.loss(mu, logvar, g)
            pbar.set_description('nll: {:.4f}'.format(nll.item()/len(g_batch)))
            Nll += nll.item()
            #if args.model.startswith('SVAE'):  
            #    g = model.construct_igraph(g[:, :, :model.nvt], g[:, :, model.nvt:], False)
            for _ in range(encode_times):
                z = model.reparameterize(mu, logvar)
                for _ in range(decode_times):
                    g_recon = model.decode(z)
                    n_perfect += sum(is_same_DAG(g0, g1) for g0, g1 in zip(g, g_recon))
            g_batch = []
            
    Nll /= len(test_data)
    pred_loss /= len(test_data)
    pred_rmse = math.sqrt(pred_loss)
    acc = n_perfect / (len(test_data) * encode_times * decode_times)
    print('Test average recon loss: {0}, recon accuracy: {1:.4f}'.format(Nll, acc))
    return Nll, acc



'''Training begins here'''
min_loss = math.inf  # >= python 3.5
min_loss_epoch = None
loss_name = os.path.join(args.res_dir, 'train_loss.txt')
loss_plot_name = os.path.join(args.res_dir, 'train_loss_plot.pdf')
test_results_name = os.path.join(args.res_dir, 'test_results.txt')
if os.path.exists(loss_name) and not args.keep_old:
    os.remove(loss_name)

start_epoch = args.continue_from if args.continue_from is not None else 0
for epoch in range(start_epoch + 1, args.epochs + 1):
    train_loss, recon_loss, kld_loss, type_loss, pos_loss= train(epoch)
    pred_loss = 0.0
    with open(loss_name, 'a') as loss_file:
        loss_file.write("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} \n".format(
            train_loss/len(train_data), 
            recon_loss/len(train_data), 
            kld_loss/len(train_data), 
            type_loss/len(train_data), 
            pos_loss/len(train_data)
            ))
    scheduler.step(train_loss)
    if epoch % args.save_interval == 0:
        print("save current model...")
        model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
        optimizer_name = os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
        scheduler_name = os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
        torch.save(model.state_dict(), model_name)
        torch.save(optimizer.state_dict(), optimizer_name)
        torch.save(scheduler.state_dict(), scheduler_name)
        

'''Testing begins here'''
Nll, acc = test()
r_valid_dag, r_valid_ckt, r_novel = prior_validity(train_data, model, infer_batch_size=self.infer_batch_size, 
    data_type=data_type,  subg=subg, device=device, scale_to_train_range=True)

test_results_name = os.path.join(args.res_dir, 'decode_results.txt')
with open(test_results_name, 'a') as result_file:
    result_file.write(" recon acc: {:.4f} r_valid_dag: {:.4f} r_valid_ckt: {:.4f} r_novel: {:.4f}\n".format(acc, r_valid_dag, r_valid_ckt,
            r_novel))

pdb.set_trace()



