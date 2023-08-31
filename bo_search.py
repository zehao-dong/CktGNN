import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import json
import torch
import scipy.stats as stats
import numpy as np
from collections import defaultdict
from search_methods.utils_search import *
from search_methods.bo_cktbench301 import bo_expected_improvement_search
from search_methods.dngo_cktbench301 import dngo_expected_improvement_search
import pickle


parser = argparse.ArgumentParser(description="BO Search on Ckt-Bench-301")
parser.add_argument("--runs", type=int, default=20, help="number of runs")
parser.add_argument('--dim', type=int, default=66, help='feature dimension')
parser.add_argument('--epochs', type=int, default=30, help='outer loop epochs')
parser.add_argument('--init_size', type=int, default=10, help='init samples')
parser.add_argument('--topk', type=int, default=5, help='acquisition samples')
parser.add_argument('--rounds', type=int, default=20, help='rounds allowed for local minimum')
parser.add_argument('--search-method', type=str, default='bo', help='bo or dngo')
parser.add_argument('--output_path', type=str, default='bo_search', help='store BO results')

parser.add_argument('--data-name', default='ckt_bench_301', help='graph dataset name')
parser.add_argument('--save-appendix', default='', 
                    help='what is appended to data-name as save-name for results')
parser.add_argument('--model', default='CktGNN', help='model to use')

args = parser.parse_args()
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}{}'.format(args.data_name,args.save_appendix))
embed_name = args.model + '_embeddings' + '.pt'
embedding_path = os.path.join(args.res_dir, embed_name)
save_path = os.path.join(args.res_dir, args.output_path)

if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 
if not os.path.exists(save_path):
    os.makedirs(save_path)


for run in range(args.runs):
    if args.search_method == 'bo':
        bo_expected_improvement_search(embedding_path, args.rounds, save_path, args.init_size, args.topk, run)
    elif args.search_method == 'dngo':
        dngo_expected_improvement_search(embedding_path, args.rounds, save_path, args.init_size, args.topk, run)








