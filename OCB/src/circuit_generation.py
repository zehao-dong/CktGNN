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
import networkx as nx
from random import shuffle
from utils_src import *
from amp_generator import *

parser = argparse.ArgumentParser(description='Train Variational Autoencoders for Circuits')
parser.add_argument('--data-name', default='circuit101', help='graph dataset name')
parser.add_argument('--ng', type=int, default=10000, help='number of circuits in the dataset')
parser.add_argument('--node_feat', type=str, default='discrete', help='node feature type: discrete or continuous')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
np.random.seed(args.seed)
random.seed(args.seed)
print(args)


NODE_TYPE = {
    'R': 0,
    'C': 1,
    '+gm+':2,
    '-gm+':3,
    '+gm-':4,
    '-gm-':5,
    'sudo_in':6,
    'sudo_out':7,
    'In': 8,
    'Out':9
}

SUBG_NODE = {
    0: ['In'],
    1: ['Out'],
    2: ['R'],
    3: ['C'],
    4: ['R','C'],
    5: ['R','C'],
    6: ['+gm+'],
    7: ['-gm+'],
    8: ['+gm-'],
    9: ['-gm-'],
    10: ['C', '+gm+'],
    11: ['C', '-gm+'],
    12: ['C', '+gm-'],
    13: ['C', '-gm-'],
    14: ['R', '+gm+'],
    15: ['R', '-gm+'],
    16: ['R', '+gm-'],
    17: ['R', '-gm-'],
    18: ['C', 'R', '+gm+'],
    19: ['C', 'R', '-gm+'],
    20: ['C', 'R', '+gm-'],
    21: ['C', 'R', '-gm-'],
    22: ['C', 'R', '+gm+'],
    23: ['C', 'R', '-gm+'],
    24: ['C', 'R', '+gm-'],
    25: ['C', 'R', '-gm-']
}

SUBG_CON = {
    0: None,
    1: None,
    2: None,
    3: None,
    4: 'series',
    5: 'parral',
    6: None,
    7: None,
    8: None,
    9: None,
    10: 'parral',
    11: 'parral',
    12: 'parral',
    13: 'parral',
    14: 'parral',
    15: 'parral',
    16: 'parral',
    17: 'parral',
    18: 'parral',
    19: 'parral',
    20: 'parral',
    21: 'parral',
    22: 'series',
    23: 'series',
    24: 'series',
    25: 'series'
}

SUBG_INDI = {0: [],
 1: [],
 2: [0],
 3: [1],
 4: [0, 1],
 5: [0, 1],
 6: [2],
 7: [2],
 8: [2],
 9: [2],
 10: [1, 2],
 11: [1, 2],
 12: [1, 2],
 13: [1, 2],
 14: [0, 2],
 15: [0, 2],
 16: [0, 2],
 17: [0, 2],
 18: [1, 0, 2],
 19: [1, 0, 2],
 20: [1, 0, 2],
 21: [1, 0, 2],
 22: [1, 0, 2],
 23: [1, 0, 2],
 24: [1, 0, 2],
 25: [1, 0, 2]
 }


args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.data_dir = os.path.join(args.file_dir, args.data_name)
if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir) 

txt_name1= os.path.join(args.data_dir, args.data_name + '.txt')
txt_name2= os.path.join(args.data_dir, args.data_name + '_conti.txt')

if args.node_feat == 'discrete':
    if os.path.isfile(txt_name1):
        pass
    else:
        circuit_generation_dis(args.ng, SUBG_NODE, SUBG_CON, NODE_TYPE, start_type=2, end_type=26, dataset=txt_name1)
else:
    if os.path.isfile(txt_name2):
        pass
    else:
       circuit_generation(args.ng, SUBG_NODE, SUBG_CON, NODE_TYPE, start_type=2, end_type=26, dataset=txt_name2)








