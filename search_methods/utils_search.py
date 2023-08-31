import os
import sys
sys.path.insert(0, os.getcwd())
from pybnn.bohamiann import Bohamiann
import argparse
import json
import torch
import scipy.stats as stats
import numpy as np
from collections import defaultdict


def load(path, device = None): 
    data = torch.load(path, map_location=device) 
    print('load pretrained embeddings from {}'.format(path))
    features = data['embeddings']
    valid_foms = data['foms']
    valid_foms = torch.Tensor(valid_foms)
    print('loading finished. pretrained embeddings shape {}'.format(features.shape))
    return features, valid_foms

def get_samples(features, valid_foms, visited, init_size):
    init_inds = np.random.permutation(list(range(features.shape[0])))[:init_size]
    ind_dedup = []
    for idx in init_inds:
        if idx not in visited:
            visited[idx] = True
            ind_dedup.append(idx)
    init_inds = torch.Tensor(ind_dedup).long()
    init_feat_samples = features[init_inds]
    init_valid_fom_samples = valid_foms[init_inds]
    return init_feat_samples, init_valid_fom_samples, visited

def propose_location(ei, features, valid_foms, visited, k):
    ei = ei.view(-1)
    print('remaining length of indices set:', len(features) - len(visited))
    indices = torch.argsort(ei)[-k:]
    ind_dedup = []
    for idx in indices:
        if idx not in visited:
            visited[idx] = True
            ind_dedup.append(idx)
    ind_dedup = torch.Tensor(ind_dedup).long()
    proposed_x, proposed_fom_valid= features[ind_dedup], valid_foms[ind_dedup]
    return proposed_x, proposed_fom_valid, visited

def step(query, features, valid_foms, visited):
    dist = torch.norm(features - query.view(1, -1), dim=1)
    knn = (-1 * dist).topk(dist.shape[0])
    min_dist, min_idx = knn.values, knn.indices
    i = 0
    while True:
        if len(visited) == dist.shape[0]:
            print("cannot find in the dataset")
            exit()
        if min_idx[i].item() not in visited:
            visited[min_idx[i].item()] = True
            break
        i += 1

    return features[min_idx[i].item()], valid_foms[min_idx[i].item()], visited





