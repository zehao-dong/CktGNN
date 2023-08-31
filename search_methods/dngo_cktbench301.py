import os
import sys
sys.path.insert(0, os.getcwd())
from pybnn.dngo import DNGO
import argparse
import json
import torch
import scipy.stats as stats
import numpy as np
from collections import defaultdict
from search_methods.utils_search import *

def dngo_expected_improvement_search(embedding_path, total_rounds, save_path, init_size, k, seed):
    """ implementation of CATE-DNGO-LS on the NAS-Bench-101 search space """
    BEST_FOM = 197.22961595458466
    PREV_BEST = 0
    CURR_BEST_FOM = 0.
    MAX_BUDGET = 150
    window_size = 512
    counter = 0
    round = 0
    rt = 0.
    visited = {}
    best_trace = defaultdict(list)
    features, valid_foms = load(embedding_path)
    feat_samples, valid_fom_samples, visited = get_samples(features, valid_foms, visited, init_size)

    for feat, fom in zip(feat_samples, valid_fom_samples):
        counter += 1
        if fom  > CURR_BEST_FOM:
            CURR_BEST_FOM = fom

        best_trace['regret_fom'].append(float(BEST_FOM - CURR_BEST_FOM))
        best_trace['counter'].append(counter)

    while counter <= MAX_BUDGET:
        if round == total_rounds:
            feat_samples, valid_fom_samples, visited = get_samples(features, valid_foms, visited, init_size)
            for feat, fom in zip(feat_samples, valid_fom_samples):
                counter += 1
                if fom  > CURR_BEST_FOM:
                    CURR_BEST_FOM = fom
                best_trace['regret_fom'].append(float(BEST_FOM - CURR_BEST_FOM))
                best_trace['counter'].append(counter)
            round = 0

        print("current best fom: {}".format(CURR_BEST_FOM))
        print("counter: {}".format(counter))
        print(feat_samples.shape)
        print(valid_fom_samples.shape)
        print('begin training BO model')
        model = DNGO(num_epochs=args.epochs, n_units_1=128, n_units_2=128, n_units_3=128, do_mcmc=False, normalize_output=False)
        model.train(X=feat_samples.numpy(), y=valid_fom_samples.view(-1).numpy(), do_optimize=True)
        print('BO model training finished')
        #print(model.network)
        m = []
        v = []
        chunks = int(features.shape[0] / window_size)
        if features.shape[0] % window_size > 0:
            chunks += 1
        features_split = torch.split(features, window_size, dim=0)
        for i in range(chunks):
            if i % 100 == 0:
                print('processing chunk: {}'.format(i))
            m_split, v_split = model.predict(features_split[i].numpy())
            m.extend(list(m_split))
            v.extend(list(v_split))

        mean = torch.Tensor(m)
        sigma = torch.Tensor(v)
        u = (mean - torch.Tensor([1.0]).expand_as(mean)) / sigma
        ei = sigma * (u * stats.norm.cdf(u) + 1 + stats.norm.pdf(u))
        print('begin determining next position')
        feat_next, fom_next_valid, visited = propose_location(ei, features, valid_foms, visited, k)
        print('next positions detected')
        # add proposed networks to the pool
        for feat, fom in zip(feat_next, fom_next_valid):
            if fom  > CURR_BEST_FOM:
                CURR_BEST_FOM = fom
            feat_samples = torch.cat((feat_samples, feat.view(1, -1)), dim=0)
            valid_fom_samples = torch.cat((valid_fom_samples.view(-1, 1), fom.view(1, 1)), dim=0)
            counter += 1
            best_trace['regret_fom'].append(float(BEST_FOM - CURR_BEST_FOM))
            best_trace['counter'].append(counter)
            if counter >= MAX_BUDGET:
                break

        if PREV_BEST < CURR_BEST_FOM:
            PREV_BEST = CURR_BEST_FOM
        else:
            round += 1

    res = dict()
    res['regret_fom'] = best_trace['regret_fom']
    res['counter'] = best_trace['counter']
    res['detected_best_fom'] = CURR_BEST_FOM
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    print('save to {}'.format(save_path))
    
    pkl_name_ = os.path.join(save_path, 'run_{}.pkl'.format(seed))
    with open(pkl_name_, 'wb') as f:
        pickle.dump(res, f)
