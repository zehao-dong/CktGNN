import pandas as pd
import numpy as np
import tqdm
import csv
import torch
from tqdm import tqdm
import igraph
import networkx as nx
import random
import pickle
import gzip
from shutil import copy
from copy import deepcopy

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

def load_module_state(model, state_name, device):
    pretrained_dict0 = torch.load(state_name, map_location=device)   #, map_location=torch.device('cpu')
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict0.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return

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
    out_name = os.path.join(file_dir, perform.csv)
    perform_df.to_csv(out_name)
    return perform_df


class MyException(Exception):
    def __init__(self, msg):
        self.msg = msg

def is_same_DAG(g0, g1):
    # note that it does not check isomorphism
    if g0.vcount() != g1.vcount():
        return False
    for vi in range(g0.vcount()):
        if g0.vs[vi]['type'] != g1.vs[vi]['type']:
            return False
        if set(g0.neighbors(vi, 'in')) != set(g1.neighbors(vi, 'in')):
            return False
    return True

def pygraph_to_igraph(pygraph):
    n_v = pygraph.x.shape[0]
    attr_v = pygraph.x
    edge_idxs = pygraph.edge_index
    g = igraph.Graph(directed=True)
    g.add_vertices(n_v)
    for i in range(n_v):
        g.vs[i]['type'] = torch.argmax(attr_v[i,:-1]).item()
        g.vs[i]['feat'] = attr_v[i,-1].item()
    edges = []
    for src, tgt in zip(edge_idxs[1], edge_idxs[0]):
        edges += [(src, tgt)]
    g.add_edges(edges)
    return g

def is_same_igraph_pygraph(g0, g1):
    g0_ = pygraph_to_igraph(g0)
    return is_same_DAG(g0_, g1)

def ratio_same_DAG(G0, G1):
    # how many G1 are in G0
    res = 0
    for g1 in tqdm(G1):
        for g0 in G0:
            if is_same_DAG(g1, g0):
                res += 1
                break
    return res / len(G1)


def is_valid_DAG(g, subg=True):
    # Check if the given igraph g is a valid DAG computation graph
    # first need to have no directed cycles
    # second need to have no zero-indegree nodes except input
    # third need to have no zero-outdegree nodes except output
    # i.e., ensure nodes are connected
    # fourth need to have exactly one input node
    # finally need to have exactly one output node
    if subg:
        START_TYPE=0
        END_TYPE=1
    else:
        START_TYPE=8 
        END_TYPE=9
    res = g.is_dag()
    #return res
    n_start, n_end = 0, 0
    for v in g.vs:
        if v['type'] == START_TYPE:
            n_start += 1
        elif v['type'] == END_TYPE:
            n_end += 1
        if v.outdegree() == 0 and v['type'] != END_TYPE:
            return False
    return res and n_start == 1 and n_end == 1

def is_valid_Circuit(g, subg=True):
    # Check if the given igraph g is a amp circuits
    # first checks whether the circuit topology is a DAG
    # second checks the node type in the main path
    if subg:
        cond1 = is_valid_DAG(g, subg=True)
        cond2 = True
        for v in g.vs:
            pos = v['pos']
            subg_feats = [v['r'], v['c'], v['gm']]
            if pos in [2,3,4]: # i.e. in the main path
                if v['type'] in [8,9]:
                    cond2 = False
        return cond1 and cond2
    else:
        cond1 = is_valid_DAG(g, subg=False)
        cond2 = True
        diameter_path = g.get_diameter(directed=True) #find the main path the diameter path must start/end at the sudo input/end node
        if len(diameter_path) < 3:
            cond2 = False
        for i, v_ in enumerate(diameter_path):
            v = g.vs[v_]
            if i == 0:
                if v['type'] != 8:
                    cond2 = False
            elif i == len(diameter_path) - 1:
                if v['type'] != 9:
                    cond2 = False
            else:
                #if v['type'] not in [1,2,3]: # main path nodes must come from subg_type = 6 or 7 or 10 or 11
                if v['type'] in [4, 5]:
                    cond2 = False
                    predecessors_ = g.predecessors(i)
                    successors_ = g.successors(i)
                    for v_p in predecessors_:
                        v_p_succ = g.successors(v_p)
                        for v_cand in v_p_succ:
                            inster_set = set(g.successors(v_cand)) & set(successors_)
                            if g.vs[v_cand]['type'] in [0,1] and len(inster_set) > 0:
                                cond2 = True
        return cond1 and cond2

def extract_latent_z(data, model, data_type='igraph', start_idx=0, infer_batch_size=64):
    model.eval()
    Z = []
    g_batch = []
    for i, g  in enumerate(tqdm(data)):
        if data_type== 'tensor':
            g_ = g.to(device)
        elif data_type== 'pygraph':
            g_ = deepcopy(g)
        else:
            g_ = g.copy()  
        g_batch.append(g_)
        if len(g_batch) == infer_batch_size or i == len(data) - 1:

            g_batch = model._collate_fn(g_batch)
            mu, _ = model.encode(g_batch)
            mu = mu.cpu().detach().numpy()
            Z.append(mu)
            g_batch = []
    
    return np.concatenate(Z, 0)


def prior_validity(train_data, model, infer_batch_size=64, data_type='igraph', subg=True, device=None, scale_to_train_range=False):
    # data_type: igraph, pygraph
    
    if scale_to_train_range:
        Z_train = extract_latent_z(train_data, model, data_type, 0, infer_batch_size)
        z_mean, z_std = Z_train.mean(0), Z_train.std(0)
        z_mean, z_std = torch.FloatTensor(z_mean).to(device), torch.FloatTensor(z_std).to(device)
    
    n_latent_points = 1000
    decode_times = 10
    valid_dags = 0
    valid_ckts = 0
    print('Prior validity experiment begins...')
    G = []
    G_valid = []
    Ckt_valid = []
    if data_type == 'igraph':
        G_train = train_data
    elif data_type == 'pygraph':
        G_train = [pygraph_to_igraph(g) for g in train_data]
    elif data_type == 'tensor':
        G_train = [g.to(device) for g in G_train]
        G_train = model._collate_fn(G_train)
        G_train = model.construct_igraph(G_train[:, :, :model.nvt], G_train[:, :, model.nvt:], False)
    else:
        raise NotImplementedError()
    
    pbar = tqdm(range(n_latent_points))
    cnt = 0
    for i in pbar:
        cnt += 1
        if cnt == infer_batch_size or i == n_latent_points - 1:
            z = torch.randn(cnt, model.nz).to(model.get_device())
            if scale_to_train_range:
                z = z * z_std + z_mean  # move to train's latent range
            for j in range(decode_times):
                g_batch = model.decode(z)
                G.extend(g_batch)
                for g in g_batch:
                    if is_valid_DAG(g, subg):
                        valid_dags += 1
                        G_valid.append(g)
                    if is_valid_Circuit(g, subg=subg):
                        valid_ckts += 1
                        Ckt_valid.append(g)
            cnt = 0

    r_valid_dag = valid_dags / (n_latent_points * decode_times)
    print('Ratio of valid DAG decodings from the prior: {:.4f}'.format(r_valid_dag))

    r_valid_ckt = valid_ckts / (n_latent_points * decode_times)
    print('Ratio of valid Circuits decodings from the prior: {:.4f}'.format(r_valid_ckt))

    r_novel = 1 - ratio_same_DAG(G_train, G_valid)
    print('Ratio of novel graphs out of training data: {:.4f}'.format(r_novel))
    return r_valid_dag, r_valid_ckt, r_novel

#### train test dataset loader

# load datasets
def train_test_generator_topo_simple(ng=10000, name='circuit_example'):
    g_list = []
    n_graph = ng
    with open(name, 'r') as f:
        for g_id in tqdm(range(n_graph)):
            all_rows= []
            row = f.readline().strip().split()
            num_subg, num_node, stage = [int(w) for w in row]
            # loading subg based graph information
            g = igraph.Graph(directed=True)
            g.add_vertices(num_subg)
            for i in range(num_subg):
                # ith row is the node with index i
                row_ = f.readline().strip().split()
                row = [float(w) for w in row_]
                all_rows.append(row)
                subg_type = int(row[0])
                #i = int(row[1])
                g.vs[i]['type'] = subg_type
                num_edges = int(row[3])
                vid = int(row[2])
                g.vs['vid'] = vid
                predecessors = [int(row[w]) for w in range(4, 4 + num_edges)]
                if i != 0:
                    for j in predecessors:
                        g.add_edge(j, i)
                if i == 0:
                    #subg_nod = row[4 + num_edges]
                    subg_nod_types = [8]
                    subg_nod_feats = [0.0]
                    subg_flat_adj = [1]
                #elif i == 1:
                #    subg_nod_types = [9]
                #    subg_nod_feats = [0.0]
                #    subg_flat_adj = [1]
                else:
                    #print(i)
                    if num_edges == 0:
                        subg_nod = int(row[5])
                    else:
                        subg_nod = int(row[4 + num_edges])
                    #subg_nod = int(row[4 + num_edges])
                    #print([row[w] for w in range(5 + num_edges, 5 + num_edges + subg_nod)])
                    subg_nod_types = [int(row[w]) for w in range(5 + num_edges, 5 + num_edges + subg_nod)]
                    subg_nod_feats = [row[w] for w in range(5 + num_edges + subg_nod, 5 + num_edges + 2 * subg_nod)]
                    subg_flat_adj = [int(row[w]) for w in range(5 + num_edges + 2 * subg_nod, 5 + num_edges + 2 * subg_nod + subg_nod * subg_nod)]
                    #print(subg_flat_adj)
                g.vs[i]['subg_ntypes'] = subg_nod_types
                g.vs[i]['subg_nfeats'] = subg_nod_feats
                g.vs[i]['subg_adj'] = subg_flat_adj
            # loading overall graph information
            g_all = igraph.Graph(directed=True)
            g_all.add_vertices(num_node)
            for i in range(num_node):
                row_ = f.readline().strip().split()
                row = [float(w) for w in row_]
                all_rows.append(row)
                type_ = int(row[0])
                vid_ = int(row[1])
                feat_ = row[2]
                g_all.vs[i]['type'] = type_
                g_all.vs[i]['feat'] = feat_
                g_all.vs[i]['vid'] = vid_
                if len(row) > 3:
                    predecessors = [int(row[w]) for w in range(3, len(row))]
                    for j in predecessors:
                        g_all.add_edge(j,i)
            subg_order = g.topological_sorting()
            allg_order = g_all.topological_sorting()
            subg_row_info = all_rows[:num_subg]
            allg_row_info = all_rows[num_subg:]
            
            g_sort = igraph.Graph(directed=True)
            g_sort.add_vertices(num_subg)
            dic_order = {i:j for i,j in zip(subg_order,range(num_subg))}
            #print(dic_order)
            for i, idx in enumerate(subg_order):
                #print(row)
                row = subg_row_info[idx]
                subg_type = int(row[0])
                g_sort.vs[i]['type'] = subg_type
                vid = int(row[2])
                g_sort.vs[i]['vid'] = vid
                num_edges = int(row[3])
                predecessors = [dic_order[int(row[w])] for w in range(4, 4 + num_edges)]
                if i != 0:
                    for j in predecessors:
                        g_sort.add_edge(j, i)
                if i == 0:
                    #subg_nod = row[4 + num_edges]
                    subg_nod_types = [8]
                    subg_nod_feats = [0.0]
                    subg_flat_adj = [1]
                #elif i == 1:
                #    subg_nod_types = [9]
                #    subg_nod_feats = [0.0]
                #    subg_flat_adj = [1]
                else:
                    #print(i)
                    if num_edges == 0:
                        subg_nod = int(row[5])
                    else:
                        subg_nod = int(row[4 + num_edges])
                    #subg_nod = int(row[4 + num_edges])
                    #print([row[w] for w in range(5 + num_edges, 5 + num_edges + subg_nod)])
                    subg_nod_types = [int(row[w]) for w in range(5 + num_edges, 5 + num_edges + subg_nod)]
                    subg_nod_feats = [row[w] for w in range(5 + num_edges + subg_nod, 5 + num_edges + 2 * subg_nod)]
                    subg_flat_adj = [int(row[w]) for w in range(5 + num_edges + 2 * subg_nod, 5 + num_edges + 2 * subg_nod + subg_nod * subg_nod)]
                    #print(subg_flat_adj)
                g_sort.vs[i]['subg_ntypes'] = subg_nod_types
                g_sort.vs[i]['subg_nfeats'] = subg_nod_feats
                g_sort.vs[i]['subg_adj'] = subg_flat_adj               
            
            
            g_all_sort = igraph.Graph(directed=True)
            g_all_sort.add_vertices(num_node)
            dic_order = {i:j for i,j in zip(allg_order,range(num_node))}
            for i, idx in enumerate(allg_order):
                row = allg_row_info[idx]
                type_ = int(row[0])
                vid_ = int(row[1])
                feat_ = row[2]
                g_all_sort.vs[i]['type'] = type_
                g_all_sort.vs[i]['feat'] = feat_
                g_all_sort.vs[i]['vid'] = vid_
                if len(row) > 3:
                    predecessors = [dic_order[int(row[w])] for w in range(3, len(row))]
                    for j in predecessors:
                        g_all_sort.add_edge(j,i)
            g_sort = r_c_gm_extractor(g_sort, subg_indi = SUBG_INDI)
            g_list.append((g_sort, g_all_sort))
    return g_list[:int(ng*0.9)], g_list[int(ng*0.9):]



def train_test_generator_topo_order_dist(ng=10000, name='circuit_example'):
    g_list = []
    n_graph = ng
    with open(name, 'r') as f:
        for g_id in tqdm(range(n_graph)):
            all_rows= []
            row = f.readline().strip().split()
            num_subg, num_node, stage = [int(w) for w in row]
            # loading subg based graph information
            g = igraph.Graph(directed=True)
            g.add_vertices(num_subg)
            for i in range(num_subg):
                # ith row is the node with index i
                row_ = f.readline().strip().split()
                row = [int(w) for w in row_]
                all_rows.append(row)
                subg_type = int(row[0])
                #i = int(row[1])
                g.vs[i]['type'] = subg_type
                num_edges = int(row[3])
                predecessors = [int(row[w]) for w in range(4, 4 + num_edges)]
                if i != 0:
                    for j in predecessors:
                        g.add_edge(j, i)
                if i == 0:
                    #subg_nod = row[4 + num_edges]
                    subg_nod_types = [8]
                    subg_nod_feats = [0.0]
                    subg_flat_adj = [1]
                #elif i == 1:
                #    subg_nod_types = [9]
                #    subg_nod_feats = [0.0]
                #    subg_flat_adj = [1]
                else:
                    #print(i)
                    if num_edges == 0:
                        subg_nod = int(row[5])
                    else:
                        subg_nod = int(row[4 + num_edges])
                    #subg_nod = int(row[4 + num_edges])
                    #print([row[w] for w in range(5 + num_edges, 5 + num_edges + subg_nod)])
                    subg_nod_types = [int(row[w]) for w in range(5 + num_edges, 5 + num_edges + subg_nod)]
                    subg_nod_feats = [row[w] for w in range(5 + num_edges + subg_nod, 5 + num_edges + 2 * subg_nod)]
                    subg_flat_adj = [int(row[w]) for w in range(5 + num_edges + 2 * subg_nod, 5 + num_edges + 2 * subg_nod + subg_nod * subg_nod)]
                    #print(subg_flat_adj)
                g.vs[i]['subg_ntypes'] = subg_nod_types
                g.vs[i]['subg_nfeats'] = subg_nod_feats
                g.vs[i]['subg_adj'] = subg_flat_adj
            # loading overall graph information
            g_all = igraph.Graph(directed=True)
            g_all.add_vertices(num_node)
            for i in range(num_node):
                row_ = f.readline().strip().split()
                row = [int(w) for w in row_]
                all_rows.append(row)
                type_ = int(row[0])
                feat_ = row[1]
                g_all.vs[i]['type'] = type_
                g_all.vs[i]['feat'] = feat_
                if len(row) > 2:
                    predecessors = [int(row[w]) for w in range(2, len(row))]
                    for j in predecessors:
                        g_all.add_edge(j,i)
            subg_order = g.topological_sorting()
            allg_order = g_all.topological_sorting()
            subg_row_info = all_rows[:num_subg]
            allg_row_info = all_rows[num_subg:]
            
            g_sort = igraph.Graph(directed=True)
            g_sort.add_vertices(num_subg)
            dic_order = {i:j for i,j in zip(subg_order,range(num_subg))}
            #print(dic_order)
            for i, idx in enumerate(subg_order):
                #print(row)
                row = subg_row_info[idx]
                subg_type = int(row[0])
                g_sort.vs[i]['type'] = subg_type
                num_edges = int(row[3])
                predecessors = [dic_order[int(row[w])] for w in range(4, 4 + num_edges)]
                if i != 0:
                    for j in predecessors:
                        g_sort.add_edge(j, i)
                if i == 0:
                    #subg_nod = row[4 + num_edges]
                    subg_nod_types = [8]
                    subg_nod_feats = [0.0]
                    subg_flat_adj = [1]
                #elif i == 1:
                #    subg_nod_types = [9]
                #    subg_nod_feats = [0.0]
                #    subg_flat_adj = [1]
                else:
                    #print(i)
                    if num_edges == 0:
                        subg_nod = int(row[5])
                    else:
                        subg_nod = int(row[4 + num_edges])
                    #subg_nod = int(row[4 + num_edges])
                    #print([row[w] for w in range(5 + num_edges, 5 + num_edges + subg_nod)])
                    subg_nod_types = [int(row[w]) for w in range(5 + num_edges, 5 + num_edges + subg_nod)]
                    subg_nod_feats = [row[w] for w in range(5 + num_edges + subg_nod, 5 + num_edges + 2 * subg_nod)]
                    subg_flat_adj = [int(row[w]) for w in range(5 + num_edges + 2 * subg_nod, 5 + num_edges + 2 * subg_nod + subg_nod * subg_nod)]
                    #print(subg_flat_adj)
                g_sort.vs[i]['subg_ntypes'] = subg_nod_types
                g_sort.vs[i]['subg_nfeats'] = subg_nod_feats
                g_sort.vs[i]['subg_adj'] = subg_flat_adj               
            
            
            g_all_sort = igraph.Graph(directed=True)
            g_all_sort.add_vertices(num_node)
            dic_order = {i:j for i,j in zip(allg_order,range(num_node))}
            for i, idx in enumerate(allg_order):
                row = allg_row_info[idx]
                type_ = int(row[0])
                feat_ = row[1]
                g_all_sort.vs[i]['type'] = type_
                g_all_sort.vs[i]['feat'] = feat_
                if len(row) > 2:
                    predecessors = [dic_order[int(row[w])] for w in range(2, len(row))]
                    for j in predecessors:
                        g_all_sort.add_edge(j,i)
            
            g_list.append((g_sort, g_all_sort))
    return g_list[:int(ng*0.9)], g_list[int(ng*0.9):]


