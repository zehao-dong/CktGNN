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

def subg_exist(p = 0.5, start_type=0, end_type=10):
    pe = np.random.uniform(0,1)
    if pe <= p:
        return np.random.randint(low=start_type, high=end_type, size=1)[0]
    else:
        return None
    
def main_path_subg(p = 0.5):
    pe = np.random.uniform(0,1)
    candidate = [6,7,10,11]
    if pe <= p:
        idx = np.random.randint(low=0, high=4, size=1)[0]
        return candidate[idx]
    else:
        return None
    
def gnd_subg(p = 0.5):
    pe = np.random.uniform(0,1)
    candidate = [2,3,4,5]
    if pe <= p:
        idx = np.random.randint(low=0, high=4, size=1)[0]
        return candidate[idx]
    else:
        return None
    
def inter_select(stage=3):
    if stage == 3:
        candidate = [0,1,2,3]
    elif stage == 2:
        candidate = [0,1,2]
    else:
        raise MyException('Undefined number of stages')
    id1 = random.choice(candidate)
    candidate.pop(id1)
    id2 = random.choice(candidate)
    idx = [id1, id2]
    if 1 in idx:
        out_id = 1
        for j in idx:
            if j != 1:
                in_id = j
    else:
        in_id, out_id = min(idx), max(idx)
    return in_id, out_id


def rand_thre(p=0.5):
    pe = np.random.uniform(0,1)
    if pe <= p:
        return True
    else:
        return False

def compute_num_nodes(subg_list, subg_node):
    return np.sum([len(subg_node[i]) for i in subg_list])

def val_generator(min_val=0, max_val=1001, scale=10, size=5):
    return np.random.randint(low=min_val, high=max_val, size=size)/np.float(scale)

def subg_flaten_adj(num_node, con_type = 'series'):
    if num_node == 1:
        return [0,1,0,1,0,1,0,1,0]
    elif num_node == 2:
        if con_type == 'series':
            return [0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0]
        elif con_type == 'parral':
            return [0,1,1,0,1,0,0,1,0,1,0,1,0,1,1,0]
        else:
            raise MyException('Undefined connection type')
    elif num_node == 3:
        if con_type == 'series':
            return [0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,0,1,0]
        elif con_type == 'parral':
            return [0,1,1,1,0,1,0,0,0,1,0,1,0,0,1,0,0,1,0,1,0,1,1,1,0]
        else:
            raise MyException('Undefined connection type')
    else:
        raise MyException('Undefined subgraph type')
        
def subg_feature_type(subg, subg_node, node_type, min_val=0, max_val=1001, scale=10, size=5):
    sub_types = [6]
    sub_feats = [-1]
    sub_types += [node_type[i] for i in subg_node[subg]]
    size = len(subg_node[subg])
    sub_feats += list(val_generator(min_val, max_val, scale, size))
    sub_types += [7]
    sub_feats += [-1]
    return sub_types, sub_feats

def subg_feature_type_dis(subg, subg_node, node_type, min_val=1, max_val=102, scale=1, size=5):
    sub_types = [6]
    sub_feats = [-1] #[0]
    sub_types += [node_type[i] for i in subg_node[subg]]
    size = len(subg_node[subg])
    sub_feats += list(val_generator(min_val, max_val, scale, size))
    sub_types += [7]
    sub_feats += [-1] # [1]
    return sub_types, sub_feats

def r_c_gm_extractor(g, subg_indi = SUBG_INDI):
    for v in g.vs:
        type_ = v['type']
        subg_feats = v['subg_nfeats'][1:-1]
        if type_ == 0 or type_ == 1:
            v['r'], v['c'], v['gm'] = 0, 0, 0
        else:
            name_indi = ['r', 'c', 'gm']
            for i in range(3):
                if i not in subg_indi[type_]:
                    v[name_indi[i]] = 0
            for k,i in enumerate(subg_indi[type_]):
                v[name_indi[i]] = int(subg_feats[k])
    return g

'''Network visualization'''
def plot_circuits(g_pair, res_dir, name, backbone=False, data_type='igraph', pdf=False):
    # backbone: puts all nodes in a straight line
    file_name = os.path.join(res_dir, name+'.png')
    if pdf:
        file_name = os.path.join(res_dir, name+'.pdf')
    if data_type == 'igraph':
        draw_subg_ckt(g_pair[0], file_name, backbone)
    elif data_type == 'pygraph':
        draw_ckt(g_pair[1], file_name)
    return file_name


def draw_subg_ckt(g, path, backbone=False):
    graph = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open')
    if g is None:
        add_subg_node(graph, 0, 0)
        graph.layout(prog='dot')
        graph.draw(path)
        return
    for idx in range(g.vcount()):
        add_subg_node(graph, idx, g.vs[idx]['type'])
    for idx in range(g.vcount()):
        for node in g.get_adjlist(igraph.IN)[idx]:
            if node == idx-1 and backbone:
                graph.add_edge(node, idx, weight=1)
            else:
                graph.add_edge(node, idx, weight=0)
    graph.layout(prog='dot')
    graph.draw(path)


def add_subg_node(graph, node_id, label, shape='box', style='filled'):
    if label == 0:  
        label = 'input'
        color = 'orchid'
    elif label == 1:
        label = 'output'
        color = 'pink'
    elif label == 2:
        label = 'R'
        color = 'yellow'
    elif label == 3:
        label = 'C'
        color = 'lawngreen'
    elif label == 4:
        label = 'R serie C'
        color = 'greenyellow'
    elif label == 5:
        label = 'R paral C'
        color = 'yellowgreen'
    elif label == 6:
        label = '+gm+'
        color = 'cyan'
    elif label == 7:
        label = '-gm+'
        color = 'lightblue'
    elif label == 8:
        label = '+gm-'
        color = 'deepskyblue'
    elif label == 9:
        label = '-gm-'
        color = 'dodgerblue'
    elif label == 10:
        label = 'C paral +gm+'
        color = 'lime'
    elif label == 11:
        label = 'C paral -gm+'
        color = 'seagreen'
    elif label == 12:
        label = 'C paral +gm-'
        color = 'springgreen'
    elif label == 13:
        label = 'C paral -gm-'
        color = 'limegreen'
    elif label == 14:
        label = 'R paral +gm+'
        color = 'lightcoral'
    elif label == 15:
        label = 'R paral -gm+'
        color = 'coral'
    elif label == 16:
        label = 'R paral +gm-'
        color = 'salmon'
    elif label == 17:
        label = 'R paral gm-'
        color = 'red'
    elif label == 18:
        label = 'R paral C paral +gm+'
        color = 'darkorange'
    elif label == 19:
        label = 'R paral C paral -gm+'
        color = 'bisque'
    elif label == 20:
        label = 'R paral C paral +gm-'
        color = 'nawajowhite'
    elif label == 21:
        label = 'R paral C paral -gm-'
        color = 'orange'
    elif label == 22:
        label = 'R serie C serie +gm+'
        color = 'plum'
    elif label == 23:
        label = 'R serie C serie -gm+'
        color = 'violet'
    elif label == 24:
        label = 'R serie C serie +gm-'
        color = 'mediumpurple'
    elif label == 25:
        label = 'R serie C serie -gm-'
        color = 'blueviolet'
    else:
        label = ''
        color = 'aliceblue'
    #label = f"{label}\n({node_id})"
    label = f"{label}"
    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style, fontsize=24)

def draw_ckt(g, path, backbone=False):
    graph = pgv.AGraph(directed=True, strict=True, fontname='Helvetica', arrowtype='open')
    if g is None:
        add_node(graph, 0, 0)
        graph.layout(prog='dot')
        graph.draw(path)
        return
    for idx in range(g.vcount()):
        add_node(graph, idx, g.vs[idx]['type'])
    for idx in range(g.vcount()):
        for node in g.get_adjlist(igraph.IN)[idx]:
            if node == idx-1 and backbone:
                graph.add_edge(node, idx, weight=1)
            else:
                graph.add_edge(node, idx, weight=0)
    graph.layout(prog='dot')
    graph.draw(path)


def add_node(graph, node_id, label, shape='box', style='filled'):
    if label == 8:  
        label = 'input'
        color = 'orchid'
    elif label == 9:
        label = 'output'
        color = 'pink'
    elif label == 0:
        label = 'R'
        color = 'yellow'
    elif label == 1:
        label = 'C'
        color = 'lawngreen'
    elif label == 2:
        label = '+gm+'
        color = 'cyan'
    elif label == 3:
        label = '-gm+'
        color = 'lightblue'
    elif label == 4:
        label = '+gm-'
        color = 'deepskyblue'
    elif label == 5:
        label = '-gm-'
        color = 'dodgerblue'
    elif label == 6:
        label = 'sudo_in'
        color = 'silver'
    elif label == 7:
        label = 'sudo_out'
        color = 'light_grey'
    else:
        label = ''
        color = 'aliceblue'
    #label = f"{label}\n({node_id})"
    label = f"{label}"
    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style, fontsize=24)

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
