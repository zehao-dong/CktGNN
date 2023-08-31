import pandas as pd
import numpy as np
import csv
import pandas as pd
import torch
from tqdm import tqdm
import igraph
import networkx as nx
import random
from utils_src import *


def circuit_generation(num_graphs, subg_node, sung_con, node_type, start_type=2, end_type=26, dataset='circuit', K=20):
    print('generating circuits')
    pbar = tqdm(range(num_graphs))
    k_total = 0
    with open(dataset, 'w') as f:
        for cir in pbar:
            if k_total%K == 0:
                indicator = [1] * 2 + [0] * 6
                subg_list = []
                node_list = []
                subg_list.append(0)
                subg_list.append(1)
                node_list.append(0)
                node_list.append(1)
                for sub_i in range(6): # determined by the domain knowledge
                    if sub_i < 2:
                        #subg = subg_exist(p = 1.01,start_type=start_type, end_type=end_type)
                        subg = main_path_subg(p=1.01)
                    elif sub_i == 2:
                        #subg = subg_exist(p = 0.9,start_type=start_type, end_type=end_type)  
                        subg = main_path_subg(p=0.8)
                    elif sub_i == 4:
                        subg = gnd_subg(p=0.6)
                    else:
                        subg = subg_exist(p = 0.6,start_type=start_type, end_type=end_type)

                    if subg is not None:
                        subg_list.append(subg)
                        indicator[sub_i+2] = 1
                # overall information       
                num_subg = len(subg_list)
                #print(subg_list)
                #print(subg_node)
                num_nodes = compute_num_nodes(subg_list, subg_node)
                if indicator[4] > 0:
                    stage = 3
                else:
                    stage = 2
            for val in [num_subg, num_nodes, stage]:
                f.write(str(val))
                f.write(' ')
            f.write('\r\n')
            """sub graph information: as DAGs"""
            ### [subg type,  index, position , number edges (as end), predecessive ind, 
            # nodes in subg, nodes' feature in sub, nodes' type in sub, flatten adj]
            if k_total%K == 0:
                pre_subg_dict = {}
                num_edge_dict = {}
                pos_subg_dict = {}
                # init
                pre_subg_dict[0] = []
                num_edge_dict[0] = 0
                pos_subg_dict[0] = 0
                pre_subg_dict[2] = [0]
                #num_edge_dict[2] = 1
                pos_subg_dict[2] = 2
                pre_subg_dict[3] = [2]
                #num_edge_dict[3] = 1
                pos_subg_dict[3] = 3
                pos_subg_dict[1] = 1
                if indicator[4] > 0: # three stage amplifier
                    stage = 3
                    pre_subg_dict[4] = [3]
                    #num_edge_dict[4] = 1
                    pos_subg_dict[4] = 4
                    if indicator[5] > 0.1 and indicator[6] > 0.1 and indicator[7] > 0.1:
                        pre_subg_dict[1] = [4,6]
                        pre_subg_dict[7] = [0]

                        in_id1, out_ind1 = inter_select(stage=3) # for position 5
                        in_id2, out_ind2 = inter_select(stage=3) # for position 7
                        pre_subg_dict[5] = [in_id1]
                        pre_subg_dict[out_ind1].append(5)
                        pre_subg_dict[7] = [in_id2]
                        pre_subg_dict[out_ind2].append(7)
                        if rand_thre(p=0.5):
                            pre_subg_dict[6] = [2]
                        else:
                            pre_subg_dict[6] = [3]
                        #pre_subg_dict[6] = [3]
                        pos_subg_dict[5] = 5
                        pos_subg_dict[6] = 6
                        pos_subg_dict[7] = 7
                    elif indicator[5] > 0.1 and indicator[6] > 0.1 and indicator[7] <= 0.1:
                        pre_subg_dict[1] = [4,6]
                        in_id1, out_ind1 = inter_select(stage=3) # for position 5
                        pre_subg_dict[5] = [in_id1]
                        pre_subg_dict[out_ind1].append(5)
                        if rand_thre(p=0.5):
                            pre_subg_dict[6] = [2]
                        else:
                            pre_subg_dict[6] = [3]
                        #pre_subg_dict[6] = [3]
                        pos_subg_dict[5] = 5
                        pos_subg_dict[6] = 6
                        pre_subg_dict[7] = []

                    elif indicator[5] > 0.1 and indicator[6] <= 0.1 and indicator[7] > 0.1 :
                        pre_subg_dict[1] = [4]
                        in_id1, out_ind1 = inter_select(stage=3) # for position 5
                        in_id2, out_ind2 = inter_select(stage=3) # for position 7
                        pre_subg_dict[5] = [in_id1]
                        pre_subg_dict[out_ind1].append(5)
                        pre_subg_dict[6] = [in_id2]
                        pre_subg_dict[out_ind2].append(6)
                        #pre_subg_dict[6] = []
                        pos_subg_dict[5] = 5
                        pos_subg_dict[6] = 7
                        pre_subg_dict[7] = []

                    elif indicator[5] > 0.1 and indicator[6] <= 0.1 and indicator[7] <= 0.1 :
                        pre_subg_dict[1] = [4]
                        in_id1, out_ind1 = inter_select(stage=3) # for position 5
                        pre_subg_dict[5] = [in_id1]
                        pre_subg_dict[out_ind1].append(5)
                        #pre_subg_dict[5] = [2]
                        pre_subg_dict[6] = []
                        pre_subg_dict[7] = []
                        pos_subg_dict[5] = 5

                    elif indicator[5] <= 0.1 and indicator[6] > 0.1 and indicator[7] > 0.1:
                        pre_subg_dict[1] = [4,5]
                        in_id1, out_ind1 = inter_select(stage=3) # for position 5
                        pre_subg_dict[6] = [in_id1]
                        pre_subg_dict[out_ind1].append(6)
                        if rand_thre(p=0.5):
                            pre_subg_dict[5] = [2]
                        else:
                            pre_subg_dict[5] = [3]
                        #pre_subg_dict[5] = [3] 
                        #pre_subg_dict[6] = []
                        pos_subg_dict[5] = 6
                        pos_subg_dict[6] = 7
                        pre_subg_dict[7] = []

                    elif indicator[5] <= 0.1 and indicator[6] > 0.1 and indicator[7] <= 0.1:
                        pre_subg_dict[1] = [4,5]
                        if rand_thre(p=0.5):
                            pre_subg_dict[5] = [2]
                        else:
                            pre_subg_dict[5] = [3]
                        #pre_subg_dict[5] = [3] 
                        pre_subg_dict[6] = []
                        pre_subg_dict[7] = []
                        pos_subg_dict[5] = 6

                    elif indicator[5] <= 0.1 and indicator[6] <= 0.1 and indicator[7] > 0.1:
                        pre_subg_dict[1] = [4]
                        in_id1, out_ind1 = inter_select(stage=3) # for position 5
                        pre_subg_dict[5] = [in_id1]
                        pre_subg_dict[out_ind1].append(5)

                        pre_subg_dict[6] = []
                        pre_subg_dict[7] = []
                        #pre_subg_dict[5] = []
                        pos_subg_dict[5] = 7

                    else:
                        pre_subg_dict[1] = [4]
                        pre_subg_dict[5] = []
                        pre_subg_dict[6] = []
                        pre_subg_dict[7] = []
                    num_edge_dict[1] = len(pre_subg_dict[1]) 
                    num_edge_dict[2] = len(pre_subg_dict[2]) 
                    num_edge_dict[3] = len(pre_subg_dict[3]) 
                    num_edge_dict[4] = len(pre_subg_dict[4]) 
                    num_edge_dict[5] = len(pre_subg_dict[5])  
                    num_edge_dict[6] = len(pre_subg_dict[6])
                    num_edge_dict[7] = len(pre_subg_dict[7]) 
                    #pre_subg_dict[1] = []
                    #for k in [4,5,6]:
                    #    if indicator[k] > 0.1:
                    #        pre_subg_dict[1].append(k)
                    #num_edge_dict[1] = len(pre_subg_dict[1])
                    #pre_subg_dict[5] = [2] if indicator[5] > 0.1 else []
                    #num_edge_dict[5] = 1 if indicator[5] > 0.1 else 0
                    #pre_subg_dict[6] = [3] if indicator[6] > 0.1 else []
                    #num_edge_dict[6] = 1 if indicator[5] > 0.1 else 0
                else: # two stage amplifier
                    stage =2
                    if indicator[5] > 0.1 and indicator[6] > 0.1 and indicator[7] > 0.1:
                        pre_subg_dict[1] = [3,5]
                        in_id1, out_ind1 = inter_select(stage=2) # for position 5
                        in_id2, out_ind2 = inter_select(stage=2) # for position 7
                        pre_subg_dict[4] = [in_id1]
                        pre_subg_dict[out_ind1].append(4)
                        pre_subg_dict[6] = [in_id2]
                        pre_subg_dict[out_ind2].append(6)

                        pre_subg_dict[5] = [2]
                        pos_subg_dict[4] = 5
                        pos_subg_dict[5] = 6
                        pos_subg_dict[6] = 7

                    elif indicator[5] > 0.1 and indicator[6] > 0.1 and indicator[7] <= 0.1:
                        pre_subg_dict[1] = [3,5]
                        in_id1, out_ind1 = inter_select(stage=2) # for position 5
                        pre_subg_dict[4] = [in_id1]
                        pre_subg_dict[out_ind1].append(4)

                        pre_subg_dict[5] = [2]
                        pre_subg_dict[6] = []
                        pos_subg_dict[4] = 5
                        pos_subg_dict[5] = 6

                    elif indicator[5] > 0.1 and indicator[6] <= 0.1 and indicator[7] > 0.1:
                        pre_subg_dict[1] = [3]
                        in_id1, out_ind1 = inter_select(stage=2) # for position 5
                        in_id2, out_ind2 = inter_select(stage=2) # for position 7
                        pre_subg_dict[4] = [in_id1]
                        pre_subg_dict[out_ind1].append(4)
                        pre_subg_dict[5] = [in_id2]
                        pre_subg_dict[out_ind2].append(5)

                        #pre_subg_dict[5] = []
                        pre_subg_dict[6] = []
                        pos_subg_dict[4] = 5
                        pos_subg_dict[5] = 7

                    elif indicator[5] > 0.1 and indicator[6] <= 0.1 and indicator[7] <= 0.1:
                        pre_subg_dict[1] = [3]
                        in_id1, out_ind1 = inter_select(stage=2) # for position 5
                        pre_subg_dict[4] = [in_id1]
                        pre_subg_dict[out_ind1].append(4)
                        pre_subg_dict[5] = []
                        pre_subg_dict[6] = []
                        pos_subg_dict[4] = 5

                    elif indicator[5] <= 0.1 and indicator[6] > 0.1 and indicator[7] > 0.1:
                        pre_subg_dict[1] = [3,4]
                        in_id1, out_ind1 = inter_select(stage=2) # for position 5
                        pre_subg_dict[5] = [in_id1]
                        pre_subg_dict[out_ind1].append(5)

                        pre_subg_dict[4] = [2] 
                        #pre_subg_dict[5] = []
                        pre_subg_dict[6] = []
                        pos_subg_dict[4] = 6
                        pos_subg_dict[5] = 7

                    elif indicator[5] <= 0.1 and indicator[6] > 0.1 and indicator[7] <= 0.1:
                        pre_subg_dict[1] = [3,4]
                        pre_subg_dict[4] = [2] 
                        pre_subg_dict[5] = []
                        pre_subg_dict[6] = []
                        pos_subg_dict[4] = 6

                    elif indicator[5] <= 0.1 and indicator[6] <= 0.1 and indicator[7] > 0.1:
                        pre_subg_dict[1] = [3]
                        in_id1, out_ind1 = inter_select(stage=2) # for position 5
                        pre_subg_dict[4] = [in_id1]
                        pre_subg_dict[out_ind1].append(4)

                        #pre_subg_dict[4] = [] 
                        pre_subg_dict[5] = []
                        pre_subg_dict[6] = []
                        pos_subg_dict[4] = 7
                    else:
                        pre_subg_dict[1] = [3]
                        pre_subg_dict[4] = [] 
                        pre_subg_dict[5] = []
                        pre_subg_dict[6] = []
                    num_edge_dict[1] = len(pre_subg_dict[1]) 
                    num_edge_dict[2] = len(pre_subg_dict[2]) 
                    num_edge_dict[3] = len(pre_subg_dict[3]) 
                    num_edge_dict[4] = len(pre_subg_dict[4]) 
                    num_edge_dict[5] = len(pre_subg_dict[5])  
                    num_edge_dict[6] = len(pre_subg_dict[6])
                    #num_edge_dict[7] = len(pre_subg_dict[7]) 
                    #num_edge_dict[1] = len(pre_subg_dict[1])
                    #num_edge_dict[4] = len(pre_subg_dict[4])
                    #num_edge_dict[5] = len(pre_subg_dict[5])
                    #num_edge_dict[6] = len(pre_subg_dict[6]) 
                    #num_edge_dict[7] = len(pre_subg_dict[7]) 

                    #pre_subg_dict[4] = [2] if indicator[5] > 0.1 else []
                    #num_edge_dict[4] = 1 if indicator[5] > 0.1 else 0
                    #pre_subg_dict[5] = [2] if indicator[6] > 0.1 else []
                    #num_edge_dict[5] = 1 if indicator[5] > 0.1 else 0
            k_total += 1
            
            SUB_FEAT = {}
            ### [subg type,  index, position , number edges (as end), predecessive ind, 
            # nodes in subg, nodes' feature in sub, nodes' type in sub, flatten adj]
            for i in range(num_subg):
                sub_inform = []
                if i == 0:
                    sub_inform = [0, i, 0, 0, 0, 1, 8, 0, 1] 
                    sub_feats = [-1,0, -1]
                elif i == 1:
                    subg_t = subg_list[1]
                    pos_ = pos_subg_dict[1]
                    num_edge = num_edge_dict[1]
                    predecessive_ind = pre_subg_dict[1]
                    sub_inform = [subg_t, 1, pos_, num_edge] + predecessive_ind + [1, 9, 0, 1]
                    sub_feats = [-1,0,-1]
                    SUB_FEAT[1] = sub_feats
                else:
                    subg_t = subg_list[i]
                    num_edge = num_edge_dict[i]
                    pos_ = pos_subg_dict[i]
                    predecessive_ind = pre_subg_dict[i]
                    if num_edge == 0 and len(predecessive_ind) == 0:
                        predecessive_ind = [0]
                    sub_types, sub_feats = subg_feature_type(subg_list[i], subg_node, node_type, 
                                                             min_val=0, max_val=1001, scale=10, size=5) # can be edited
                    assert(len(sub_types) == len(sub_feats))
                    #print(sub_types)
                    #print(sub_feats)
                    nodes_in_subg = len(sub_types)
                    #flatten_adj = subg_flaten_adj(len(subg_node[subg_list[i]]), con_type = sung_con[subg_list[i]])
                    flatten_adj = subg_flaten_adj(nodes_in_subg-2, con_type = sung_con[subg_list[i]])
                    sub_inform = [subg_t, i, pos_, num_edge] + predecessive_ind + [nodes_in_subg] + sub_types + sub_feats + flatten_adj
                    #print(sub_inform)
                #SUB_INF[subg_list[i]] = sub_inform 
                SUB_FEAT[i] = sub_feats
                for val in sub_inform:
                    f.write(str(val))
                    f.write(' ')
                f.write('\r\n')
            
                
            """graph information: as DAGs"""
            #all_adj = np.zeros((num_nodes,num_nodes))
            #print(stage)
            #print(pre_subg_dict)
            all_predecessive_dict = {}
            all_type_dict = {}
            all_feat_dict = {}
            
            ind_order = []
            if stage == 3:
                main_path = [0,2,3,4,1]
                for i in main_path:
                    if i == 0:
                        ind_order.append(i)
                    else:
                        for j in pre_subg_dict[i]:
                            if j not in ind_order:
                                ind_order.append(j)
                        ind_order.append(i)
            elif stage == 2:
                main_path = [0,2,3,1]
                for i in main_path:
                    if i == 0:
                        ind_order.append(i)
                    else:
                        for j in pre_subg_dict[i]:
                            if j not in ind_order:
                                ind_order.append(j)
                        ind_order.append(i)
            else:
                raise MyException('Undefined number of stages')
            # summerize the ind order
            #for val in ind_order:
            #    f.write(str(val))
            #    f.write(' ')
            #f.write('\r\n')
            
            ind_dict = {}
            node_count = 0
            #print(ind_order)
            for i in ind_order:
                #print(i)
                #print(pre_subg_dict[i])
                num_nodes_subg = len(subg_node[subg_list[i]])
                ind_dict[i] = [node_count, node_count + num_nodes_subg - 1]
                insubg_id = 0
                #print(ind_dict)
                #print(pre_subg_dict)
                for node_id in range(node_count, node_count + num_nodes_subg):
                    all_type_dict[node_id] = node_type[subg_node[subg_list[i]][insubg_id]]
                    all_feat_dict[node_id] = SUB_FEAT[i][insubg_id + 1]
                    if sung_con[subg_list[i]] == 'series':
                        pre_nodes = []
                        if insubg_id == 0:
                            for j in pre_subg_dict[i]:
                                if sung_con[subg_list[j]] == 'series':
                                    pre_nodes.append(ind_dict[j][1])
                                elif sung_con[subg_list[j]] == 'parral':
                                    for h in range(ind_dict[j][0],ind_dict[j][1]+1):
                                        pre_nodes.append(h)
                                else:
                                    pre_nodes.append(ind_dict[j][0])
                        else:
                            pre_nodes.append(node_id-1)
                        all_predecessive_dict[node_id] = pre_nodes
                    elif sung_con[subg_list[i]] == 'parral':
                        pre_nodes = []
                        for j in pre_subg_dict[i]:
                            if sung_con[subg_list[j]] == 'series':
                                pre_nodes.append(ind_dict[j][1])
                            elif sung_con[subg_list[j]] == 'parral':
                                for h in range(ind_dict[j][0],ind_dict[j][1]+1):
                                    pre_nodes.append(h)
                            else:
                                pre_nodes.append(ind_dict[j][0])
                        all_predecessive_dict[node_id] = pre_nodes
                    else:
                        pre_nodes = []
                        for j in pre_subg_dict[i]:
                            if sung_con[subg_list[j]] == 'series':
                                pre_nodes.append(ind_dict[j][1])
                            elif sung_con[subg_list[j]] == 'parral':
                                for h in range(ind_dict[j][0],ind_dict[j][1]+1):
                                    pre_nodes.append(h)
                            else:
                                pre_nodes.append(ind_dict[j][0])
                        all_predecessive_dict[node_id] = pre_nodes
                    insubg_id += 1
                node_count += num_nodes_subg
            ### [node type, node feat, previous node]    
            for i in range(num_nodes):
                type_ =  all_type_dict[i]
                feat_ = all_feat_dict[i]
                predecessors_ = all_predecessive_dict[i]
                inform = [type_, feat_] + predecessors_
                for val in inform:
                    f.write(str(val))
                    f.write(' ')
                f.write('\r\n')

                
def circuit_generation_dis(num_graphs, subg_node, sung_con, node_type, start_type=2, end_type=26, dataset='circuit', K=20):
    print('generating circuits')
    pbar = tqdm(range(num_graphs))
    k_total = 0
    with open(dataset, 'w') as f:
        for cir in pbar:
            if k_total % K == 0:
                indicator = [1] * 2 + [0] * 6
                subg_list = []
                node_list = []
                subg_list.append(0)
                subg_list.append(1)
                node_list.append(0)
                node_list.append(1)
                for sub_i in range(6): # determined by the domain knowledge
                    if sub_i < 2:
                        #subg = subg_exist(p = 1.01,start_type=start_type, end_type=end_type)
                        subg = main_path_subg(p=1.01)
                    elif sub_i == 2:
                        #subg = subg_exist(p = 0.9,start_type=start_type, end_type=end_type)  
                        subg = main_path_subg(p=0.8)
                    elif sub_i == 4:
                        subg = gnd_subg(p=0.6)
                    else:
                        subg = subg_exist(p = 0.6,start_type=start_type, end_type=end_type)

                    if subg is not None:
                        subg_list.append(subg)
                        indicator[sub_i+2] = 1
                # overall information       
                num_subg = len(subg_list)
                #print(subg_list)
                #print(subg_node)
                num_nodes = compute_num_nodes(subg_list, subg_node)
                if indicator[4] > 0:
                    stage = 3
                else:
                    stage = 2
            for val in [num_subg, num_nodes, stage]:
                f.write(str(val))
                f.write(' ')
            f.write('\r\n')
            """sub graph information: as DAGs"""
            ### [subg type,  index, position , number edges (as end), predecessive ind, 
            # nodes in subg, nodes' feature in sub, nodes' type in sub, flatten adj]
            if k_total % K == 0:
                pre_subg_dict = {}
                num_edge_dict = {}
                pos_subg_dict = {}
                # init
                pre_subg_dict[0] = []
                num_edge_dict[0] = 0
                pos_subg_dict[0] = 0
                pre_subg_dict[2] = [0]
                #num_edge_dict[2] = 1
                pos_subg_dict[2] = 2
                pre_subg_dict[3] = [2]
                #num_edge_dict[3] = 1
                pos_subg_dict[3] = 3
                pos_subg_dict[1] = 1
                if indicator[4] > 0: # three stage amplifier
                    stage = 3
                    pre_subg_dict[4] = [3]
                    #num_edge_dict[4] = 1
                    pos_subg_dict[4] = 4
                    if indicator[5] > 0.1 and indicator[6] > 0.1 and indicator[7] > 0.1:
                        pre_subg_dict[1] = [4,6]
                        pre_subg_dict[7] = [0]

                        in_id1, out_ind1 = inter_select(stage=3) # for position 5
                        in_id2, out_ind2 = inter_select(stage=3) # for position 7
                        pre_subg_dict[5] = [in_id1]
                        pre_subg_dict[out_ind1].append(5)
                        pre_subg_dict[7] = [in_id2]
                        pre_subg_dict[out_ind2].append(7)
                        if rand_thre(p=0.5):
                            pre_subg_dict[6] = [2]
                        else:
                            pre_subg_dict[6] = [3]
                        #pre_subg_dict[6] = [3]
                        pos_subg_dict[5] = 5
                        pos_subg_dict[6] = 6
                        pos_subg_dict[7] = 7
                    elif indicator[5] > 0.1 and indicator[6] > 0.1 and indicator[7] <= 0.1:
                        pre_subg_dict[1] = [4,6]
                        in_id1, out_ind1 = inter_select(stage=3) # for position 5
                        pre_subg_dict[5] = [in_id1]
                        pre_subg_dict[out_ind1].append(5)
                        if rand_thre(p=0.5):
                            pre_subg_dict[6] = [2]
                        else:
                            pre_subg_dict[6] = [3]
                        #pre_subg_dict[6] = [3]
                        pos_subg_dict[5] = 5
                        pos_subg_dict[6] = 6
                        pre_subg_dict[7] = []

                    elif indicator[5] > 0.1 and indicator[6] <= 0.1 and indicator[7] > 0.1 :
                        pre_subg_dict[1] = [4]
                        in_id1, out_ind1 = inter_select(stage=3) # for position 5
                        in_id2, out_ind2 = inter_select(stage=3) # for position 7
                        pre_subg_dict[5] = [in_id1]
                        pre_subg_dict[out_ind1].append(5)
                        pre_subg_dict[6] = [in_id2]
                        pre_subg_dict[out_ind2].append(6)
                        #pre_subg_dict[6] = []
                        pos_subg_dict[5] = 5
                        pos_subg_dict[6] = 7
                        pre_subg_dict[7] = []

                    elif indicator[5] > 0.1 and indicator[6] <= 0.1 and indicator[7] <= 0.1 :
                        pre_subg_dict[1] = [4]
                        in_id1, out_ind1 = inter_select(stage=3) # for position 5
                        pre_subg_dict[5] = [in_id1]
                        pre_subg_dict[out_ind1].append(5)
                        #pre_subg_dict[5] = [2]
                        pre_subg_dict[6] = []
                        pre_subg_dict[7] = []
                        pos_subg_dict[5] = 5

                    elif indicator[5] <= 0.1 and indicator[6] > 0.1 and indicator[7] > 0.1:
                        pre_subg_dict[1] = [4,5]
                        in_id1, out_ind1 = inter_select(stage=3) # for position 5
                        pre_subg_dict[6] = [in_id1]
                        pre_subg_dict[out_ind1].append(6)
                        if rand_thre(p=0.5):
                            pre_subg_dict[5] = [2]
                        else:
                            pre_subg_dict[5] = [3]
                        #pre_subg_dict[5] = [3] 
                        #pre_subg_dict[6] = []
                        pos_subg_dict[5] = 6
                        pos_subg_dict[6] = 7
                        pre_subg_dict[7] = []

                    elif indicator[5] <= 0.1 and indicator[6] > 0.1 and indicator[7] <= 0.1:
                        pre_subg_dict[1] = [4,5]
                        if rand_thre(p=0.5):
                            pre_subg_dict[5] = [2]
                        else:
                            pre_subg_dict[5] = [3]
                        #pre_subg_dict[5] = [3] 
                        pre_subg_dict[6] = []
                        pre_subg_dict[7] = []
                        pos_subg_dict[5] = 6

                    elif indicator[5] <= 0.1 and indicator[6] <= 0.1 and indicator[7] > 0.1:
                        pre_subg_dict[1] = [4]
                        in_id1, out_ind1 = inter_select(stage=3) # for position 5
                        pre_subg_dict[5] = [in_id1]
                        pre_subg_dict[out_ind1].append(5)

                        pre_subg_dict[6] = []
                        pre_subg_dict[7] = []
                        #pre_subg_dict[5] = []
                        pos_subg_dict[5] = 7

                    else:
                        pre_subg_dict[1] = [4]
                        pre_subg_dict[5] = []
                        pre_subg_dict[6] = []
                        pre_subg_dict[7] = []
                    num_edge_dict[1] = len(pre_subg_dict[1]) 
                    num_edge_dict[2] = len(pre_subg_dict[2]) 
                    num_edge_dict[3] = len(pre_subg_dict[3]) 
                    num_edge_dict[4] = len(pre_subg_dict[4]) 
                    num_edge_dict[5] = len(pre_subg_dict[5])  
                    num_edge_dict[6] = len(pre_subg_dict[6])
                    num_edge_dict[7] = len(pre_subg_dict[7]) 
                    
                else: # two stage amplifier
                    stage =2
                    if indicator[5] > 0.1 and indicator[6] > 0.1 and indicator[7] > 0.1:
                        pre_subg_dict[1] = [3,5]
                        in_id1, out_ind1 = inter_select(stage=2) # for position 5
                        in_id2, out_ind2 = inter_select(stage=2) # for position 7
                        pre_subg_dict[4] = [in_id1]
                        pre_subg_dict[out_ind1].append(4)
                        pre_subg_dict[6] = [in_id2]
                        pre_subg_dict[out_ind2].append(6)

                        pre_subg_dict[5] = [2]
                        pos_subg_dict[4] = 5
                        pos_subg_dict[5] = 6
                        pos_subg_dict[6] = 7

                    elif indicator[5] > 0.1 and indicator[6] > 0.1 and indicator[7] <= 0.1:
                        pre_subg_dict[1] = [3,5]
                        in_id1, out_ind1 = inter_select(stage=2) # for position 5
                        pre_subg_dict[4] = [in_id1]
                        pre_subg_dict[out_ind1].append(4)

                        pre_subg_dict[5] = [2]
                        pre_subg_dict[6] = []
                        pos_subg_dict[4] = 5
                        pos_subg_dict[5] = 6

                    elif indicator[5] > 0.1 and indicator[6] <= 0.1 and indicator[7] > 0.1:
                        pre_subg_dict[1] = [3]
                        in_id1, out_ind1 = inter_select(stage=2) # for position 5
                        in_id2, out_ind2 = inter_select(stage=2) # for position 7
                        pre_subg_dict[4] = [in_id1]
                        pre_subg_dict[out_ind1].append(4)
                        pre_subg_dict[5] = [in_id2]
                        pre_subg_dict[out_ind2].append(5)

                        #pre_subg_dict[5] = []
                        pre_subg_dict[6] = []
                        pos_subg_dict[4] = 5
                        pos_subg_dict[5] = 7

                    elif indicator[5] > 0.1 and indicator[6] <= 0.1 and indicator[7] <= 0.1:
                        pre_subg_dict[1] = [3]
                        in_id1, out_ind1 = inter_select(stage=2) # for position 5
                        pre_subg_dict[4] = [in_id1]
                        pre_subg_dict[out_ind1].append(4)
                        pre_subg_dict[5] = []
                        pre_subg_dict[6] = []
                        pos_subg_dict[4] = 5

                    elif indicator[5] <= 0.1 and indicator[6] > 0.1 and indicator[7] > 0.1:
                        pre_subg_dict[1] = [3,4]
                        in_id1, out_ind1 = inter_select(stage=2) # for position 5
                        pre_subg_dict[5] = [in_id1]
                        pre_subg_dict[out_ind1].append(5)

                        pre_subg_dict[4] = [2] 
                        #pre_subg_dict[5] = []
                        pre_subg_dict[6] = []
                        pos_subg_dict[4] = 6
                        pos_subg_dict[5] = 7

                    elif indicator[5] <= 0.1 and indicator[6] > 0.1 and indicator[7] <= 0.1:
                        pre_subg_dict[1] = [3,4]
                        pre_subg_dict[4] = [2] 
                        pre_subg_dict[5] = []
                        pre_subg_dict[6] = []
                        pos_subg_dict[4] = 6

                    elif indicator[5] <= 0.1 and indicator[6] <= 0.1 and indicator[7] > 0.1:
                        pre_subg_dict[1] = [3]
                        in_id1, out_ind1 = inter_select(stage=2) # for position 5
                        pre_subg_dict[4] = [in_id1]
                        pre_subg_dict[out_ind1].append(4)

                        #pre_subg_dict[4] = [] 
                        pre_subg_dict[5] = []
                        pre_subg_dict[6] = []
                        pos_subg_dict[4] = 7
                    else:
                        pre_subg_dict[1] = [3]
                        pre_subg_dict[4] = [] 
                        pre_subg_dict[5] = []
                        pre_subg_dict[6] = []
                    num_edge_dict[1] = len(pre_subg_dict[1]) 
                    num_edge_dict[2] = len(pre_subg_dict[2]) 
                    num_edge_dict[3] = len(pre_subg_dict[3]) 
                    num_edge_dict[4] = len(pre_subg_dict[4]) 
                    num_edge_dict[5] = len(pre_subg_dict[5])  
                    num_edge_dict[6] = len(pre_subg_dict[6])
                    #num_edge_dict[7] = len(pre_subg_dict[7]) 
                    #num_edge_dict[1] = len(pre_subg_dict[1])
                    #num_edge_dict[4] = len(pre_subg_dict[4])
                    #num_edge_dict[5] = len(pre_subg_dict[5])
                    #num_edge_dict[6] = len(pre_subg_dict[6]) 
                    #num_edge_dict[7] = len(pre_subg_dict[7]) 

                    #pre_subg_dict[4] = [2] if indicator[5] > 0.1 else []
                    #num_edge_dict[4] = 1 if indicator[5] > 0.1 else 0
                    #pre_subg_dict[5] = [2] if indicator[6] > 0.1 else []
                    #num_edge_dict[5] = 1 if indicator[5] > 0.1 else 0
            k_total += 1
            SUB_FEAT = {}
            ### [subg type,  index, position , number edges (as end), predecessive ind, 
            # nodes in subg, nodes' feature in sub, nodes' type in sub, flatten adj]
            for i in range(num_subg):
                sub_inform = []
                if i == 0:
                    sub_inform = [0, i, 0, 0, 0, 1, 8, 0, 1] 
                    sub_feats = [0,102,0]
                elif i == 1:
                    subg_t = subg_list[1]
                    pos_ = pos_subg_dict[1]
                    num_edge = num_edge_dict[1]
                    predecessive_ind = pre_subg_dict[1]
                    sub_inform = [subg_t, 1, pos_, num_edge] + predecessive_ind + [1, 9, 0, 1]
                    sub_feats = [0,102,0]
                    SUB_FEAT[1] = sub_feats
                else:
                    subg_t = subg_list[i]
                    num_edge = num_edge_dict[i]
                    pos_ = pos_subg_dict[i]
                    predecessive_ind = pre_subg_dict[i]
                    if num_edge == 0 and len(predecessive_ind) == 0:
                        predecessive_ind = [0]
                    sub_types, sub_feats = subg_feature_type_dis(subg_list[i], subg_node, node_type, 
                                                             min_val=1, max_val=102, scale=1, size=5) # can be edited
                    assert(len(sub_types) == len(sub_feats))
                    #print(sub_types)
                    #print(sub_feats)
                    nodes_in_subg = len(sub_types)
                    #flatten_adj = subg_flaten_adj(len(subg_node[subg_list[i]]), con_type = sung_con[subg_list[i]])
                    flatten_adj = subg_flaten_adj(nodes_in_subg-2, con_type = sung_con[subg_list[i]])
                    sub_inform = [subg_t, i, pos_, num_edge] + predecessive_ind + [nodes_in_subg] + sub_types + sub_feats + flatten_adj
                    #print(sub_inform)
                #SUB_INF[subg_list[i]] = sub_inform 
                SUB_FEAT[i] = sub_feats
                for val in sub_inform:
                    f.write(str(val))
                    f.write(' ')
                f.write('\r\n')
            
                
            """graph information: as DAGs"""
            #all_adj = np.zeros((num_nodes,num_nodes))
            #print(stage)
            #print(pre_subg_dict)
            all_predecessive_dict = {}
            all_type_dict = {}
            all_feat_dict = {}
            
            ind_order = []
            if stage == 3:
                main_path = [0,2,3,4,1]
                for i in main_path:
                    if i == 0:
                        ind_order.append(i)
                    else:
                        for j in pre_subg_dict[i]:
                            if j not in ind_order:
                                ind_order.append(j)
                        ind_order.append(i)
            elif stage == 2:
                main_path = [0,2,3,1]
                for i in main_path:
                    if i == 0:
                        ind_order.append(i)
                    else:
                        for j in pre_subg_dict[i]:
                            if j not in ind_order:
                                ind_order.append(j)
                        ind_order.append(i)
            else:
                raise MyException('Undefined number of stages')
            # summerize the ind order
            #for val in ind_order:
            #    f.write(str(val))
            #    f.write(' ')
            #f.write('\r\n')
            
            ind_dict = {}
            node_count = 0
            #print(ind_order)
            for i in ind_order:
                #print(i)
                #print(pre_subg_dict[i])
                num_nodes_subg = len(subg_node[subg_list[i]])
                ind_dict[i] = [node_count, node_count + num_nodes_subg - 1]
                insubg_id = 0
                #print(ind_dict)
                #print(pre_subg_dict)
                for node_id in range(node_count, node_count + num_nodes_subg):
                    all_type_dict[node_id] = node_type[subg_node[subg_list[i]][insubg_id]]
                    all_feat_dict[node_id] = SUB_FEAT[i][insubg_id + 1]
                    if sung_con[subg_list[i]] == 'series':
                        pre_nodes = []
                        if insubg_id == 0:
                            for j in pre_subg_dict[i]:
                                if sung_con[subg_list[j]] == 'series':
                                    pre_nodes.append(ind_dict[j][1])
                                elif sung_con[subg_list[j]] == 'parral':
                                    for h in range(ind_dict[j][0],ind_dict[j][1]+1):
                                        pre_nodes.append(h)
                                else:
                                    pre_nodes.append(ind_dict[j][0])
                        else:
                            pre_nodes.append(node_id-1)
                        all_predecessive_dict[node_id] = pre_nodes
                    elif sung_con[subg_list[i]] == 'parral':
                        pre_nodes = []
                        for j in pre_subg_dict[i]:
                            if sung_con[subg_list[j]] == 'series':
                                pre_nodes.append(ind_dict[j][1])
                            elif sung_con[subg_list[j]] == 'parral':
                                for h in range(ind_dict[j][0],ind_dict[j][1]+1):
                                    pre_nodes.append(h)
                            else:
                                pre_nodes.append(ind_dict[j][0])
                        all_predecessive_dict[node_id] = pre_nodes
                    else:
                        pre_nodes = []
                        for j in pre_subg_dict[i]:
                            if sung_con[subg_list[j]] == 'series':
                                pre_nodes.append(ind_dict[j][1])
                            elif sung_con[subg_list[j]] == 'parral':
                                for h in range(ind_dict[j][0],ind_dict[j][1]+1):
                                    pre_nodes.append(h)
                            else:
                                pre_nodes.append(ind_dict[j][0])
                        all_predecessive_dict[node_id] = pre_nodes
                    insubg_id += 1
                node_count += num_nodes_subg
            ### [node type, node feat, previous node]    
            for i in range(num_nodes):
                type_ =  all_type_dict[i]
                feat_ = all_feat_dict[i]
                predecessors_ = all_predecessive_dict[i]
                inform = [type_, feat_] + predecessors_
                for val in inform:
                    f.write(str(val))
                    f.write(' ')
                f.write('\r\n')

