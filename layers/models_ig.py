import numpy as np
import igraph
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

# Some utility functions

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


def one_hot(idx, length):
    if type(idx) in [list, range]:
        if idx == []:
            return None
        idx = torch.LongTensor(idx).unsqueeze(0).t()
        x = torch.zeros((len(idx), length)).scatter_(1, idx, 1)
    else:
        idx = torch.LongTensor([idx]).unsqueeze(0)
        x = torch.zeros((1, length)).scatter_(1, idx, 1)
    return x

def inverse_adj(adj, device):
    n_node = adj.size(0)
    aug_adj = adj + torch.diag(torch.ones(n_node).to(device))
    aug_diag = torch.sum(aug_adj, dim=0)
    return torch.diag(1/aug_diag)


class subc_GNN(nn.Module):
    def __init__(self, num_cat, out_feat, dropout=0.5, num_layer=2, readout='sum', device=None):
        super(subc_GNN, self).__init__()
        self.catag_lin = nn.Linear(num_cat, out_feat)
        self.numer_lin = nn.Linear(1, out_feat)
        self.layers = nn.ModuleList()
        self.emb_dim = 2 * out_feat
        self.dropout = dropout
        self.num_cat = num_cat
        #linlayer = nn.Linear(self.emb_dim, self.emb_dim)
        #act = nn.ReLu()
        #self.layers.append(linlayer)
        #self.layers.append(act)
        #if self.dropout > 0.0001:
        #    drop = nn.Dropout(dropout)
        #    self.layers.append(drop)
        for i in range(num_layer):
            linlayer = nn.Linear(self.emb_dim, self.emb_dim)
            act = nn.ReLU()
            self.layers.append(linlayer)
            if self.dropout > 0.0001:
                drop = nn.Dropout(dropout)
                self.layers.append(drop)
            self.layers.append(act)
        self.device = device  
        self.readout = readout
        self.num_layer = num_layer
        
    def forward(self, G):
        # G is a batch of graphs
        nodes_list = [g.vcount() for g in G]
        num_graphs = len(nodes_list)
        num_nodes = sum(nodes_list)
        sub_nodes_types = []
        sub_nodes_feats = []
        num_subg_nodes = []
        for i in range(num_graphs):
            g = G[i]
            for j in range(nodes_list[i]):
                sub_nodes_types += g.vs[j]['subg_ntypes']
                sub_nodes_feats += g.vs[j]['subg_nfeats']
                num_subg_nodes.append(len(g.vs[j]['subg_ntypes']))
        all_nodes = sum(num_subg_nodes)
        all_adj = torch.zeros(all_nodes,all_nodes)
        node_count = 0
        for i in range(num_graphs):
            g = G[i]
            for j in range(nodes_list[i]):
                adj_flat = g.vs[j]['subg_adj']
                subg_n = len(g.vs[j]['subg_ntypes'])
                all_adj[node_count:node_count+subg_n, node_count:node_count+subg_n] = torch.FloatTensor(adj_flat).reshape(subg_n,subg_n)
                node_count += subg_n
        all_adj = all_adj.to(self.get_device())
        in_categ = self._one_hot(sub_nodes_types,self.num_cat)
        in_numer = torch.FloatTensor(sub_nodes_feats).to(self.get_device()).unsqueeze(0).t()
        #print(in_categ)
        #print(in_numer)
        #print(all_adj)
        in_categ = self.catag_lin(in_categ)
        in_numer = self.numer_lin(in_numer)
        x = torch.cat([in_categ, in_numer], dim=1)
        inv_deg = inverse_adj(all_adj)
        #print(in_categ)
        #print(in_numer)
        #print(inv_deg)
        if self.dropout > 0.0001:
            for i in range(self.num_layer-1):
                x = self.layers[3 * i](x)
                x = x + torch.matmul(all_adj, x)
                x = torch.matmul(inv_deg, x)
                x = self.layers[3 * i + 1](x)
                x = self.layers[3 * i + 2](x)
            x = self.layers[3 * (i+1)](x)
            x = x + torch.matmul(all_adj, x)
            x = torch.matmul(inv_deg, x)
            x = self.layers[3 * (i+1) + 2](x)
        else:
            for i in range(self.num_layer-1):
                x = self.layers[2 * i](x)
                x = x + torch.matmul(all_adj, x)
                x = torch.matmul(inv_deg, x)
                x = self.layers[2 * i + 1](x)
                #x = self.layers[3 * i + 2](x)
            x = self.layers[2 * (i+1)](x)
            x = x + torch.matmul(all_adj, x)
            x = torch.matmul(inv_deg, x)
            x = self.layers[2 * (i+1) + 1](x)
        # readout phase
        #out = torch.zeros(num_nodes, self.emb_dim).to(self.get_device())
        out = x
        node_count = 0
        new_G = []
        for i in range(num_graphs):
            g = G[i].copy()
            for j in range(nodes_list[i]):
                subg_n = len(g.vs[j]['subg_ntypes'])
                subg_represent = out[node_count:node_count+subg_n, :]
                if self.readout == 'sum':
                    subg_feat = torch.sum(subg_represent, dim=0)
                elif self.readout == 'mean':
                    subg_feat = torch.mean(subg_represent, dim=0)
                else:
                    subg_feat = None
                    raise MyException('Undefined pool method')
                g.vs[j]['subg_feat'] = subg_feat
                node_count += subg_n
            new_G.append(g)
        return new_G
        
    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
    
    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x 


class CktGNN(nn.Module):
    # topology and node feature together.
    def __init__(self, max_n, nvt, subn_nvt ,START_TYPE, END_TYPE, max_pos=8, emb_dim = 16, feat_emb_dim = 8, hs=301, nz=56, bidirectional=False, pos=True, scale=True, scale_factor=102, topo_feat_scale = 0.01):
        super(CktGNN, self).__init__()
        self.max_n = max_n  # maximum number of vertices
        self.max_pos = max_pos + 1 # number of  positions in amp: 1 sudo + 7 positions
        self.scale = scale
        self.scale_factor = scale_factor
        self.nvt = nvt  # number of device types 
        self.subn_nvt = subn_nvt + 1 
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.emb_dim = emb_dim
        self.feat_emb_dim = feat_emb_dim # continuous feature embedding dimension
        self.hs = hs  # hidden state size of each vertex
        self.nz = nz  # size of latent representation z
        self.gs = hs + feat_emb_dim # size of graph state
        self.bidir = bidirectional  # whether to use bidirectional encoding
        self.pos = pos # whether to use the prior knowledge
        self.topo_feat_scale = topo_feat_scale # balance the attntion to topology information
        self.device = None
        
        #
        if self.pos:
            self.vs = hs + self.max_pos  # vertex state size = hidden state + vid
        else:
            self.vs = hs

        # 0. encoding-related
        self.df_enc = nn.Sequential(
                nn.Linear(self.max_pos * 3, emb_dim), 
                nn.ReLU(), 
                nn.Linear(emb_dim, feat_emb_dim)
                )  # subg features can be canonized according to the position of subg

        self.grue_forward = nn.GRUCell(nvt + self.max_pos, hs)  # encoder GRU
        self.grue_backward = nn.GRUCell(nvt + self.max_pos, hs)  # backward encoder GRU
        self.fc1 = nn.Linear(self.gs, nz)  # latent mean
        self.fc2 = nn.Linear(self.gs, nz)  # latent logvar
            
        # 1. decoding-related
        self.grud = nn.GRUCell(nvt + self.max_pos, hs)  # decoder GRU
        self.fc3 = nn.Linear(nz, hs)  # from latent z to initial hidden state h0
        self.add_vertex = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.ReLU(),
                nn.Linear(hs * 2, nvt)
                )  # which type of new subg to add 
        self.add_edge = nn.Sequential(
                nn.Linear(hs * 2 + self.max_pos * 2, hs * 4), 
                nn.ReLU(), 
                nn.Linear(hs * 4, 1)
                )  # whether to add edge between v_i and v_new
        self.add_pos = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.ReLU(),
                nn.Linear(hs * 2, self.max_pos)
                )  # which position of new subg to add 
        self.df_fc = nn.Sequential(
                nn.Linear(hs, 64),
                nn.ReLU(),
                nn.Linear(64,  self.max_pos * 3)
                ) # decode subg features
        
        # 2. gate-related
        self.gate_forward = nn.Sequential(
                nn.Linear(self.vs, hs), 
                nn.Sigmoid()
                )
        self.gate_backward = nn.Sequential(
                nn.Linear(self.vs, hs), 
                nn.Sigmoid()
                )
        self.mapper_forward = nn.Sequential(
                nn.Linear(self.vs, hs, bias=False),
                )  # disable bias to ensure padded zeros also mapped to zeros
        self.mapper_backward = nn.Sequential(
                nn.Linear(self.vs, hs, bias=False), 
                )

        # 3. bidir-related, to unify sizes
        if self.bidir:
            self.hv_unify = nn.Sequential(
                    nn.Linear(hs * 2, hs), 
                    )
            self.hg_unify = nn.Sequential(
                    nn.Linear(self.gs * 2, self.gs), 
                    )

        # 4. other
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax1 = nn.LogSoftmax(1)

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
    
    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device()) # get a zero hidden state

    def _get_zero_hidden(self, n=1, prior_edge=False):
        if prior_edge:
            return self._get_zeros(n, self.hs + self.max_pos) # get a zero hidden state
        else:
            return self._get_zeros(n, self.hs) # get a zero hidden state

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x

    def _gated(self, h, gate, mapper):
        return gate(h) * mapper(h)

    def _collate_fn(self, G):
        return [g.copy() for g in G]

    def _propagate_to(self, G, v, propagator, H=None, reverse=False, decode=False):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return
        if H is not None: # H: previous hidden state 
            idx = [i for i, g in enumerate(G) if g.vcount() > v]
            H = H[idx]
        v_types = [g.vs[v]['type'] for g in G]
        pos_feats = [g.vs[v]['pos'] for g in G]
        X_v_ = self._one_hot(v_types, self.nvt)
        X_pos_ = self._one_hot(pos_feats, self.max_pos)

        X = torch.cat([X_v_, X_pos_], dim=1)
        
        if reverse:
            H_name = 'H_backward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.successors(v)] for g in G] # hidden state of 'predecessors'
            if self.pos:
                pos_ = [self._one_hot([g.vs[v_]['pos'] for v_ in g.successors(v)], self.max_pos) for g in G] # one hot of vertex index of 'predecessors', pos_=vids
            gate, mapper = self.gate_backward, self.mapper_backward
        else:
            H_name = 'H_forward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
            if self.pos:
                pos_ = [self._one_hot([g.vs[x]['pos'] for x in g.predecessors(v)], self.max_pos) for g in G] # one hot of vertex index of 'predecessors', pos_=vids
            gate, mapper = self.gate_forward, self.mapper_forward
        if self.pos:
            H_pred = [[torch.cat([x[i], y[i:i+1]], 1) for i in range(len(x))] for x, y in zip(H_pred, pos_)]
        if H is None:
            max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
            if max_n_pred == 0: ### start point
                H = self._get_zero_hidden(len(G))
            else:
                H_pred = [torch.cat(h_pred + 
                            [self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0) 
                            for h_pred in H_pred]  # pad all to same length
                H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * vs
                H = self._gated(H_pred, gate, mapper).sum(1)  # batch * hs
        Hv = propagator(X, H)
        for i, g in enumerate(G):
            g.vs[v][H_name] = Hv[i:i+1]
        return Hv

    def _propagate_from(self, G, v, propagator, H0=None, reverse=False, decode=False):
        # perform a series of propagation_to steps starting from v following a topo order
        # assume the original vertex indices are in a topological order
        if reverse:
            prop_order = range(v, -1, -1)
        else:
            prop_order = range(v, self.max_n)
        Hv = self._propagate_to(G, v, propagator, H0, reverse=reverse, decode=decode)  # the initial vertex
        for v_ in prop_order[1:]:
            #print(v_)
            self._propagate_to(G, v_, propagator, reverse=reverse, decode=decode)
            # Hv = self._propagate_to(G, v_, propagator, Hv, reverse=reverse) no need
        return Hv

    def _update_v(self, G, v, H0=None, decode=False):
        # perform a forward propagation step at v when decoding to update v's state
        self._propagate_to(G, v, self.grud, H0, reverse=False, decode=decode)
        return
    
    def _get_vertex_state(self, G, v, prior_edge=False):
        # get the vertex states at v
        Hv = []
        for g in G:
            if v >= g.vcount():
                hv = self._get_zero_hidden(prior_edge=prior_edge)
            else:
                hv = g.vs[v]['H_forward']
                if prior_edge:
                    pos_ = self._one_hot([g.vs[v]['pos']], self.max_pos)
                    hv =  torch.cat([hv, pos_], 1)
            Hv.append(hv)
        Hv = torch.cat(Hv, 0)
        return Hv

    def _get_graph_state(self, G, decode=False):
        # get the graph states
        # when decoding, use the last generated vertex's state as the graph state
        # when encoding, use the ending vertex state or unify the starting and ending vertex states
        Hg = []
        for g in G:
            hg = g.vs[g.vcount()-1]['H_forward']
            if self.bidir and not decode:  # decoding never uses backward propagation
                hg_b = g.vs[0]['H_backward']
                hg = torch.cat([hg, hg_b], 1)
            Hg.append(hg)
        Hg = torch.cat(Hg, 0)
        if self.bidir and not decode:
            Hg = self.hg_unify(Hg) # a linear model
        return Hg

    def encode(self, G):
        # encode graphs G into latent vectors
        if type(G) != list:
            G = [G]
        self._propagate_from(G, 0, self.grue_forward, H0=self._get_zero_hidden(len(G)),
                             reverse=False, decode=False)
        if self.bidir:
            self._propagate_from(G, self.max_n-1, self.grue_backward, 
                                 H0=self._get_zero_hidden(len(G)), reverse=True, decode=False)
        Hg = self._get_graph_state(G)
        #print(Hg.shape)
        
        dfs_ = []
        for g in G:
            df_ = [0] * (3 * self.max_pos)
            for v_ in range(len(g.vs)):
                pos_ = g.vs[v_]['pos']
                df_[pos_ * 3 + 0] = g.vs[v_]['r']
                df_[pos_ * 3 + 1] = g.vs[v_]['c']
                df_[pos_ * 3 + 2] = g.vs[v_]['gm']
            dfs_.append(df_)
        Hdf = torch.FloatTensor(dfs_).to(self.get_device())
        Hd = self.df_enc(Hdf)
        Hg = torch.cat([Hg, Hd], dim=1) #  concatenate the topology embedding and subg feature embedding

        mu, logvar = self.fc1(Hg), self.fc2(Hg) 
        return mu, logvar

    def reparameterize(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def _get_edge_score(self, Hvi, H, H0):
        # compute scores for edges from vi based on Hvi, H (current vertex) and H0
        # in most cases, H0 need not be explicitly included since Hvi and H contain its information
        return self.sigmoid(self.add_edge(torch.cat([Hvi, H], -1)))

    def decode(self, z, stochastic=True, node_type_dic=NODE_TYPE, subg_node=SUBG_NODE, subg_con=SUBG_CON, subg_indi=SUBG_INDI):
        # decode latent vectors z back to graphs
        # if stochastic=True, stochastically sample each action from the predicted distribution;
        # otherwise, select argmax action deterministically.
        H0 = self.tanh(self.fc3(z))  # or relu activation, similar performance
        pred_dfs = self.df_fc(H0)

        G = [igraph.Graph(directed=True) for _ in range(len(z))]
        for g in G:
            g.add_vertex(type=self.START_TYPE)
            g.vs[0]['r'] = 0.0
            g.vs[0]['c'] = 0.0
            g.vs[0]['gm'] = 0.0
            g.vs[0]['pos'] = 0
        self._update_v(G, 0, H0, decode=True) # only at the 'begining', we need a hidden state H0
        finished = [False] * len(G)
        for idx in range(1, self.max_n):
            # decide the type of the next added vertex
            if idx == self.max_n - 1:  # force the last node to be end_type
                new_types = [self.END_TYPE] * len(G)
            else:
                Hg = self._get_graph_state(G, decode=True)
                type_scores = self.add_vertex(Hg)
                pos_scores = self.add_pos(Hg)
                #pred_dfs = self.df_fc(Hg)
                if stochastic:
                    type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
                    pos_probs = F.softmax(pos_scores, 1).cpu().detach().numpy() 
                    new_types = [np.random.choice(range(self.nvt), p=type_probs[i]) 
                                 for i in range(len(G))]
                    new_pos = [np.random.choice(range(self.max_pos), p=pos_probs[i]) 
                                 for i in range(len(G))]
                else:
                    new_types = torch.argmax(type_scores, 1)
                    new_types = new_types.flatten().tolist()
                    new_pos = torch.argmax(pos_scores, 1)
                    new_pos = new_pos.flatten().tolist()

            for j,g in enumerate(G):
                if not finished[j]:
                    g.add_vertex(type=new_types[j])
                    g.vs[idx]['pos'] = new_pos[j]
                    g.vs[idx]['r'] = pred_dfs[j, new_pos[j] * 3 + 0]
                    g.vs[idx]['c'] = pred_dfs[j, new_pos[j] * 3 + 1]
                    g.vs[idx]['gm'] = pred_dfs[j, new_pos[j] * 3 + 2]
            
            self._update_v(G, idx,decode=True)
            # decide connections
            edge_scores = []
            for vi in range(idx-1, -1, -1):
                Hvi = self._get_vertex_state(G, vi, prior_edge=True)
                H = self._get_vertex_state(G, idx, prior_edge=True)
                ei_score = self._get_edge_score(Hvi, H, H0)
                if stochastic:
                    random_score = torch.rand_like(ei_score)
                    decisions = random_score < ei_score
                else:
                    decisions = ei_score > 0.5
                for i, g in enumerate(G):
                    if finished[i]:
                        continue
                    if new_types[i] == self.END_TYPE: 
                    # if new node is end_type, connect it to all loose-end vertices (out_degree==0)
                        end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0) 
                                            if v.index != g.vcount()-1])
                        for v in end_vertices:
                            g.add_edge(v, g.vcount()-1)
                        finished[i] = True
                        continue
                    if decisions[i, 0]:
                        g.add_edge(vi, g.vcount()-1)
                self._update_v(G, idx, decode=True)
        

        for g in G:
            del g.vs['H_forward']  # delete hidden states to save GPU memory

        return G

    def loss(self, mu, logvar, G_true, beta=0.005, reg_scale=0.05, pos_scale=0.5):
        # compute the loss of decoding mu and logvar to true graphs using teacher forcing
        # ensure when computing the loss of step i, steps 0 to i-1 are correct
        dfs_ = []
        for g in G_true:
            df_ = [0] * (3 * self.max_pos)
            for v_ in range(len(g.vs)):
                pos_ = g.vs[v_]['pos']
                df_[pos_ * 3 + 0] = g.vs[v_]['r']
                df_[pos_ * 3 + 1] = g.vs[v_]['c']
                df_[pos_ * 3 + 2] = g.vs[v_]['gm']
            dfs_.append(df_)
        true_dfs = torch.FloatTensor(dfs_).to(self.get_device())

        z = self.reparameterize(mu, logvar) # (bsize, hidden)
        H0 = self.tanh(self.fc3(z))  # or relu activation, similar performance
        pred_dfs = self.df_fc(H0)


        G = [igraph.Graph(directed=True) for _ in range(len(z))]
        for g in G:
            g.add_vertex(type=self.START_TYPE)
            g.vs[0]['r'] = 0.0
            g.vs[0]['c'] = 0.0
            g.vs[0]['gm'] = 0.0
            g.vs[0]['pos'] = 0
        self._update_v(G, 0, H0)

        res = 0  # log likelihood
        res_vl1 = 0
        res_vl3 = 0
        for v_true in range(1, self.max_n):
            # calculate the likelihood of adding true types of nodes
            # use start type to denote padding vertices since start type only appears for vertex 0 
            # and will never be a true type for later vertices, thus it's free to use
            true_types = [g_true.vs[v_true]['type'] if v_true < g_true.vcount()  # (bsize, 1)
                          else self.START_TYPE for g_true in G_true]
            true_pos = [g_true.vs[v_true]['pos'] if v_true < g_true.vcount()  # (bsize, 1)
                          else self.max_pos-1 for g_true in G_true]
            Hg = self._get_graph_state(G, decode=True) 
            
            type_scores = self.add_vertex(Hg) # (bsize, self.vrt)
            pos_scores = self.add_pos(Hg)
            # vertex log likelihood
            vll = self.logsoftmax1(type_scores)[np.arange(len(G)), true_types].sum() 
            vl3 = self.logsoftmax1(pos_scores)[np.arange(len(G)), true_pos].sum() 
            res = res + vll + vl3
            
            res_vl1 += vll
            res_vl3 += vl3

            for i, g in enumerate(G):
                if true_types[i] != self.START_TYPE:
                    g.add_vertex(type=true_types[i])
                    g.vs[v_true]['r'] = G_true[i].vs[v_true]['r']
                    g.vs[v_true]['c'] = G_true[i].vs[v_true]['c']
                    g.vs[v_true]['gm'] = G_true[i].vs[v_true]['gm']
                    g.vs[v_true]['pos'] = G_true[i].vs[v_true]['pos']
            self._update_v(G, v_true,decode=True)
            # calculate the mse loss of asubg nodes value
            H = self._get_vertex_state(G, v_true)
            true_edges = []
            for i, g_true in enumerate(G_true):
                true_edges.append(g_true.get_adjlist(igraph.IN)[v_true] if v_true < g_true.vcount() 
                                  else []) # get_idjlist: return a list of node index to show these directed edges. true_edges[i] = in ith graph, v_true's predecessors
            edge_scores = []
            for vi in range(v_true-1, -1, -1):
                Hvi = self._get_vertex_state(G, vi, prior_edge=True)
                H = self._get_vertex_state(G, v_true, prior_edge=True)
                ei_score = self._get_edge_score(Hvi, H, H0) # size: batch size, 1
                edge_scores.append(ei_score)
                for i, g in enumerate(G):
                    if vi in true_edges[i]:
                        g.add_edge(vi, v_true)
                self._update_v(G, v_true, decode=True)
            edge_scores = torch.cat(edge_scores[::-1], 1)  # (batch size, v_true): columns: v_true-1, ... 0

            ground_truth = torch.zeros_like(edge_scores)
            idx1 = [i for i, x in enumerate(true_edges) for _ in range(len(x))]
            idx2 = [xx for x in true_edges for xx in x]
            ground_truth[idx1, idx2] = 1.0

            # edges log-likelihood
            ell = - F.binary_cross_entropy(edge_scores, ground_truth, reduction='sum') 
            res = res + ell


        res_vl2 = self.topo_feat_scale * F.mse_loss(pred_dfs, true_dfs, reduction='sum')/300 # each subg node has 3 subg features, which scan be normalized by divide 100.
        res1 = -res  # convert likelihood to loss

        res_vl1 = -res_vl1
        res_vl3 = -res_vl3
        res = res1 + res_vl2
        #res += res_mse
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta*kld, res1, kld, res_vl1, res_vl3, res_vl2 

    def encode_decode(self, G):
        mu, logvar = self.encode(G)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def forward(self, G):
        mu, logvar = self.encode(G)
        loss, _, _ = self.loss(mu, logvar, G)
        return loss
    
    def generate_sample(self, n):
        sample = torch.randn(n, self.nz).to(self.get_device())
        G = self.decode(sample)
        return G


# Other baselines

class DVAE(nn.Module):
    def __init__(self, max_n, nvt, feat_nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False, vid=True, max_pos=8, topo_feat_scale=0.01):
        super(DVAE, self).__init__()
        self.max_n = max_n  # maximum number of vertices
        #self.max_pos = max_pos
        self.max_pos = feat_nvt + 1
        self.nvt = nvt  # number of vertex types 
        self.feat_nvt = feat_nvt + 1 # number of value type of each node in subgraphs
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.hs = hs  # hidden state size of each vertex
        self.nz = nz  # size of latent representation z
        self.gs = hs  # size of graph state
        self.bidir = bidirectional  # whether to use bidirectional encoding
        self.vid = vid
        self.topo_feat_scale = topo_feat_scale
        self.device = None
        
        self.vs = hs 
        if self.vid:
            self.nvt += self.max_n

        # 0. encoding-related
        self.grue_forward = nn.GRUCell(self.nvt + 1, hs)  # encoder GRU
        self.grue_backward = nn.GRUCell(self.nvt + 1, hs)  # backward encoder GRU
        self.fc1 = nn.Linear(self.gs, nz)  # latent mean
        self.fc2 = nn.Linear(self.gs, nz)  # latent logvar
            
        # 1. decoding-related
        self.grud = nn.GRUCell(self.nvt + 1, hs)  # decoder GRU
        self.fc3 = nn.Linear(nz, hs)  # from latent z to initial hidden state h0
        self.add_vertex = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.ReLU(),
                nn.Linear(hs * 2, nvt)
                )  # which type of new vertex to add f(h0, hg)
        self.add_edge = nn.Sequential(
                nn.Linear(hs * 2, hs * 4), 
                nn.ReLU(), 
                nn.Linear(hs * 4, 1)
                )  # whether to add edge between v_i and v_new, f(hvi, hnew)
        self.fc_feat = nn.Sequential(
                nn.Linear(hs, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
                ) 
        # 2. gate-related
        self.gate_forward = nn.Sequential(
                nn.Linear(self.vs, hs), 
                nn.Sigmoid()
                )
        self.gate_backward = nn.Sequential(
                nn.Linear(self.vs, hs), 
                nn.Sigmoid()
                )
        self.mapper_forward = nn.Sequential(
                nn.Linear(self.vs, hs, bias=False),
                )  # disable bias to ensure padded zeros also mapped to zeros
        self.mapper_backward = nn.Sequential(
                nn.Linear(self.vs, hs, bias=False), 
                )

        # 3. bidir-related, to unify sizes
        if self.bidir:
            self.hv_unify = nn.Sequential(
                    nn.Linear(hs * 2, hs), 
                    )
            self.hg_unify = nn.Sequential(
                    nn.Linear(self.gs * 2, self.gs), 
                    )

        # 4. other
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax1 = nn.LogSoftmax(1)

    def get_device(self):
        if self.device is None:
            self.device = next(self.parameters()).device
        return self.device
    
    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device()) # get a zero hidden state

    def _get_zero_hidden(self, n=1):
        return self._get_zeros(n, self.hs) # get a zero hidden state

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x

    def _gated(self, h, gate, mapper):
        return gate(h) * mapper(h)

    def _collate_fn(self, G):
        return [g.copy() for g in G]

    def _propagate_to(self, G, v, propagator, H=None, reverse=False):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return
        if H is not None: # H: previous hidden state 
            idx = [i for i, g in enumerate(G) if g.vcount() > v]
            H = H[idx]
        v_types = [g.vs[v]['type'] for g in G]
        v_feats = [g.vs[v]['feat'] for g in G]
        X = self._one_hot(v_types, self.nvt)
        Y = torch.FloatTensor(v_feats).view(-1,1).to(self.get_device())
        X = torch.cat([X,Y],dim=1)
        if reverse:
            H_name = 'H_backward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.successors(v)] for g in G] # hidden state of 'predecessors'
            if self.vid:
                vids = [self._one_hot(g.successors(v), self.max_n) for g in G] # one hot of vertex index of 'predecessors'
            gate, mapper = self.gate_backward, self.mapper_backward
        else:
            H_name = 'H_forward'  # name of the hidden states attribute
            H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
            if self.vid:
                vids = [self._one_hot(g.predecessors(v), self.max_n) for g in G]
            gate, mapper = self.gate_forward, self.mapper_forward
        if self.vid:
            H_pred = [[torch.cat([x[i], y[i:i+1]], 1) for i in range(len(x))] for x, y in zip(H_pred, vids)]
        # if h is not provided, use gated sum of v's predecessors' states as the input hidden state
        if H is None:
            max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
            if max_n_pred == 0: ### start point
                H = self._get_zero_hidden(len(G))
            else:
                H_pred = [torch.cat(h_pred + 
                            [self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0) 
                            for h_pred in H_pred]  # pad all to same length
                H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * vs
                H = self._gated(H_pred, gate, mapper).sum(1)  # batch * hs
        Hv = propagator(X, H)
        for i, g in enumerate(G):
            g.vs[v][H_name] = Hv[i:i+1]
        return Hv

    def _propagate_from(self, G, v, propagator, H0=None, reverse=False):
        # perform a series of propagation_to steps starting from v following a topo order
        # assume the original vertex indices are in a topological order
        if reverse:
            prop_order = range(v, -1, -1)
        else:
            prop_order = range(v, self.max_n)
        Hv = self._propagate_to(G, v, propagator, H0, reverse=reverse)  # the initial vertex
        for v_ in prop_order[1:]:
            #print(v_)
            self._propagate_to(G, v_, propagator, reverse=reverse)
            # Hv = self._propagate_to(G, v_, propagator, Hv, reverse=reverse) no need
        return Hv

    def _update_v(self, G, v, H0=None):
        # perform a forward propagation step at v when decoding to update v's state
        self._propagate_to(G, v, self.grud, H0, reverse=False)
        return
    
    def _get_vertex_state(self, G, v):
        # get the vertex states at v
        Hv = []
        for g in G:
            if v >= g.vcount():
                hv = self._get_zero_hidden()
            else:
                hv = g.vs[v]['H_forward']
            Hv.append(hv)
        Hv = torch.cat(Hv, 0)
        return Hv

    def _get_graph_state(self, G, decode=False):
        # get the graph states
        # when decoding, use the last generated vertex's state as the graph state
        # when encoding, use the ending vertex state or unify the starting and ending vertex states
        Hg = []
        for g in G:
            hg = g.vs[g.vcount()-1]['H_forward']
            if self.bidir and not decode:  # decoding never uses backward propagation
                hg_b = g.vs[0]['H_backward']
                hg = torch.cat([hg, hg_b], 1)
            Hg.append(hg)
        Hg = torch.cat(Hg, 0)
        if self.bidir and not decode:
            Hg = self.hg_unify(Hg) # a linear model
        return Hg

    def encode(self, G):
        # encode graphs G into latent vectors
        if type(G) != list:
            G = [G]
        self._propagate_from(G, 0, self.grue_forward, H0=self._get_zero_hidden(len(G)),
                             reverse=False)
        if self.bidir:
            self._propagate_from(G, self.max_n-1, self.grue_backward, 
                                 H0=self._get_zero_hidden(len(G)), reverse=True)
        Hg = self._get_graph_state(G)
        mu, logvar = self.fc1(Hg), self.fc2(Hg) 
        return mu, logvar

    def reparameterize(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def _get_edge_score(self, Hvi, H, H0):
        # compute scores for edges from vi based on Hvi, H (current vertex) and H0
        # in most cases, H0 need not be explicitly included since Hvi and H contain its information
        return self.sigmoid(self.add_edge(torch.cat([Hvi, H], -1)))

    def decode(self, z, stochastic=True):
        # decode latent vectors z back to graphs
        # if stochastic=True, stochastically sample each action from the predicted distribution;
        # otherwise, select argmax action deterministically.
        H0 = self.tanh(self.fc3(z))  # or relu activation, similar performance
        G = [igraph.Graph(directed=True) for _ in range(len(z))]
        for g in G:
            g.add_vertex(type=self.START_TYPE)
            g.vs[0]['feat'] = 0
        self._update_v(G, 0, H0) # only at the 'begining', we need a hidden state H0
        finished = [False] * len(G)
        for idx in range(1, self.max_n):
            # decide the type of the next added vertex
            if idx == self.max_n - 1:  # force the last node to be end_type
                new_types = [self.END_TYPE] * len(G)
                new_feats = [0] * len(G)
            else:
                Hg = self._get_graph_state(G, decode=True)
                type_scores = self.add_vertex(Hg)
                feat_pred = self.fc_feat(Hg)
                #vid_scores = self.vid_fc(Hg)
                if stochastic:
                    type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
                    new_types = [np.random.choice(range(self.nvt), p=type_probs[i]) 
                                 for i in range(len(G))]
                else:
                    new_types = torch.argmax(type_scores, 1)
                    new_types = new_types.flatten().tolist()
            
            for i, g in enumerate(G):
                if not finished[i]:
                    g.add_vertex(type=new_types[i])
                    g.vs[idx]['feat'] = feat_pred[i]
                    #g.vs[idx]['vid'] = new_vids[i]
            self._update_v(G, idx)

            # decide connections
            edge_scores = []
            for vi in range(idx-1, -1, -1):
                Hvi = self._get_vertex_state(G, vi)
                H = self._get_vertex_state(G, idx)
                ei_score = self._get_edge_score(Hvi, H, H0)
                if stochastic:
                    random_score = torch.rand_like(ei_score)
                    decisions = random_score < ei_score
                else:
                    decisions = ei_score > 0.5
                for i, g in enumerate(G):
                    if finished[i]:
                        continue
                    if new_types[i] == self.END_TYPE: 
                    # if new node is end_type, connect it to all loose-end vertices (out_degree==0)
                        end_vertices = set([v.index for v in g.vs.select(_outdegree_eq=0) 
                                            if v.index != g.vcount()-1])
                        for v in end_vertices:
                            g.add_edge(v, g.vcount()-1)
                        finished[i] = True
                        continue
                    if decisions[i, 0]:
                        g.add_edge(vi, g.vcount()-1)
                self._update_v(G, idx)

        for g in G:
            del g.vs['H_forward']  # delete hidden states to save GPU memory
        return G

    def loss(self, mu, logvar, G_true, beta=0.005):
        # compute the loss of decoding mu and logvar to true graphs using teacher forcing
        # ensure when computing the loss of step i, steps 0 to i-1 are correct
        z = self.reparameterize(mu, logvar) # (bsize, hidden)
        H0 = self.tanh(self.fc3(z))  # or relu activation, similar performance
        G = [igraph.Graph(directed=True) for _ in range(len(z))]
        for g in G:
            g.add_vertex(type=self.START_TYPE)
            g.vs[0]['feat'] = 0
            #g.vs[0]['vid'] = 0
        self._update_v(G, 0, H0)
        res = 0  # log likelihood
        res_vll= 0
        res_vl2 = 0
        for v_true in range(1, self.max_n):

            true_types = [g_true.vs[v_true]['type'] if v_true < g_true.vcount()  # (bsize, 1)
                          else self.START_TYPE for g_true in G_true]
            true_feats = [g_true.vs[v_true]['feat'] if v_true < g_true.vcount()  # (bsize, 1)
                          else 0 for g_true in G_true]
        
            Hg = self._get_graph_state(G, decode=True) 
            type_scores = self.add_vertex(Hg) # (bsize, self.vrt)
            #feat_scores = self.fc_feat(Hg)
            pred_feats = self.fc_feat(Hg)
            true_feads = torch.FloatTensor(true_feats).view(-1,1).to(self.get_device())
            vll = self.logsoftmax1(type_scores)[np.arange(len(G)), true_types].sum()  
            vl2  = self.topo_feat_scale * F.mse_loss(pred_feats, true_feads, reduction='sum')/100 # to normlaize feat
             
            res = res + vll - vl2
            res_vll += vll
            res_vl2 += vl2
            for i, g in enumerate(G):
                if true_types[i] != self.START_TYPE:
                    g.add_vertex(type=true_types[i])
                    g.vs[v_true]['feat'] = true_feats[i]
                    #g.vs[v_true]['vid'] = true_vids[i]
            #print(g.vs[1])
            self._update_v(G, v_true)

            # calculate the likelihood of adding true edges
            true_edges = []
            for i, g_true in enumerate(G_true):
                true_edges.append(g_true.get_adjlist(igraph.IN)[v_true] if v_true < g_true.vcount() 
                                  else []) # get_idjlist: return a list of node index to show these directed edges. true_edges[i] = in ith graph, v_true's predecessors
            edge_scores = []
            for vi in range(v_true-1, -1, -1):
                Hvi = self._get_vertex_state(G, vi)
                H = self._get_vertex_state(G, v_true)
                ei_score = self._get_edge_score(Hvi, H, H0) # size: batch size, 1
                edge_scores.append(ei_score)
                for i, g in enumerate(G):
                    if vi in true_edges[i]:
                        g.add_edge(vi, v_true)
                self._update_v(G, v_true)
            edge_scores = torch.cat(edge_scores[::-1], 1)  # (batch size, v_true): columns: v_true-1, ... 0

            ground_truth = torch.zeros_like(edge_scores)
            idx1 = [i for i, x in enumerate(true_edges) for _ in range(len(x))]
            idx2 = [xx for x in true_edges for xx in x]
            ground_truth[idx1, idx2] = 1.0

            # edges log-likelihood
            ell = - F.binary_cross_entropy(edge_scores, ground_truth, reduction='sum') 
            res = res + ell

        res = -res  # convert likelihood to loss
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta*kld, res, kld, res_vll, 0, res_vl2

    def encode_decode(self, G):
        mu, logvar = self.encode(G)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def forward(self, G):
        mu, logvar = self.encode(G)
        loss, _, _ = self.loss(mu, logvar, G)
        return loss
    
    def generate_sample(self, n):
        sample = torch.randn(n, self.nz).to(self.get_device())
        G = self.decode(sample)
        return G



class DVAE_GCN(DVAE):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, hs=501, nz=56, bidirectional=False, levels=3):
        # bidirectional means passing messages ignoring edge directions
        super(DVAE_GCN, self).__init__(max_n, nvt, START_TYPE, END_TYPE, hs, nz, bidirectional)
        self.levels = levels
        self.gconv = nn.ModuleList()
        self.gconv.append(
                nn.Sequential(
                    nn.Linear(nvt, hs), 
                    nn.ReLU(), 
                    )
                )
        for lv in range(1, levels):
            self.gconv.append(
                    nn.Sequential(
                        nn.Linear(hs, hs), 
                        nn.ReLU(), 
                        )
                    )

    def _get_feature(self, g, v, lv=0):
        # get the node feature vector of v
        if lv == 0:  # initial level uses type features
            v_type = g.vs[v]['type']
            x = self._one_hot(v_type, self.nvt)
        else:
            x = g.vs[v]['H_forward']
        return x

    def _get_zero_x(self, n=1):
        # get zero predecessor states X, used for padding
        return torch.zeros(n, self.nvt).to(self.get_device())

    def _get_graph_state(self, G, decode=False, start=0, end_offset=0):
        # get the graph states
        # sum all node states between start and n-end_offset as the graph state
        Hg = []
        max_n_nodes = max(g.vcount() for g in G)
        for g in G:
            hg = torch.cat([g.vs[i]['H_forward'] for i in range(start, g.vcount() - end_offset)],
                           0).unsqueeze(0)  # 1 * n * hs
            if g.vcount() < max_n_nodes:
                hg = torch.cat([hg, 
                    torch.zeros(1, max_n_nodes - g.vcount(), hg.shape[2]).to(self.get_device())],
                    1)  # 1 * max_n * hs
            Hg.append(hg)
        # sum node states as the graph state
        Hg = torch.cat(Hg, 0).sum(1)  # batch * hs
        return Hg  # batch * hs

    def _GCN_propagate_to(self, G, v, lv=0):
        # propagate messages to vertex index v for all graphs in G
        # return the new messages (states) at v
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return

        if self.bidir:  # ignore edge directions, accept all neighbors' messages
            H_nei = [[self._get_feature(g, v, lv)/(g.degree(v)+1)] + 
                     [self._get_feature(g, x, lv)/math.sqrt((g.degree(x)+1)*(g.degree(v)+1)) 
                     for x in g.neighbors(v)] for g in G]
        else:  # only accept messages from predecessors (generalizing GCN to directed cases)
            H_nei = [[self._get_feature(g, v, lv)/(g.indegree(v)+1)] + 
                     [self._get_feature(g, x, lv)/math.sqrt((g.outdegree(x)+1)*(g.indegree(v)+1)) 
                     for x in g.predecessors(v)] for g in G]
            
        max_n_nei = max([len(x) for x in H_nei])  # maximum number of neighbors
        H_nei = [torch.cat(h_nei + [self._get_zeros(max_n_nei - len(h_nei), h_nei[0].shape[1])], 0).unsqueeze(0) 
                 for h_nei in H_nei]  # pad all to same length
        H_nei = torch.cat(H_nei, 0)  # batch * max_n_nei * nvt
        Hv = self.gconv[lv](H_nei.sum(1))  # batch * hs
        for i, g in enumerate(G):
            g.vs[v]['H_forward'] = Hv[i:i+1]
        return Hv

    def encode(self, G):
        # encode graphs G into latent vectors
        # GCN propagation is now implemented in a non-parallel way for consistency, but
        # can definitely be parallel to speed it up. However, the major computation cost
        # comes from the generation, which is not parallellizable.
        if type(G) != list:
            G = [G]
        prop_order = range(self.max_n)
        for lv in range(self.levels):
            for v_ in prop_order:
                self._GCN_propagate_to(G, v_, lv)
        Hg = self._get_graph_state(G, start=1, end_offset=1)  # does not use the dummy input 
                                                              # and output nodes
        mu, logvar = self.fc1(Hg), self.fc2(Hg) 
        return mu, logvar



