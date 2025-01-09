import dgl.sampling
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GraphConv
import copy
import random
from sklearn.neighbors import kneighbors_graph
from scipy import sparse


sigma = 1e-6
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
random.seed(1024)
  
torch.cuda.manual_seed_all(1024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

from ChebnetII_pro import ChebnetII_prop
from torch.nn import Linear
class ChebNetII(torch.nn.Module):
    def __init__(self, num_features, hidden=512, K=10, dprate=0.50, dropout=0.50, is_bns=True, act_fn='relu'):
        super(ChebNetII, self).__init__()
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, hidden)
        self.prop1 = ChebnetII_prop(K=K)
        assert act_fn in ['relu', 'prelu']
        self.act_fn = nn.PReLU() if act_fn == 'prelu' else nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(num_features, momentum=0.01)
        self.is_bns = is_bns
        self.dprate = dprate
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, highpass=True):

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index, highpass=highpass)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index, highpass=highpass)


        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.is_bns:
            x = self.bn(x)

        x = self.lin1(x)
        x = self.act_fn(x)

        return x

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)

        x = self.act_fn(x)
        x = self.layer2(x)

        return x

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, use_ln=False):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))
        self.use_ln = use_ln
        self.lns = nn.ModuleList()

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            for i in range(n_layers - 1):
                self.lns.append(nn.BatchNorm1d(hid_dim))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x):
        for i in range(self.n_layers - 1):
            if not self.use_ln:
                x = F.relu(self.convs[i](graph, x))
            else:
                x = F.relu(self.lns[i](self.convs[i](graph, x)))

        x = self.convs[-1](graph, x)

        return x



class MLP_generator(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super(MLP_generator, self).__init__()
        self.linears = torch.nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, output_dim))
        for layer in range(num_layers - 1):
            self.linears.append(nn.Linear(output_dim, output_dim))
        self.num_layers = num_layers


    def forward(self, embedding):
        h = embedding
        for layer in range(self.num_layers - 1):
            h = F.relu(self.linears[layer](h))
        neighbor_embedding = self.linears[self.num_layers - 1](h)
        return neighbor_embedding




class ConditionalSelection(nn.Module):
    def __init__(self, in_dim, h_dim):
        super(ConditionalSelection, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dim, h_dim * 2),
            nn.LayerNorm([h_dim * 2]),
            nn.ReLU(),
        )

    def forward(self, x, tau=1, hard=False):
        shape = x.shape
        x = self.fc(x)
        x = x.view(shape[0], 2, -1)
        x = F.gumbel_softmax(x, dim=1, tau=tau, hard=hard)
        return x[:, 0, :], x[:, 1, :]
class Gate(nn.Module):
    def __init__(self,hidden) -> None:
        super().__init__()

        self.cs = ConditionalSelection(hidden,hidden)
        self.context = 0
        self.pm = []
        self.gm = []
        self.pm_ = []
        self.gm_ = []

    def forward(self, rep, context=None, tau=1, hard=False):


        gate_in = rep

        if context != None:
            context = F.normalize(context, p=2, dim=1)
            self.context = torch.tile(context, (rep.shape[0], 1))

        if self.context != None:
            gate_in = rep * self.context

        pm, gm = self.cs(gate_in, tau=tau, hard=hard)

        rep_p = rep * pm
        rep_g = rep * gm
        return rep_p, rep_g

    def set_context(self, head):
        headw_p = head.weight.data.clone()
        headw_p.detach_()
        self.context = torch.sum(headw_p, dim=0, keepdim=True)


def extract(mask,b):

    indices = mask.indices()
    values = mask.values()


    b_tensor = torch.tensor(b).to(indices.device)

    mask_row = torch.isin(indices[0], b_tensor)
    mask_col = torch.isin(indices[1], b_tensor)
    mask = mask_row & mask_col

    new_indices = indices[:, mask]

    for i, idx in enumerate(b):
        new_indices[0][new_indices[0] == idx] = i

        new_indices[1][new_indices[1] == idx] = i
    new_values = values[mask]

    sub_pos_mask = torch.sparse_coo_tensor(new_indices, new_values, torch.Size([len(b), len(b)])).to_dense()
    sub_pos_mask.fill_diagonal_(1)

    return sub_pos_mask


class Model_our(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, num_layers, temp, use_mlp=False,  num_MLP=1,
                 gamma=0.5,k=6):
        super(Model_our, self).__init__()
        if use_mlp:
            self.encoder = MLP(in_dim, hid_dim, out_dim, use_bn=True)
        else:
            self.encoder = GCN(in_dim, hid_dim, out_dim, num_layers)
            # self.encoder = MLP(in_dim, hid_dim, out_dim, use_bn=True)
            self.encoder_target = MLP(in_dim, hid_dim, out_dim, use_bn=True)

        self.temp = temp
        self.out_dim = out_dim
        self.gamma =  gamma
        self.num_MLP = num_MLP
        if num_MLP != 0:
            self.projector = MLP_generator(hid_dim, hid_dim, num_MLP)
            self.projector1 = MLP_generator(hid_dim, hid_dim, num_MLP)
        self.conv1 = ChebNetII(in_dim, hid_dim, k)
        self.conv2 = ChebNetII(in_dim, hid_dim, k)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def get_embedding(self, graph, feat):
        trans_feature = self.encoder_target(graph, feat)
        return trans_feature.detach()


    def set_mask_knn(self, X, edges, k,  device, dataset,metric='cosine'):
        import os
        if k != 0:
            path = '../data/knn/{}'.format(dataset)
            if not os.path.exists(path):
                os.makedirs(path)
            file_name = path + '/{}_{}.npz'.format(dataset, k)
            if os.path.exists(file_name):
                knn = sparse.load_npz(file_name)
                # print('Load exist knn graph.')
            else:
                print('Computing knn graph...')
                knn = kneighbors_graph(X, k, metric=metric)
                sparse.save_npz(file_name, knn)
                print('Done. The knn graph is saved as: {}.'.format(file_name))
            knn = (torch.tensor(knn.toarray()) + torch.eye(X.shape[0])).to_sparse()
        else:
            knn = torch.eye(X.shape[0])
        import scipy.sparse as sp
        self.pos_mask = knn.to(device)
        # self.neg_mask = (1 - self.pos_mask).to(device)

        from torch_sparse import SparseTensor
        values = torch.ones(edges.shape[1], dtype=torch.float64).to(device)
        # Define the size of the sparse matrix
        size = (X.shape[0],X.shape[0])
        # Create the sparse COO tensor
        sparse_tensor = torch.sparse_coo_tensor(indices=edges, values=values, size=size)

        self.dense_adj = sparse_tensor.coalesce()



    def infonce(self, anchor, sample, pos_mask, neg_mask, tau):
        sim = self.similarity(anchor, sample) / tau
        exp_sim = torch.exp(sim) * neg_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()


    def similarity(self, h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()

    def forward(self, graph, feat):

        trans_feature = self.encoder_target(graph, feat)

        h1 = self.conv1(x=feat, edge_index=graph.edge_index, highpass=True)
        h2 = self.conv2(x=feat, edge_index=graph.edge_index, highpass=False)

        if self.num_MLP != 0:
            trans = F.normalize(self.projector(trans_feature))
        else:
            trans = F.normalize(trans_feature)

        nnodes = trans.shape[0]
        if (self.batch_size == 0) or (self.batch_size > nnodes):
            pos_mask = self.pos_mask.to_dense()
            dense_ = self.dense_adj.to_dense()
            loss2 = self.infonce(trans, F.normalize(h1), pos_mask, 1-pos_mask, self.temp)
            loss3 = self.infonce(trans, F.normalize(h2), dense_, 1-dense_, self.temp)
            loss = (1 - self.gamma) * loss2 + self.gamma * loss3
        else:
            node_idxs = list(range(nnodes))
            random.shuffle(node_idxs)
            batches = split_batch(node_idxs, self.batch_size)
            loss = 0
            for b in batches:
                weight = len(b) / nnodes
                semantic = extract(self.pos_mask,b)
                structure = extract(self.dense_adj ,b)

                loss2 = self.infonce(trans[b], F.normalize(h1)[b], semantic, 1-semantic, self.temp)
                loss3 = self.infonce(trans[b], F.normalize(h2)[b], structure, 1-structure, self.temp)

                loss += ((1 - self.gamma) * loss2 + self.gamma * loss3) * weight


        return loss


