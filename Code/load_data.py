
from torch_geometric.datasets import Planetoid, WebKB, Amazon, WikipediaNetwork, Actor, HeterophilousGraphDataset,Coauthor
import scipy.io
from sklearn.preprocessing import label_binarize
from torch_geometric.data import Data
import torch_geometric.transforms as T
import scipy.io
import numpy as np
import scipy.sparse
import torch
import csv
import pandas as pd
import json
from ogb.nodeproppred import NodePropPredDataset
from os import path
# import gdown
from torch_sparse import SparseTensor
# from google_drive_downloader import GoogleDriveDownloader as gdd
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from scipy import sparse as sp
from sklearn.metrics import roc_auc_score, f1_score

from torch_sparse import SparseTensor
# from google_drive_downloader import GoogleDriveDownloader as gdd



def homo_ratio(data, pred, train_mask):
    pred[train_mask] = data.y[train_mask]
    eq = pred[data.edge_index[0]] == pred[data.edge_index[1]]
    return eq.sum() / eq.shape[0]

def rand_train_test_idx(label, train_prop, valid_prop, test_prop, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    labeled_nodes = torch.where(label != -1)[0]

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)
    test_num = int(n * test_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:train_num + valid_num + test_num]

    train_idx = train_indices
    valid_idx = val_indices
    test_idx = test_indices

    return {'train': train_idx.numpy(), 'valid': valid_idx.numpy(), 'test': test_idx.numpy()}

def index_to_mask(splits_lst, num_nodes):
    mask_len = len(splits_lst)
    train_mask = torch.zeros((mask_len, num_nodes), dtype=torch.bool)
    val_mask = torch.zeros((mask_len, num_nodes), dtype=torch.bool)
    test_mask = torch.zeros((mask_len, num_nodes), dtype=torch.bool)

    for i in range(mask_len):
        train_mask[i][splits_lst[i]['train']] = True
        val_mask[i][splits_lst[i]['valid']] = True
        test_mask[i][splits_lst[i]['test']] = True

    return train_mask.T, val_mask.T, test_mask.T


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).detach().cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


def eval_acc(y_true, y_pred):

    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label



from typing import Optional, Callable
import os.path as osp
import torch
import numpy as np
from torch_geometric.utils import to_undirected
from torch_geometric.data import InMemoryDataset, download_url, Data

class WikipediaNetwork2(InMemoryDataset):
    r"""The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"chameleon"`,
            :obj:`"crocodile"`, :obj:`"squirrel"`).
        geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
            pre-processing data as introduced in the `"Geom-GCN: Geometric
            Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
            in which the average monthly traffic of the web page is converted
            into five categories to predict.
            If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
            available.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    raw_url = 'https://graphmining.ai/datasets/ptg/wiki'
    processed_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                     'geom-gcn/master')

    def __init__(self, root: str, name: str, geom_gcn_preprocess: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        self.geom_gcn_preprocess = geom_gcn_preprocess
        assert self.name in ['chameleon', 'crocodile', 'squirrel']
        if geom_gcn_preprocess and self.name == 'crocodile':
            raise AttributeError("The dataset 'crocodile' is not available in "
                                 "case 'geom_gcn_preprocess=True'")
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'raw')
        else:
            return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'processed')
        else:
            return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        if self.geom_gcn_preprocess:
            return (['out1_node_feature_label.txt', 'out1_graph_edges.txt'] +
                    [f'{self.name}_split_0.6_0.2_{i}.npz' for i in range(10)])
        else:
            return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        if self.geom_gcn_preprocess:
            for filename in self.raw_file_names[:2]:
                url = f'{self.processed_url}/new_data/{self.name}/{filename}'
                download_url(url, self.raw_dir)
            for filename in self.raw_file_names[2:]:
                url = f'{self.processed_url}/splits/{filename}'
                download_url(url, self.raw_dir)
        else:
            download_url(f'{self.raw_url}/{self.name}.npz', self.raw_dir)

    def process(self):
        if self.geom_gcn_preprocess:
            with open(self.raw_paths[0], 'r') as f:
                data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)
            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

            with open(self.raw_paths[1], 'r') as f:
                data = f.read().split('\n')[1:-1]
                data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            # edge_index = to_undirected(edge_index, num_nodes=x.size(0))
            print('test')
            train_masks, val_masks, test_masks = [], [], []
            for filepath in self.raw_paths[2:]:
                f = np.load(filepath)
                train_masks += [torch.from_numpy(f['train_mask'])]
                val_masks += [torch.from_numpy(f['val_mask'])]
                test_masks += [torch.from_numpy(f['test_mask'])]
            train_mask = torch.stack(train_masks, dim=1).to(torch.bool)
            val_mask = torch.stack(val_masks, dim=1).to(torch.bool)
            test_mask = torch.stack(test_masks, dim=1).to(torch.bool)

            data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)

        else:
            data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
            x = torch.from_numpy(data['features']).to(torch.float)
            edge_index = torch.from_numpy(data['edges']).to(torch.long)
            edge_index = edge_index.t().contiguous()
            # edge_index = to_undirected(edge_index, num_nodes=x.size(0))
            y = torch.from_numpy(data['label']).to(torch.float)
            train_mask = torch.from_numpy(data['train_mask']).to(torch.bool)
            test_mask = torch.from_numpy(data['test_mask']).to(torch.bool)
            val_mask = torch.from_numpy(data['val_mask']).to(torch.bool)
            data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
import gdown
def load_dataset(dataname, train_prop, valid_prop, test_prop, num_masks):
    #666
    assert dataname in ('cora', 'citeseer', 'pubmed', 'cs', 'physics','computers', 'photo',
                        'texas', 'wisconsin', 'cornell', 'squirrel', 'chameleon', 'crocodile', 'actor',
                        'twitch', 'fb100', 'Penn94', 'deezer', 'year', 'snap-patents', 'pokec', 'yelpchi', 'gamer',
                        'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'genius',
                        'roman_empire', 'amazon_ratings', 'minesweeper', 'tolokers', 'questions'), 'Invalid dataset'

    if dataname in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='./data/graph/', name=dataname)
        data = dataset[0]
        # data.train_mask = torch.unsqueeze(data.train_mask, dim=1)
        # data.val_mask = torch.unsqueeze(data.val_mask, dim=1)
        # data.test_mask = torch.unsqueeze(data.test_mask, dim=1)

        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname in ['texas', 'wisconsin', 'cornell']:
        dataset = WebKB(root='./data/graph/', name=dataname)
        data = dataset[0]

        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname in ['actor']:
        dataset = Actor(root='./data/graph/actor')
        data = dataset[0]

        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname in ['squirrel', 'chameleon']:
        preProcDs = WikipediaNetwork(root='./data/graph/', name=dataname, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(root='./data/graph/', name=dataname, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index

        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname in ['crocodile']:
        dataset = WikipediaNetwork2(root='./data/graph/', name=dataname, geom_gcn_preprocess=False)
        data = dataset[0]
        data.y = data.y.long()

        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname in ['computers', 'photo']:
        dataset = Amazon(root='./data/graph/', name=dataname)
        data = dataset[0]
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop,test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname in ['cs', 'physics']:
        dataset = Coauthor(root='./data/graph/', name=dataname,transform=T.NormalizeFeatures())
        data = dataset[0]
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop,test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname == 'Penn94':
        data = load_fb100_dataset(dataname, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop, num_masks=num_masks)

    elif dataname == 'twitch':
        data = load_twitch_dataset('DE', train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop, num_masks=num_masks)

    elif dataname == 'fb100':
        data = load_fb100_dataset('', train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop, num_masks=num_masks)

    elif dataname == 'deezer':
        data = load_deezer()
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname == 'genius':
        data = load_genius()
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname == 'year':
        data = load_arxiv_year()
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname == 'snap-patents':
        data = load_snap_patents()
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname == 'pokec':
        data = load_pokec()
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname == 'yelpchi':
        data = load_yelpchi()
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname == 'gamer':
        data = load_twitch_gamer_dataset()
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname in ('ogbn-arxiv', 'ogbn-products', 'ogbn-proteins'):
        dataset = PygNodePropPredDataset(name=dataname,root='./data/graph/')
        data = dataset[0]
        data.y = data.y.squeeze()
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)
        
    elif dataname in ['roman_empire', 'amazon_ratings', 'minesweeper', 'tolokers', 'questions']:
        dataset = HeterophilousGraphDataset(root='./data/graph/', name=dataname)
        data = dataset[0]
        # data.y = data.y.squeeze()
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    return data


dataset_drive_url = {
    'twitch-gamer_feat' : '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges' : '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP', # Wiki 1.9M
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u', # Wiki 1.9M
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK' # Wiki 1.9M
}
DATAPATH = './data/graph/'



def load_deezer():
    # filename = 'deezer-europe'
    # dataset = NCDataset(filename)
    deezer = scipy.io.loadmat('./data/graph/deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()

    data = Data(x=node_feat, edge_index=edge_index, y=label, num_nodes=label.shape[0])
    return data


def load_genius():
    filename = 'genius'
    # dataset = NCDataset(filename)
    fulldata = scipy.io.loadmat(f'./data/graph/genius.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
    label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    data = Data(x=node_feat, edge_index=edge_index, y=label, num_nodes=label.shape[0])
    return data


def load_fb100_dataset(sub_dataname, train_prop, valid_prop, test_prop, num_masks):
    assert sub_dataname in ('Amherst41', 'Cornell5', 'Johns Hopkins55', 'Penn94', 'Reed98'), 'Invalid dataset'
    A, metadata = load_fb100(sub_dataname)
    # dataset = NCDataset(filename)
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
    metadata = metadata.astype(np.int64)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled
    label = torch.tensor(label, dtype=torch.long)

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]

    # if sub_dataname == 'Penn94':
    #     splits_lst = np.load('./data/graph/splits/fb100-Penn94-splits.npy', allow_pickle=True)
    # else:
    splits_lst = [rand_train_test_idx(label, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
    train_mask, val_mask, test_mask = index_to_mask(splits_lst, num_nodes)

    data = Data(x=node_feat, edge_index=edge_index, y=label,
                train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)

    return data


def load_fb100(filename):
    # e.g. filename = Rutgers89 or Cornell5 or Wisconsin87 or Amherst41
    # columns are: student/faculty, gender, major,
    #              second major/minor, dorm/house, year/ high school
    # 0 denotes missing entry
    mat = scipy.io.loadmat('./data/graph/facebook100/' + filename + '.mat')
    A = mat['A']
    metadata = mat['local_info']
    return A, metadata


def load_twitch_dataset(sub_dataname, train_prop, valid_prop, test_prop, num_masks):
    assert sub_dataname in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    A, label, features = load_twitch(sub_dataname)
    # dataset = NCDataset(lang)
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long)
    num_nodes = node_feat.shape[0]

    # if sub_dataname == 'DE':
    #     splits_lst = np.load('./data/graph/splits/twitch-e-DE-splits.npy', allow_pickle=True)
    # else:
    splits_lst = [rand_train_test_idx(label, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                         for _ in range(num_masks)]
    train_mask, val_mask, test_mask = index_to_mask(splits_lst, num_nodes)

    data = Data(x=node_feat, edge_index=edge_index, y=label,
                train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)

    return data


def load_twitch(sub_dataname):
    assert sub_dataname in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    filepath = './data/graph/twitch/' + sub_dataname
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    with open(f"{filepath}/musae_{sub_dataname}_target.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            # handle FR case of non-unique rows
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2] == "True"))
                node_ids.append(int(row[5]))

    node_ids = np.array(node_ids, dtype=np.int)
    with open(f"{filepath}/musae_{sub_dataname}_edges.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
    with open(f"{filepath}/musae_{sub_dataname}_features.json", 'r') as f:
        j = json.load(f)
    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)
    inv_node_ids = {node_id: idx for (idx, node_id) in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]

    n = label.shape[0]
    A = scipy.sparse.csr_matrix((np.ones(len(src)),
                                 (np.array(src), np.array(targ))),
                                shape=(n, n))
    features = np.zeros((n, 3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    features = features[:, np.sum(features, axis=0) != 0]  # remove zero cols
    new_label = label[reorder_node_ids]
    label = new_label

    return A, label, features


def load_arxiv_year(nclass=5):
    # filename = 'arxiv-year'
    # dataset = NCDataset(filename)
    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv')
    graph = ogb_dataset.graph
    edge_index = torch.as_tensor(graph['edge_index'])
    x = torch.as_tensor(graph['node_feat'])

    label = even_quantile_labels(graph['node_year'].flatten(), nclass, verbose=False)
    y = torch.as_tensor(label).reshape(-1)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def load_snap_patents(nclass=5):
    if not path.exists(f'{DATAPATH}snap_patents.mat'):
        p = dataset_drive_url['snap-patents']
        print(f"Snap patents url: {p}")
        gdown.download(id=dataset_drive_url['snap-patents'], \
                       output=f'{DATAPATH}snap_patents.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}snap_patents.mat')

    # dataset = NCDataset('snap_patents')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'].todense(), dtype=torch.float)
    num_nodes = int(fulldata['num_nodes'])

    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    label = torch.tensor(label, dtype=torch.long)
    data = Data(x=node_feat, edge_index=edge_index, y=label, num_nodes=num_nodes)
    return data


def load_pokec():
    if not path.exists(f'{DATAPATH}pokec.mat'):
        gdown.download(id=dataset_drive_url['pokec'], \
                       output=f'{DATAPATH}pokec.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}pokec.mat')

    # dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    label = torch.tensor(label, dtype=torch.long)
    data = Data(x=node_feat, edge_index=edge_index, y=label, num_nodes=num_nodes)
    return data


def load_yelpchi():
    if not path.exists(f'{DATAPATH}YelpChi.mat'):
        gdown.download(id=dataset_drive_url['yelp-chi'], \
            output=f'{DATAPATH}YelpChi.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}YelpChi.mat')
    A = fulldata['homo']
    edge_index = np.array(A.nonzero())
    node_feat = fulldata['features']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    # dataset = NCDataset('YelpChi')
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long)
    data = Data(x=node_feat, edge_index=edge_index, y=label, num_nodes=num_nodes)
    return data


def load_twitch_gamer_dataset(task="mature", normalize=True):
    if not path.exists(f'{DATAPATH}twitch-gamer_feat.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_feat'],
                       output=f'{DATAPATH}twitch-gamer_feat.csv', quiet=False)
    if not path.exists(f'{DATAPATH}twitch-gamer_edges.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_edges'],
                       output=f'{DATAPATH}twitch-gamer_edges.csv', quiet=False)

    edges = pd.read_csv(f'{DATAPATH}twitch-gamer_edges.csv')
    nodes = pd.read_csv(f'{DATAPATH}twitch-gamer_feat.csv')
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)
    label, features = load_twitch_gamer(nodes, task)
    node_feat = torch.tensor(features, dtype=torch.float)

    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)

    # dataset = NCDataset("twitch-gamer")
    # dataset.graph = {'edge_index': edge_index,
    #                  'node_feat': node_feat,
    #                  'edge_feat': None,
    #                  'num_nodes': num_nodes}
    label = torch.tensor(label)
    data = Data(x=node_feat, edge_index=edge_index, y=label, num_nodes=num_nodes)
    return data


def load_twitch_gamer(nodes, task="dead_account"):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding

    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()

    return label, features

