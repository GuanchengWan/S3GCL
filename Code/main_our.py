import time
from collections import Counter

from args import get_args
from load_data import load_dataset
from model_our import Model_our
from model_our import LogReg
import statistics
import torch_geometric
import torch
import torch as th
import torch.nn as nn
import  numpy as np
import warnings
import random
seed = 1024
warnings.filterwarnings('ignore')
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False



args = get_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'


if __name__ == '__main__':
    print(args)
    # load hyperparameters
    dataname = args.dataname
    hid_dim = args.hid_dim
    out_dim = args.hid_dim
    n_layers = args.n_layers
    temp = args.temp
    epochs = args.epochs
    lr1 = args.lr1
    wd1 = args.wd1
    lr2 = args.lr2
    wd2 = args.wd2
    device = args.device


    data = load_dataset(dataname, 0.6, 0.2, 0.2, 5)
    from torch_geometric.utils import to_undirected, remove_self_loops

    # to dgl
    data.edge_index = to_undirected(data.edge_index)
    import dgl
    graph = dgl.graph((data.edge_index[0], data.edge_index[1]))
    feat = data.x
    labels = data.y
    graph = graph.to(device)
    feat = feat.to(device)
    graph = graph.remove_self_loop().add_self_loop()
    graph.edge_index = torch.stack(graph.edges())
    num_class = (torch.max(labels)+1).item()
    print("Nodes:" + str(feat.shape[0]))
    in_dim = feat.shape[1]
    model = Model_our(in_dim, hid_dim, out_dim, n_layers, temp, args.use_mlp,  args.num_MLP,args.gamma,args.k)
    model = model.to(device)

    model.set_mask_knn(feat.cpu(), graph.edge_index, args.semantic, device, dataset=dataname)

    model.batch_size = args.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    start = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(graph, feat)
        loss.backward()
        optimizer.step()
        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))

    end = time.time()
    def to_MB(byte):
        return byte / 1024.0 / 1024.0
    max_memory_allocated = torch.cuda.max_memory_allocated(device)
    print("=== Evaluation ===")
    graph = graph.remove_self_loop().add_self_loop()


    embeds = model.get_embedding(graph, feat)
    results = []
    # train
    for i in range(data.train_mask.shape[1]):
        print('MASK:', i)
        train_mask = data.train_mask[:, i]
        val_mask = data.val_mask[:, i]
        test_mask = data.test_mask[:, i]

        train_idx_tmp = train_mask
        val_idx_tmp = val_mask
        test_idx_tmp = test_mask
        train_embs = embeds[train_idx_tmp]
        val_embs = embeds[val_idx_tmp]
        test_embs = embeds[test_idx_tmp]

        label = labels.to(device)

        train_labels = label[train_idx_tmp]
        val_labels = label[val_idx_tmp]
        test_labels = label[test_idx_tmp]

        train_feat = feat[train_idx_tmp]
        val_feat = feat[val_idx_tmp]
        test_feat = feat[test_idx_tmp]

        logreg = LogReg(train_embs.shape[1], num_class)
        opt = th.optim.Adam(logreg.parameters(), lr=lr2, weight_decay=wd2)

        logreg = logreg.to(device)
        loss_fn = nn.CrossEntropyLoss()

        best_val_acc = 0
        eval_acc = 0

        for epoch in range(600):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            preds = th.argmax(logits, dim=1)
            train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()
            logreg.eval()
            with th.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)

                val_preds = th.argmax(val_logits, dim=1)
                test_preds = th.argmax(test_logits, dim=1)

                val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
                test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    eval_acc = test_acc
                if epoch % 30 == 0:
                    print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc, test_acc))
        results.append(eval_acc.item())
        print(f'\033[0;30;43m Validation Accuracy: {best_val_acc}, Test Accuracy: {eval_acc} \033[0m')
    print("mean:")
    print(statistics.mean(results)*100)
    print("std:")
    print(statistics.stdev(results)*100)
