def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='S3GCL')

    parser.add_argument('--lr1', type=float, default=5e-4, help='')
    parser.add_argument('--lr2', type=float, default=1e-2, help='')
    parser.add_argument('--wd1', type=float, default=1e-6, help='')
    parser.add_argument('--wd2', type=float, default=1e-5, help='')
    parser.add_argument('--n_layers', type=int, default=2, help='')
    parser.add_argument('--use_mlp', action='store_true', default=False, help='')
    parser.add_argument('--temp', type=float, default=0.5, help='')

    parser.add_argument("--batch_size", type=int, default=10000, help='Hidden layer dim.')

    parser.add_argument('--gpu', type=int, default=5, help='GPU index.')
    parser.add_argument('--num_MLP', type=int, default=1)
    parser.add_argument("--hid_dim", type=int, default=1024, help='Hidden layer dim.')

    parser.add_argument('--epochs', type=int, default=200, help='Training epochs.')
    parser.add_argument('--k', type=int, default=10, help='pass')
    parser.add_argument('--semantic', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0)

    parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')

    return parser.parse_args()