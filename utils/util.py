import torch
import numpy as np
import pandas as pd
from scipy import sparse

def dataloader(*args):
    def loaddata(idx, todense=True):
        xs = (arg[idx] for arg in args)
        xs = tuple(x.toarray() if sparse.isspmatrix(x) and todense else x for x in xs)
        return xs[0] if len(xs) == 1 else xs
    return loaddata
##
def load_train_data(csv_file, n_items=None):
    """
    Input:
    csv_file (string): the csv file contains two columns (uid, sid).
    n_items (int): the number of items.

    Output:
    data (sparse matrix): each row representing a user's activities over the `n_items` items.
    """
    df = pd.read_csv(csv_file)
    n_users = len(df.uid.unique())
    if n_items is None: n_items = len(df.sid.unique())

    row_ind, col_ind = df.uid, df.sid
    weight = df.weight if "weight" in df.columns else np.ones_like(row_ind)
    data = sparse.csr_matrix((weight, (row_ind, col_ind)),
                             dtype='float32',
                             shape=(n_users, n_items))
    return data

def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    """
    Input:
        csv_file_tr (string): the csv file contains two columns (uid, sid).
        csv_file_te (string): the csv file contains two columns (uid, sid).
        n_items (int): the number of items.

    Output:
        two sparse matrices: each row representing a user's activities over the `n_items` items.
    """
    df_tr = pd.read_csv(csv_file_tr)
    df_te = pd.read_csv(csv_file_te)

    start_idx, end_idx = df_tr.uid.min(), df_tr.uid.max()

    rows_tr, cols_tr = df_tr.uid - start_idx, df_tr.sid
    rows_te, cols_te = df_te.uid - start_idx, df_te.sid

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)),
                                dtype='float32',
                                shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)),
                                dtype='float32',
                                shape=(end_idx - start_idx + 1, n_items))
    
    return data_tr, data_te

def load_train_data_mf(csv_file, n_items=None):
    data = load_train_data(csv_file, n_items)
    return np.arange(data.shape[0]), data

def load_test_data_mf(csv_file, n_items):
    """
    Input:
        csv_file (string): the csv file contains two columns (uid, sid).
    """
    df = pd.read_csv(csv_file)

    start_idx, end_idx = df.uid.min(), df.uid.max()
    rows, cols = df.uid - start_idx, df.sid

    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)),
                             dtype='float32',
                             shape=(end_idx - start_idx + 1, n_items))
    return np.arange(start_idx, end_idx+1), data

def recall_at_k(X, Y, k=1, dtype=torch.float64):
    """
    Computing recall@k.

    Input:
        X array(n, m): the predicted matrix with each row representing a user's preference over m items.
        Y array(n, m): the ground truth matrix, assuming that all zero entries indicate no relevance.
    """
    X = torch.tensor(X, dtype=dtype)
    Y = torch.tensor(Y, dtype=dtype)
    n = X.shape[0]

    vals, inds = torch.topk(X, k, dim=1, sorted=False)
    X = torch.zeros_like(X, dtype=bool)
    X[torch.arange(n).reshape(-1, 1), inds] = True
    Y = Y > 0

    nnz = torch.sum(Y, dim=1).cpu().numpy()
    num = torch.sum(X * Y, dim=1).cpu().numpy()
    idx = nnz > 0
    num = num[idx]
    nnz = nnz[idx]
    return num / np.minimum(k, nnz)

def ndcg_at_k(X, Y, k=1, dtype=torch.float64):
    """
    Computing the NDCG@k.

    Input:
        X array(n, m): the predicted matrix with each row representing a user's preference over m items.
        Y array(n, m): the ground truth matrix, assuming that all zero entries indicate no relevance.
    """
    X = torch.tensor(X, dtype=dtype)
    Y = torch.tensor(Y, dtype=dtype)
    n = X.shape[0]

    vals, inds = torch.topk(X, k, dim=1)
    
    discount = 1. / torch.log2(torch.arange(2, k+2, dtype=dtype))

    DCG = torch.sum(Y[torch.arange(n).reshape(-1, 1), inds] * discount, dim=1)
    nnz = torch.sum(Y > 0, dim=1).cpu().numpy()
    iDCG = torch.tensor([torch.sum(discount[0:min(i, k)]) for i in nnz])
    ndcg = (DCG / iDCG).cpu().numpy()
    return ndcg[np.logical_not(np.isnan(ndcg))]

############## utility functions for movieLens/netflix #################################
def random_select(df, num_users, num_items):
    """
    Select a subset of df with `num_users` and `num_items` randomly.
    """
    users = np.random.choice(df.userId.unique(), size=(num_users,), replace=False)
    df = df[df.userId.isin(users)]
    items = np.random.choice(df.movieId.unique(), size=(num_items,), replace=False)
    df = df[df.movieId.isin(items)]
    return df

def filter_triplets(tp, min_uc=5, min_ic=0):
    """
    Filter out cold movies and less active users.

    Input:
        tp (DataFrame): userId, movieId, rating, timestamp.

    Output:
        tp (DataFrame): filtered records.
        usercount (Series): the count of movies watched by each user.
        itemcount (Series): the count of users watching each movie.
    """
    def get_count(df, id):
        return df.groupby(id).size()
    if min_ic > 0: # remove cold movies
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_ic])]

    if min_uc > 0: # remove less active users
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount

def split_train_test_proportion(data, test_prop=0.2):
    """
    Spliting a user's records into two parts with proportions 1 - test_prop, test_prop, respectively.

    Input:
        data (DataFrame): userId, movieId, rating, timestamp.
    """
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    for i, (_, group) in enumerate(data_grouped_by_user):
        ## the number of items consumed by one user
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            test_idx = np.random.choice(n_items_u, size=int(test_prop*n_items_u), replace=False)
            idx[test_idx] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])

        if i % 1000 == 0:
            print("%d users sampled" % i)

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

def split_train_vald_test_proportion(data, test_prop=0.2):
    """
    Spliting a user's records into three parts with proportions 1 - 2*test_prop, test_prop, test_prop, respectively.

    Input:
        data (DataFrame): userId, movieId, rating, timestamp.
    """
    data_grouped_by_user = data.groupby('userId')
    tr_list, vd_list, te_list = list(), list(), list()

    for i, (_, group) in enumerate(data_grouped_by_user):
        ## the number of items consumed by one user
        n_items_u = len(group)
        n_test = int(test_prop*n_items_u)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            test_idx = np.random.choice(n_items_u, size=2*n_test, replace=False)
            idx[test_idx] = True
            vd_te_group = group[idx]

            tr_list.append(group[np.logical_not(idx)])
            vd_list.append(vd_te_group[:n_test])
            te_list.append(vd_te_group[n_test:])

        if i % 1000 == 0:
            print("%d users sampled" % i)

    data_tr = pd.concat(tr_list)
    data_vd = pd.concat(vd_list)
    data_te = pd.concat(te_list)

    return data_tr, data_vd, data_te
############## utility functions for movieLens #################################