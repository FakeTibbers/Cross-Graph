import numpy as np
import scipy.sparse as sp
import random


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_gnd1']: adj})
    feed_dict.update({placeholders['adj_gnd2']: adj})
    return feed_dict


# def mask_test_edges(adj):
#     # Function to build test set with 10% positive links
#     # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
#     # TODO: Clean up.
#
#     # Remove diagonal elements
#     adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
#     adj.eliminate_zeros()
#     # Check that diag is zero:
#     assert np.diag(adj.todense()).sum() == 0
#
#     adj_triu = sp.triu(adj)
#     adj_tuple = sparse_to_tuple(adj_triu)
#     edges = adj_tuple[0]
#     edges_all = sparse_to_tuple(adj)[0]
#     num_test = int(np.floor(edges.shape[0] / 10.))
#     num_val = int(np.floor(edges.shape[0] / 20.))
#
#     all_edge_idx = list(range(edges.shape[0]))
#     np.random.shuffle(all_edge_idx)
#     val_edge_idx = all_edge_idx[:num_val]
#     test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
#     test_edges = edges[test_edge_idx]
#     val_edges = edges[val_edge_idx]
#     train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
#
#     def ismember(a, b, tol=5):
#         rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
#         return np.any(rows_close)
#
#     test_edges_false = []
#     while len(test_edges_false) < len(test_edges):
#         idx_i = np.random.randint(0, adj.shape[0])
#         idx_j = np.random.randint(0, adj.shape[0])
#         if idx_i == idx_j:
#             continue
#         if ismember([idx_i, idx_j], edges_all):
#             continue
#         if test_edges_false:
#             if ismember([idx_j, idx_i], np.array(test_edges_false)):
#                 continue
#             if ismember([idx_i, idx_j], np.array(test_edges_false)):
#                 continue
#         test_edges_false.append([idx_i, idx_j])
#
#     val_edges_false = []
#     while len(val_edges_false) < len(val_edges):
#         idx_i = np.random.randint(0, adj.shape[0])
#         idx_j = np.random.randint(0, adj.shape[0])
#         if idx_i == idx_j:
#             continue
#         if ismember([idx_i, idx_j], train_edges):
#             continue
#         if ismember([idx_j, idx_i], train_edges):
#             continue
#         if ismember([idx_i, idx_j], val_edges):
#             continue
#         if ismember([idx_j, idx_i], val_edges):
#             continue
#         if val_edges_false:
#             if ismember([idx_j, idx_i], np.array(val_edges_false)):
#                 continue
#             if ismember([idx_i, idx_j], np.array(val_edges_false)):
#                 continue
#         val_edges_false.append([idx_i, idx_j])
#
#     assert ~ismember(test_edges_false, edges_all)
#     assert ~ismember(val_edges_false, edges_all)
#     assert ~ismember(val_edges, train_edges)
#     assert ~ismember(test_edges, train_edges)
#     assert ~ismember(val_edges, test_edges)
#
#     data = np.ones(train_edges.shape[0])
#
#     # Re-build adj matrix
#     adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
#     adj_train = adj_train + adj_train.T
#
#     # NOTE: these edge lists only contain single direction of edge!
#     return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


# The original GAE mask_test_edges() sometimes goes wrong, we use this updated one from the GAE github.
def mask_test_edges(adj, test_percent=10., val_percent=5.):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[None, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert adj.diagonal().sum() == 0

    edges_positive, _, _ = sparse_to_tuple(adj)
    edges_positive = edges_positive[edges_positive[:, 1] > edges_positive[:, 0],
                     :]  # filtering out edges from lower triangle of adjacency matrix
    val_edges, val_edges_false, test_edges, test_edges_false = None, None, None, None

    # number of positive (and negative) edges in test and val sets:
    num_test = int(np.floor(edges_positive.shape[0] / (100. / test_percent)))
    num_val = int(np.floor(edges_positive.shape[0] / (100. / val_percent)))

    # sample positive edges for test and val sets:
    edges_positive_idx = np.arange(edges_positive.shape[0])
    np.random.shuffle(edges_positive_idx)
    val_edge_idx = edges_positive_idx[:num_val]
    test_edge_idx = edges_positive_idx[num_val:(num_val + num_test)]
    test_edges = edges_positive[test_edge_idx]  # positive test edges
    val_edges = edges_positive[val_edge_idx]  # positive val edges
    train_edges = np.delete(edges_positive, np.hstack([test_edge_idx, val_edge_idx]), axis=0)  # positive train edges

    # the above strategy for sampling without replacement will not work for sampling negative edges on large graphs, because the pool of negative edges is much much larger due to sparsity
    # therefore we'll use the following strategy:
    # 1. sample random linear indices from adjacency matrix WITH REPLACEMENT (without replacement is super slow). sample more than we need so we'll probably have enough after all the filtering steps.
    # 2. remove any edges that have already been added to the other edge lists
    # 3. convert to (i,j) coordinates
    # 4. swap i and j where i > j, to ensure they're upper triangle elements
    # 5. remove any duplicate elements if there are any
    # 6. remove any diagonal elements
    # 7. if we don't have enough edges, repeat this process until we get enough

    positive_idx, _, _ = sparse_to_tuple(adj)  # [i,j] coord pairs for all true edges
    positive_idx = positive_idx[:, 0] * adj.shape[0] + positive_idx[:, 1]  # linear indices

    test_edges_false = np.empty((0, 2), dtype='int64')
    idx_test_edges_false = np.empty((0,), dtype='int64')
    while len(test_edges_false) < len(test_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0] ** 2, 2 * (num_test - len(test_edges_false)), replace=True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique=True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx, colidx)).transpose()
        # step 4:
        lowertrimask = coords[:, 0] > coords[:, 1]
        coords[lowertrimask] = coords[lowertrimask][:, ::-1]
        # step 5:
        coords = np.unique(coords, axis=0)  # note: coords are now sorted lexicographically
        np.random.shuffle(coords)  # not any more
        # step 6:
        coords = coords[coords[:, 0] != coords[:, 1]]
        # step 7:
        coords = coords[:min(num_test, len(idx))]
        test_edges_false = np.append(test_edges_false, coords, axis=0)
        idx = idx[:min(num_test, len(idx))]
        idx_test_edges_false = np.append(idx_test_edges_false, idx)

    val_edges_false = np.empty((0, 2), dtype='int64')
    idx_val_edges_false = np.empty((0,), dtype='int64')
    while len(val_edges_false) < len(val_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0] ** 2, 2 * (num_val - len(val_edges_false)), replace=True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_val_edges_false, assume_unique=True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx, colidx)).transpose()
        # step 4:
        lowertrimask = coords[:, 0] > coords[:, 1]
        coords[lowertrimask] = coords[lowertrimask][:, ::-1]
        # step 5:
        coords = np.unique(coords, axis=0)  # note: coords are now sorted lexicographically
        np.random.shuffle(coords)  # not any more
        # step 6:
        coords = coords[coords[:, 0] != coords[:, 1]]
        # step 7:
        coords = coords[:min(num_val, len(idx))]
        val_edges_false = np.append(val_edges_false, coords, axis=0)
        idx = idx[:min(num_val, len(idx))]
        idx_val_edges_false = np.append(idx_val_edges_false, idx)

    # sanity checks:
    train_edges_linear = train_edges[:, 0] * adj.shape[0] + train_edges[:, 1]
    test_edges_linear = test_edges[:, 0] * adj.shape[0] + test_edges[:, 1]
    assert not np.any(np.in1d(idx_test_edges_false, positive_idx))
    assert not np.any(np.in1d(idx_val_edges_false, positive_idx))
    assert not np.any(np.in1d(val_edges[:, 0] * adj.shape[0] + val_edges[:, 1], train_edges_linear))
    assert not np.any(np.in1d(test_edges_linear, train_edges_linear))
    assert not np.any(np.in1d(val_edges[:, 0] * adj.shape[0] + val_edges[:, 1], test_edges_linear))

    # Re-build adj matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def add_train_edges(adj, num_mod_edge):
    print('Adding ' + str(num_mod_edge) + ' edges')
    adj_dense = adj.todense()
    num_nodes = adj.shape[0]
    num_added_edge = 0
    c_row_list = []
    c_col_list = []
    while num_added_edge < num_mod_edge:
        rd_row = random.randint(0, num_nodes - 1)
        rd_col = random.randint(0, num_nodes - 1)
        if adj_dense[rd_row, rd_col] == 0 and adj_dense[rd_col, rd_row] == 0 and rd_row != rd_col:
            num_added_edge += 1
            adj_dense[rd_row, rd_col] = 1
            adj_dense[rd_col, rd_row] = 1
            c_row_list.append(rd_row)
            c_col_list.append(rd_col)

    adj_csr = sp.csr_matrix(adj_dense)

    return adj_csr, c_col_list, c_row_list

def get_cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size