from __future__ import division
from __future__ import print_function

import time
import datetime
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.optimizer import OptimizerAE
from gae.input_data import load_data_with_label
from gae.model import GCNModelAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, \
     add_train_edges, get_cluster_acc, mask_test_edges

from sklearn.cluster import KMeans
from sklearn import metrics
from munkres import Munkres

# This code is GAE-based Cross-Graph.
# You may need to run the code multiple times for an averaged stable performance.

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'Cross-Graph', 'Model string (Cross-Graph) (GAE).')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

flags.DEFINE_integer('mod_edge', 40, 'Percentage of edges added to the structure.')
flags.DEFINE_float('label_concession', 0.02, 'The tuning range of label towards the prediction.')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

num_cluster = 7

# Load data
adj, features, labels = load_data_with_label(dataset_str)
num_nodes = adj.shape[0]
num_edges = int(adj[adj == 1].size / 2)
print('Number of edges: ' + str(num_edges))
num_add_edges = int(num_edges * FLAGS.mod_edge * 0.01)
print('Number of added edges: ' + str(num_add_edges))

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

adj_train, c_row_list, c_col_list = add_train_edges(adj_train, num_add_edges)
c_row_index = np.array(c_row_list)
c_col_index = np.array(c_col_list)
c_index = c_row_index * num_nodes + c_col_index

adj = adj_train

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

# Some preprocessing
adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_gnd1': tf.placeholder(tf.float32),
    'adj_gnd2': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
# model = None
GCN1 = GCNModelAE(placeholders, num_features, features_nonzero)
GCN2 = GCNModelAE(placeholders, num_features, features_nonzero)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    opt1 = OptimizerAE(preds=GCN1.reconstructions,
                      labels=placeholders['adj_gnd1'],
                      pos_weight=pos_weight,
                      norm=norm)
    opt2 = OptimizerAE(preds=GCN2.reconstructions,
                       labels=placeholders['adj_gnd2'],
                       pos_weight=pos_weight,
                       norm=norm)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []


def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(GCN1.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def clusteringAcc(true_label, pred_label):
    # best mapping between true_label and predict label
    l1 = list(set(true_label))
    numclass1 = len(l1)

    l2 = list(set(pred_label))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('Class Not equal, Error!!!!')
        return 0

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred_label[i1] == c2]

            # print(len(mps_d))
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()

    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(pred_label))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(true_label, new_predict)
    f1_macro = metrics.f1_score(true_label, new_predict, average='macro')
    precision_macro = metrics.precision_score(true_label, new_predict, average='macro')
    recall_macro = metrics.recall_score(true_label, new_predict, average='macro')
    f1_micro = metrics.f1_score(true_label, new_predict, average='micro')
    precision_micro = metrics.precision_score(true_label, new_predict, average='micro')
    recall_micro = metrics.recall_score(true_label, new_predict, average='micro')

    # f1 = 2 * (precision * recall) / (precision + recall)
    return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro


def sigmoid(x):
    return 1/(1+np.exp(-x))


cost_val = []
acc_val = []
val_roc_score1 = []
val_roc_score2 = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label_dense = adj_label.todense()

gnd1 = (np.array(adj_label_dense)).reshape(-1)
gnd2 = (np.array(adj_label_dense)).reshape(-1)

# Construct feed dictionary
feed_dict = construct_feed_dict(adj_norm, gnd1, features, placeholders)
feed_dict.update({placeholders['dropout']: FLAGS.dropout})

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()

    rec1 = sess.run(GCN1.reconstructions, feed_dict=feed_dict)
    rec1 = sigmoid(rec1)
    gnd1 = rec1 * gnd1 * FLAGS.label_concession + gnd1 * (1 - FLAGS.label_concession)

    rec2 = sess.run(GCN2.reconstructions, feed_dict=feed_dict)
    rec2 = sigmoid(rec2)
    gnd2 = rec2 * gnd2 * FLAGS.label_concession + gnd2 * (1 - FLAGS.label_concession)

    if model_str == 'GAE':
        pass
    elif model_str == 'Cross-Graph':
        feed_dict.update({placeholders['adj_gnd1']: gnd2})
        feed_dict.update({placeholders['adj_gnd2']: gnd1})
    else:
        print("model name error!")
        exit()

    # Run single weight update
    outs1 = sess.run([opt1.opt_op, opt1.cost, opt1.accuracy, GCN1.embeddings], feed_dict=feed_dict)
    outs2 = sess.run([opt2.opt_op, opt2.cost, opt2.accuracy, GCN2.embeddings], feed_dict=feed_dict)

    # Compute average loss
    avg_cost1 = outs1[1]
    avg_accuracy1 = outs1[2]
    avg_cost2 = outs2[1]
    avg_accuracy2 = outs2[2]

    roc_curr1, ap_curr1 = get_roc_score(val_edges, val_edges_false, emb=sess.run(GCN1.z_mean, feed_dict=feed_dict))
    val_roc_score1.append(roc_curr1)
    roc_curr2, ap_curr2 = get_roc_score(val_edges, val_edges_false, emb=sess.run(GCN2.z_mean, feed_dict=feed_dict))
    val_roc_score2.append(roc_curr2)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss1=", "{:.5f}".format(avg_cost1), "train_loss2=", "{:.5f}".format(avg_cost2),
          "train_acc1=", "{:.5f}".format(avg_accuracy1), "train_acc2=", "{:.5f}".format(avg_accuracy2),
          "val_roc1=", "{:.5f}".format(val_roc_score1[-1]), "val_roc2=", "{:.5f}".format(val_roc_score2[-1]),
          "val_ap1=", "{:.5f}".format(ap_curr1), "val_ap2=", "{:.5f}".format(ap_curr2),
          "time=", "{:.5f}".format(time.time() - t))

    emb1 = outs1[3]
    emb2 = outs2[3]

    kmeans = KMeans(n_clusters=num_cluster, random_state=0)
    y_pred1 = kmeans.fit_predict(emb1)
    y_pred2 = kmeans.fit_predict(emb2)
    y_truth = labels.argmax(1)

    acc1 = np.round(get_cluster_acc(y_truth, y_pred1), 5)
    nmi1 = np.round(metrics.normalized_mutual_info_score(y_truth, y_pred1), 5)
    ari1 = np.round(metrics.adjusted_rand_score(y_truth, y_pred1), 5)
    acc2 = np.round(get_cluster_acc(y_truth, y_pred2), 5)
    nmi2 = np.round(metrics.normalized_mutual_info_score(y_truth, y_pred2), 5)
    ari2 = np.round(metrics.adjusted_rand_score(y_truth, y_pred2), 5)

    print("ACC1: ", "\033[1;32m {:.5f} \033[0m".format(acc1), ",NMI1: ", "{:.5f}".format(nmi1), ",ARI1: ""{:.5f}".format(ari1),
          "ACC2: ", "\033[1;32m {:.5f} \033[0m".format(acc2), ",NMI2: ", "{:.5f}".format(nmi2), ",ARI2: ""{:.5f}".format(ari2))

print("Optimization Finished!")

roc_score1, ap_score1 = get_roc_score(test_edges, test_edges_false, emb=sess.run(GCN1.z_mean, feed_dict=feed_dict))
roc_score2, ap_score2 = get_roc_score(test_edges, test_edges_false, emb=sess.run(GCN2.z_mean, feed_dict=feed_dict))
print('Test ROC score 1 & 2: ' + str(roc_score1) + ' & ' + str(roc_score2))
print('Test AP score 1 & 2: ' + str(ap_score1) + ' & ' + str(ap_score2))
