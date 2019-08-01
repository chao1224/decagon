from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
import time
import os

import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
import pandas as pd
from sklearn import metrics

from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing
import logging
from tqdm import tqdm
import pickle
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s-%(message)s')


# Train on CPU (hide GPU) due to memory constraints
# os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Train on GPU
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

np.random.seed(0)

###########################################################
#
# Functions
#
###########################################################


def get_accuracy_scores(edges_pos, edges_neg, edge_type):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update(
        {placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)
        assert adj_mats_orig[edge_type[:2]
                             ][edge_type[2]][u, v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]
                             ][edge_type[2]][u, v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(
        zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=50)

    return roc_sc, aupr_sc, apk_sc


def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.placeholder(tf.int32, name='batch'),
        'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i, j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders


""" Generate real dataset
"""
# paramter
val_test_size = 0.0001

# read file
data_path = 'data/'
combo_pd = pd.read_csv(data_path+'bio-decagon-combo.csv')
ppi_pd = pd.read_csv(data_path+'bio-decagon-ppi.csv')
tarAll_pd = pd.read_csv(data_path+'bio-decagon-targets-all.csv')

# build vocab


def build_vocab(words):
    vocab = defaultdict(int)
    for word in words:
        if word not in vocab.keys():
            vocab[word] = len(vocab)
    return vocab

gene_list = list(ppi_pd['Gene 1'].unique()) + list(ppi_pd['Gene 2'].unique())
drug_list = list(combo_pd['STITCH 1'].unique()) + list(combo_pd['STITCH 2'].unique())
gene_list = gene_list[:1000]
drug_list = drug_list[:1000]
gene_vocab = build_vocab(gene_list)
drug_vocab = build_vocab(drug_list)

# stat
n_genes = len(gene_vocab)
n_drugs = len(drug_vocab)
n_drugdrug_rel_types = len(combo_pd['Polypharmacy Side Effect'].unique())
print('# of gene %d' % n_genes)
print('# of drug %d' % n_drugs)
print('# of rel_types %d' % n_drugdrug_rel_types)


def pk_save(obj, file_path):
    return pickle.dump(obj, open(file_path, 'wb'))


def pk_load(file_path):
    if os.path.exists(file_path):
        return pickle.load(open(file_path, 'rb'))
    else:
        return None

################# build gene-gene net #################
gene1_list, gene2_list = ppi_pd['Gene 1'].tolist(), ppi_pd['Gene 2'].tolist()
data_list, gene_idx1_list, gene_idx2_list = [], [], []
for u, v in zip(gene1_list, gene2_list):
    u, v = gene_vocab.get(u, -1), gene_vocab.get(v, -1)
    if u == -1 or v == -1:
        continue
    data_list.extend([1, 1])
    gene_idx1_list.extend([u, v])
    gene_idx2_list.extend([v, u])
gene_adj = sp.csr_matrix((data_list, (gene_idx1_list, gene_idx2_list)))
print('gene-gene / protein-protein adj: {}\t{}\tnumber of edges: {}'.format(type(gene_adj), gene_adj.shape,
                                                                            gene_adj.nnz))
logging.info('{} --- {}'.format(gene_adj[u, v], gene_adj[v, u]))
gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()
print()

################# build gene-drug net #################
stitch_list, gene_list = tarAll_pd['STITCH'].tolist(), tarAll_pd['Gene'].tolist()
data_list, drug_idx_list, gene_idx_list = [], [], []
for u, v in zip(stitch_list, gene_list):
    u, v = drug_vocab.get(u, -1), gene_vocab.get(v, -1)
    if u == -1 or v == -1:
        continue
    data_list.append(1)
    drug_idx_list.append(u)
    gene_idx_list.append(v)
gene_drug_adj = sp.csr_matrix((data_list, (gene_idx_list, drug_idx_list)))
drug_gene_adj = gene_drug_adj.transpose(copy=True)

# logging.info('gene_drug_adj: {}'.format(gene_drug_adj.shape))
# logging.info('drug_gene_adj: {}'.format(drug_gene_adj.shape))
# tv, tu = 219, 5618
# logging.info('In gene-drug adj: {}'.format(gene_drug_adj[tu, tv]))
# logging.info('In drug-gene adj: {}'.format(drug_gene_adj[tv, tu]))
# print()

################# build drug-drug net #################
drug_drug_adj_list = []
drug1_list, drug2_list, se_list = combo_pd['STITCH 1'].tolist(), combo_pd['STITCH 2'].tolist(), combo_pd[
    'Polypharmacy Side Effect'].tolist()
se_dict = {}
for u, v, se in zip(drug1_list, drug2_list, se_list):
    u, v = drug_vocab.get(u, -1), drug_vocab.get(v, -1)
    if u == -1 or v == -1:
        continue
    if se not in se_dict:
        se_dict[se] = {'row': [], 'col': [], 'data': []}
    se_dict[se]['row'].extend([u, v])
    se_dict[se]['col'].extend([v, u])
    se_dict[se]['data'].extend([1, 1])

for key, value in se_dict.iteritems():
    drug_drug_adj = sp.csr_matrix((value['data'], (value['row'], value['col'])), shape=(n_drugs, n_drugs))
    drug_drug_adj_list.append(drug_drug_adj)
    # print('Side Effect: {}'.format(key))
    # print('drug-drug network: {}\tedge number: {}'.format(drug_drug_adj.shape, drug_drug_adj.nnz))
logging.info('{} adjs with edges >= 500'.format(1098))

drug_drug_adj_list = sorted(drug_drug_adj_list, key=lambda x: x.nnz)[::-1][:964]
drug_drug_adj_list = drug_drug_adj_list[:10]
# drug_degree_list = map(lambda x: x.sum(axis=0).squeeze(), drug_drug_adj_list)
print('# of filtered rel_types:%d' % len(drug_drug_adj_list))
drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]
for i in range(10):
    logging.info('shape:{}\t{} match {}'.format(drug_drug_adj_list[i].shape, drug_drug_adj_list[i].nnz,
                                                np.sum(drug_degrees_list[i])))
print()
print('Done data loading')

# data representation
adj_mats_orig = {
    (0, 0): [gene_adj, gene_adj.transpose(copy=True)],
    (0, 1): [gene_drug_adj],
    (1, 0): [drug_gene_adj],
    (1, 1): drug_drug_adj_list + [x.transpose(copy=True) for x in drug_drug_adj_list],
}
degrees = {
    0: [gene_degrees, gene_degrees],
    1: drug_degrees_list + drug_degrees_list,
}

# featureless (genes)
gene_feat = sp.identity(n_genes)
gene_nonzero_feat, gene_num_feat = gene_feat.shape
gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

# features (drugs)
drug_feat = sp.identity(n_drugs)
drug_nonzero_feat, drug_num_feat = drug_feat.shape
drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

# data representation
num_feat = {
    0: gene_num_feat,
    1: drug_num_feat,
}
nonzero_feat = {
    0: gene_nonzero_feat,
    1: drug_nonzero_feat,
}
feat = {
    0: gene_feat,
    1: drug_feat,
}

edge_type2dim = {k: [adj.shape for adj in adjs]
                 for k, adjs in adj_mats_orig.items()}
edge_type2decoder = {
    (0, 0): 'bilinear',
    (0, 1): 'bilinear',
    (1, 0): 'bilinear',
    (1, 1): 'dedicom',
}

edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
num_edge_types = sum(edge_types.values())
print("Edge types:", "%d" % num_edge_types)

# ##########################################################
#
# Settings and placeholders
#
# ##########################################################

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0,
                   'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('bias', True, 'Bias term.')
# Important -- Do not evaluate/print validation performance every iteration as it can take
# substantial amount of time
PRINT_PROGRESS_EVERY = 150

print("Defining placeholders")
placeholders = construct_placeholders(edge_types)

###########################################################
#
# Create minibatch iterator, model and optimizer
#
###########################################################

print("Create minibatch iterator")
minibatch = EdgeMinibatchIterator(
    adj_mats=adj_mats_orig,
    feat=feat,
    edge_types=edge_types,
    batch_size=FLAGS.batch_size,
    val_test_size=val_test_size
)

print("Create model")
model = DecagonModel(
    placeholders=placeholders,
    num_feat=num_feat,
    nonzero_feat=nonzero_feat,
    edge_types=edge_types,
    decoders=edge_type2decoder,
)

print("Create optimizer")
with tf.name_scope('optimizer'):
    opt = DecagonOptimizer(
        embeddings=model.embeddings,
        latent_inters=model.latent_inters,
        latent_varies=model.latent_varies,
        degrees=degrees,
        edge_types=edge_types,
        edge_type2dim=edge_type2dim,
        placeholders=placeholders,
        batch_size=FLAGS.batch_size,
        margin=FLAGS.max_margin
    )

print("Initialize session")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
feed_dict = {}

##########################################################

# Train model

##########################################################

print("Train model")
for epoch in range(FLAGS.epochs):

    minibatch.shuffle()
    itr = 0
    while not minibatch.end():
        # Construct feed dictionary
        feed_dict = minibatch.next_minibatch_feed_dict(
            placeholders=placeholders)
        feed_dict = minibatch.update_feed_dict(
            feed_dict=feed_dict,
            dropout=FLAGS.dropout,
            placeholders=placeholders)

        t = time.time()

        # Training step: run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.batch_edge_type_idx], feed_dict=feed_dict)
        train_cost = outs[1]
        batch_edge_type = outs[2]

        # if itr % PRINT_PROGRESS_EVERY == 0:
        if itr % 1 == 0:
            val_auc, val_auprc, val_apk = get_accuracy_scores(
                minibatch.val_edges, minibatch.val_edges_false,
                minibatch.idx2edge_type[minibatch.current_edge_type_idx])

            print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1), "Edge:", "%04d" % batch_edge_type,
                  "train_loss=", "{:.5f}".format(train_cost),
                  "val_roc=", "{:.5f}".format(
                      val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
                  "val_apk=", "{:.5f}".format(val_apk), "time=", "{:.5f}".format(time.time() - t))

        itr += 1

print("Optimization finished!")

for et in range(num_edge_types):
    roc_score, auprc_score, apk_score = get_accuracy_scores(
        minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[et])
    print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[et])
    print("Edge type:", "%04d" %
          et, "Test AUROC score", "{:.5f}".format(roc_score))
    print("Edge type:", "%04d" %
          et, "Test AUPRC score", "{:.5f}".format(auprc_score))
    print("Edge type:", "%04d" %
          et, "Test AP@k score", "{:.5f}".format(apk_score))
    print()
