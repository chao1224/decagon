from __future__ import division
from __future__ import print_function

import logging
logging.basicConfig(level=logging.INFO, format='INFO: %(message)s')
  
import numpy as np
import networkx as nx
import scipy.sparse as sp
import pandas as pd
from sklearn import metrics


def build_vocab(words):
  vocab, cnt = {}, 0
  for word in words:
    if word not in vocab:
      vocab[word] = cnt
      cnt += 1
  return vocab


def load_data(data_dir='./data/'):
  combo_pd = pd.read_csv(data_dir+'bio-decagon-combo.csv')
  ppi_pd = pd.read_csv(data_dir+'bio-decagon-ppi.csv')
  tarAll_pd = pd.read_csv(data_dir+'bio-decagon-targets-all.csv')
  logging.info('combo_pd columns: {}'.format(combo_pd.columns))
  logging.info('ppi_pd columns: {}'.format(ppi_pd.columns))
  logging.info('tarAll_pd columns: {}'.format(tarAll_pd.columns))
  print()
  return combo_pd, ppi_pd, tarAll_pd


def build_adj(combo_pd, ppi_pd, tarAll_pd, drug_vocab, gene_vocab, se_vocab):
  drug_size = len(drug_vocab)
  gene_size = len(gene_vocab)

  ################# build gene-gene net #################
  gene1_list, gene2_list = ppi_pd['Gene 1'].tolist(), ppi_pd['Gene 2'].tolist()
  data_list, gene_idx1_list, gene_idx2_list = [], [], []
  for u, v in zip(gene1_list, gene2_list):
    u, v = gene_vocab[u], gene_vocab[v]
    data_list.extend([1, 1])
    gene_idx1_list.extend([u, v])
    gene_idx2_list.extend([v, u])
  gene_gene_adj = sp.csr_matrix((data_list, (gene_idx1_list, gene_idx2_list)))
  gene_degrees = np.array(gene_gene_adj.sum(axis=0)).squeeze()
  print('gene-gene / protein-protein adj: {}\t{}\tnumber of edges: {}'.format(type(gene_gene_adj), gene_gene_adj.shape, gene_gene_adj.nnz))
  logging.info('{} --- {}'.format(gene_gene_adj[u, v], gene_gene_adj[v, u]))
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
  
  logging.info('gene_drug_adj: {}'.format(gene_drug_adj.shape))
  logging.info('drug_gene_adj: {}'.format(drug_gene_adj.shape))
  tv, tu = 219, 5618
  logging.info('In gene-drug adj: {}'.format(gene_drug_adj[tu, tv]))
  logging.info('In drug-gene adj: {}'.format(drug_gene_adj[tv, tu]))
  print()


  ################# build drug-drug net #################
  drug_drug_adj_list = []
  drug1_list, drug2_list, se_list = combo_pd['STITCH 1'].tolist(), combo_pd['STITCH 2'].tolist(), combo_pd['Polypharmacy Side Effect'].tolist()
  se_dict = {}
  for u,v,se in zip(drug1_list, drug2_list, se_list):
    u, v = drug_vocab[u], drug_vocab[v]
    if se not in se_dict:
      se_dict[se] = {'row': [], 'col': [], 'data': []}
    se_dict[se]['row'].extend([u, v])
    se_dict[se]['col'].extend([v, u])
    se_dict[se]['data'].extend([1, 1])
  
  for key, value in se_dict.iteritems():
    drug_drug_adj = sp.csr_matrix((value['data'], (value['row'], value['col'])), shape=(drug_size, drug_size))
    drug_drug_adj_list.append(drug_drug_adj)
    #print('Side Effect: {}'.format(key))
    #print('drug-drug network: {}\tedge number: {}'.format(drug_drug_adj.shape, drug_drug_adj.nnz))
  logging.info('{} adjs with edges >= 500'.format(1098))
  
  drug_drug_adj_list = sorted(drug_drug_adj_list, key=lambda x: x.nnz)[::-1][:964]
  drug_degree_list = map(lambda x: x.sum(axis=0).squeeze(), drug_drug_adj_list)
  for i in range(10):
    logging.info('shape:{}\t{} match {}'.format(drug_drug_adj_list[i].shape, drug_drug_adj_list[i].nnz, np.sum(drug_degree_list[i], axis=1)[0,0]))
  print()
  print('Done data loading')
  
  return gene_gene_adj, gene_degrees, gene_drug_adj, drug_gene_adj, drug_drug_adj_list, drug_degree_list


if __name__ == '__main__':
  # read file
  combo_pd, ppi_pd, tarAll_pd = load_data()
  
  drug_vocab = build_vocab(list(combo_pd['STITCH 1'].unique()) + list(combo_pd['STITCH 2'].unique()))
  drug_size = len(drug_vocab)
  print('size of drug vocab: {}'.format(drug_size))
  
  gene_vocab = build_vocab(list(ppi_pd['Gene 1'].unique()) + list(ppi_pd['Gene 2'].unique()))
  gene_size = len(gene_vocab)
  print('size of gene vocab: {}'.format(gene_size))

  se_vocab = build_vocab(combo_pd['Polypharmacy Side Effect'].tolist())
  print('size of side effect vocab: {}'.format(len(se_vocab)))
  
  gene_gene_adj, gene_degrees, gene_drug_adj, drug_gene_adj, drug_drug_adj_list, drug_degree_list = build_adj(combo_pd, ppi_pd, tarAll_pd, drug_vocab, gene_vocab, se_vocab)
  