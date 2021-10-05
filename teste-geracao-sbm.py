# coding: utf-8
"""
teste-geracao-sbm.py
"""
import numpy as np
import networkx as nx
import random
import argparse
import pickle
import time
#from timeit import default_timer as timer
from community import *
from algoritmos import *

parser = argparse.ArgumentParser()
parser.add_argument('-n', default=50, type=int,
	                help='numero de vertices')
parser.add_argument('-p', type=float,
                    help='probabilidade de aresta intra-comunidade')
parser.add_argument('-d', type=float)
parser.add_argument('-M', type=int,
                    help='numero de instancias')
parser.add_argument('-T', type=int, default=50,
                    help='numero maximo de tentativas para gerar SBM conexo')
parser.add_argument('--seed', default=None, type=int,
	                help='seed do random para gerar o SBM')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()
n = args.n
p = args.p
d = args.d
M = args.M
T = args.T
sbm_seed = args.seed
verbose = args.verbose

# Fixar as fontes de aleatoriedade
numpy_seed = sbm_seed
np.random.seed(numpy_seed)
random.seed(sbm_seed) # stochastic block model

q = p-d

print(f'n = {n:d}, p = {p:.4f}, d = {d:.4f}, M = {M:d}, seed = {sbm_seed:d}\n')

for i in range(M):
    num_tentativas = 0
    print(f"Instancia {i:d}")
    tstart = time.time()
    while num_tentativas < T:
        num_tentativas += 1
        print(f" - Tentativa {num_tentativas:d}:")
        G = nx.planted_partition_graph(2,n,p,q)
        m = G.number_of_edges()
        m_in = 0; m_out = 0
        for u,v in G.edges:
            if (u<n and v<n) or (u>=n and v>=n):
                m_in += 1
            else:
                m_out +=1
        nc = nx.number_connected_components(G)
        conn = (nc == 1)
        print(f'    m_in = {m_in:d}, m_out = {m_out:d}, m = {m:d}')
        print(f'    number of connected components = {nc:d}')
        print()
        if conn:
            break
    tend = time.time()
    print("    num_tentativas = {:d}, tempo = {:f}".format(num_tentativas,tend-tstart))
