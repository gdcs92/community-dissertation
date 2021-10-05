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
parser.add_argument('--qmin', type=float,
	                help='dvals = np.linspace(0,p-qmin,num=10)')
parser.add_argument('--seed', default=None, type=int,
	                help='seed do random para gerar o SBM')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()
n = args.n; p = args.p
qmin = args.qmin
sbm_seed = args.seed
verbose = args.verbose

# Fixar as fontes de aleatoriedade
numpy_seed = sbm_seed
np.random.seed(numpy_seed)
random.seed(sbm_seed) # stochastic block model

dvals = np.linspace(0,p-qmin,num=10)

tentativas = np.zeros(len(dvals), dtype=int)
tempo = np.zeros(len(dvals))

for i,d in enumerate(dvals):
    q = p-d
    num_tentativas = 0
    print("i={:d}, n={:d}, p={:f}, q={:f}".format(i,n,p,q))
    tstart = time.time()
    while 1:
        num_tentativas += 1
#        print("num_tentativas: {:d}".format(num_tentativas))
        G = nx.planted_partition_graph(2,n,p,q)
        if nx.is_connected(G):
            break
    tend = time.time()
    tentativas[i] = num_tentativas
    tempo[i] = tend-tstart

    print("- tentativas = {:d}, tempo = {:f}".format(tentativas[i],tempo[i]))
