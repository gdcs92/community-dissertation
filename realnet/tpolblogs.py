# coding: utf-8
"""
tpolblogs.py
"""
import numpy as np
import networkx as nx
import pandas as pd
import argparse
import time

from algoritmos import *
from networkx.algorithms.community import modularity,greedy_modularity_communities,is_partition

parser = argparse.ArgumentParser()
parser.add_argument('-R', default=10, type=int,
	                help='numero de repeticoes')
parser.add_argument('--nrounds', default=10, type=int)
parser.add_argument('--seed', default=1, type=int, help='seed do random')
parser.add_argument('--latex', action='store_true')

args = parser.parse_args()
R = args.R
sd = args.seed
nr = args.nrounds
latex = args.latex

# Fixar as fontes de aleatoriedade
numpy_seed = sd 
np.random.seed(numpy_seed)

## LEITURA E PRE-PROCESSAMENTO DA REDE 
G = nx.read_gml("polblogs.gml")
print("Leu o arquivo polblogs.gml")
# Crio um novo atributo com o id de cada no como esta no .gml
for i,u in enumerate(G.nodes):
    G.nodes[u]['id'] = i+1
# Converte G para grafo nao-direcionado sem multiarestas,
# 'condensando' arestas bi-direcionais e multiarestas, mas nao
# remove self-loops
G = nx.Graph(G)
# Remove self-loops
for u in G.nodes:
    if G.has_edge(u,u):
        G.remove_edge(u,u)

CC = nx.connected_components(G)
largest_cc = max(CC, key=len)
# Pega o subgrafo da maior componente conexa
Gc = G.subgraph(largest_cc)
# Converte os labels para inteiros para facilitar a manipulacao
# Os labels originais sao salvos no node attribute 'url'
# e podem ser acessados com rG.nodes[v]['url']
G = nx.convert_node_labels_to_integers(Gc, label_attribute='url')

print("G.number_of_nodes():", len(G))
print("G.number_of_edges():", G.number_of_edges())
print("G.is_connected():", nx.is_connected(G))
print()

# Definicao do ground truth
N = len(G)
gt = np.zeros(N, dtype=int)
for v in range(N):
    gt[v] = G.nodes[v]['value']

print(r'# 1-labelled nodes:', sum(gt))
print(r'# 0-labelled nodes:', sum(1-gt))

##### FIM DO PRE-PROCESSAMENTO #####

algoritmos = ['spectral','mva','gam','hard','soft']
columns = ['acuracia','onelabels','acvfixos','niter','ciclo','pvfixos','tempo']

T = {}
for a in algoritmos:
    D = {}
    for c in columns:
        D[c] = np.zeros(R)

    for r in range(R):
        t1 = time.time()
        if a == 'spectral':
            results = spectral_method(G)
        if a == 'gam':
            results = gam_method(G)
        elif a == 'mva':
            results = mva_method(G)
        elif a in ['hard', 'soft']:
            results = bootstrap_method(G,restart=a,nrounds=nr)
        t2 = time.time()
        D['tempo'][r] = t2 - t1

        labels = results['labels'].astype(int, copy=False)
        D['acuracia'][r] = acc(labels,gt)
        D['onelabels'][r] = sum(labels)/N
        if a != 'spectral':
            D['ciclo'][r] = results['ciclo']
            D['niter'][r] = results['niter']
            vfixos = results['vfixos']; pvfixos = sum(vfixos)/N
            D['pvfixos'][r] = pvfixos 
            if pvfixos > 0: 
                D['acvfixos'][r] = acc(labels[vfixos], gt[vfixos])
            else:
                D['acvfixos'][r] = np.nan

    T[a] = {}
    for c in D:
        x = D[c]
        T[a][c] = [np.min(x), np.max(x), np.mean(x), np.std(x,ddof=1)] 

#main_time = t2 - t1
#print("Wall-clock time: ", main_time)
algnames = {'spectral': 'Spectral', 'mva': 'MVA', 'gam': 'GAM',
    'hard': 'hard GAMB', 'soft': 'soft GAMB'
}

print()
for a in algoritmos:
    line = ['{:.2f}'.format(i) for i in T[a]['acuracia']]
    line = line + ['{:.4f}'.format(i) for i in T[a]['tempo']]
    line = algnames[a] + ' & ' + ' & '.join(line)
    print(line)

