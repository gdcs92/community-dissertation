# coding: utf-8
"""
karate.py
"""
import networkx as nx
import graph_tool.all as gt

G = nx.karate_club_graph()
nodes = list(G.nodes)
num_edges = G.number_of_edges()
print("rG.number_of_nodes():", len(G))
print("rG.number_of_edges():", num_edges)
print("Directed?:", nx.is_directed(G))

num_selfloops = nx.number_of_selfloops(G)
print("Number of self loops:", num_selfloops)
# Remove self-loops
print("Removing self-loops...")
for u in G.nodes:
    if G.has_edge(u,u):
        G.remove_edge(u,u)

CC = list(nx.connected_components(G))
print("Number of connected components:", len(CC))
largest_cc = max(CC, key=len)
print("Size of the largest cc:", len(largest_cc))

# Pega o subgrafo da maior componente conexa
#print("Pegando a maior componente conexa...")
Gc = G.subgraph(largest_cc)
#print("Gc.number_of_nodes:", len(Gc))
#print("Gc.number_of_edges():", Gc.number_of_edges())
G = Gc

print("G.number_of_nodes():", len(G))
print("G.number_of_edges():", G.number_of_edges())
#for i,v in enumerate(G.nodes):
#    print(i,v, G.nodes[v]['name'], G.degree[v])

print("Pre-processamento de G pronto!")
### PRONTO!!!! ###

import numpy as np
from algoritmos import *
from networkx.algorithms.community import modularity,greedy_modularity_communities,is_partition

# Como nao tenho ground truth para esta rede, a principio vou
# tomar como base a particao fornecida pelo algoritmo de Kernighan-Lin

N = len(G)

# Definicao do ground truth
gt = np.zeros(N, dtype=int)
for v in range(N):
    if G.nodes[v]['club'] == 'Mr. Hi':
        gt[v] = 0
    else:
        gt[v] = 1 # 'Officer'

print(r'# 1-labelled nodes:', sum(gt))
print(r'# 0-labelled nodes:', sum(1-gt))

C = list(nx.community.greedy_modularity_communities(G))
#gt = np.zeros(N,dtype=int)
#for i,ci in enumerate(C):
#    for v in ci:
#        gt[v] = i
print("Number of communities (fastgreedy):", len(C))
print("Fastgreedy modularity: ", modularity(G,C))

results = mva_method(G, maxiter=200)
mva_s = results['labels'].astype(int, copy=False)
Cmva = [[], []]
for i,ci in enumerate(mva_s): Cmva[ci].append(i)

print("* MVA method")
for v in G.nodes:
    print(v, gt[v], mva_s[v])
print()
print("Accuracy of mva method:", acc(mva_s,gt))
print("MVA modularity:", modularity(G,Cmva))

results = gam_method(G, maxiter=200)
gam_s = results['labels'].astype(int, copy=False)
Cgam = [[], []]
for i,ci in enumerate(gam_s): Cgam[ci].append(i)
print("* GAM method")
for v in G.nodes:
    print(v, gt[v], gam_s[v])
print()
print("Accuracy of gam method:", acc(gam_s,gt))
print("GAM modularity:", modularity(G,Cgam))

exit()

results = bootstrap_method(G,'hard',nrounds=10)
hard_s = results['labels'].astype(int, copy=False)
Chard = [[], []]
for i,ci in enumerate(hard_s): Chard[ci].append(i)
print("Accuracy of hard GAMB method:", acc(hard_s,gt))
if is_partition(G,Chard):
    print("hard GAMB modularity:", modularity(G,Chard))

results = bootstrap_method(G,'soft',nrounds=10)
soft_s = results['labels'].astype(int, copy=False)
Csoft = [[], []]
for i,ci in enumerate(soft_s): Csoft[ci].append(i)
print("Accuracy of soft GAMB method:", acc(soft_s,gt))
if is_partition(G,Csoft):
    print("soft GAMB modularity:", modularity(G,Csoft))

