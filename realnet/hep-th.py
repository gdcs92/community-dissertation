# coding: utf-8
"""
hep-th.py
"""
import networkx as nx
import graph_tool.all as gt

G = nx.read_gml("hep-th.gml")
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

# Converte os labels para inteiros para facilitar a manipulacao
# Os labels originais sao salvos no node attribute 'name'
# e podem ser acessados com rG.nodes[v]['name']
G = nx.convert_node_labels_to_integers(G, label_attribute='name')

print("G.number_of_nodes():", len(G))
print("G.number_of_edges():", G.number_of_edges())
#for i,v in enumerate(G.nodes):
#    print(i,v, G.nodes[v]['name'], G.degree[v])

print("Pre-processamento de G pronto!")
### PRONTO!!!! ###

import numpy as np
from algoritmos import *
from networkx.algorithms.community import modularity, greedy_modularity_communities

# Como nao tenho ground truth para esta rede, a principio vou
# tomar como base a particao fornecida pelo algoritmo de Kernighan-Lin

N = len(G)

C = list(nx.community.greedy_modularity_communities(G))
#gt = np.zeros(N,dtype=int)
#for i,ci in enumerate(C):
#    for v in ci:
#        gt[v] = i
print("Number of communities (fastgreedy):", len(C))
print("Fastgreedy modularity: ", modularity(G,C))

#print(r'# 1-labelled nodes:', sum(gt))
#print(r'# 0-labelled nodes:', sum(1-gt))

results = mva_method(G, maxiter=200)
mva_s = results['labels'].astype(int, copy=False)
Cmva = [[], []]
for i,ci in enumerate(mva_s): Cmva[ci].append(i)
#print("Accuracy of mva method:", acc(mva_s,gt))
print("MVA modularity:", modularity(G,Cmva))

results = gam_method(G, maxiter=200)
gam_s = results['labels'].astype(int, copy=False)
Cgam = [[], []]
for i,ci in enumerate(gam_s): Cgam[ci].append(i)
#print("Accuracy of gam method:", acc(gam_s,gt))
print("GAM modularity:", modularity(G,Cgam))
