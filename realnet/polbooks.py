# coding: utf-8
"""
polbooks.py
"""
import networkx as nx

rG = nx.read_gml("polbooks.gml")
nodes = list(rG.nodes)
num_edges = rG.number_of_edges()
print("rG.number_of_nodes():", len(rG))
print("rG.number_of_edges():", num_edges)

# Crio um novo atributo com o id de cada no como esta no .gml
for i,u in enumerate(rG.nodes):
    rG.nodes[u]['id'] = i

# Converte rG para grafo nao-direcionado sem multiarestas,
# 'condensando' arestas bi-direcionais e multiarestas, mas nao
# remove self-loops
G = nx.Graph(rG)
print("G.number_of_edges():", G.number_of_edges())
print("num_self_loops(G):", nx.number_of_selfloops(G))
# Remove self-loops
print("Removendo self-loops...")
for u in G.nodes:
    if G.has_edge(u,u):
        G.remove_edge(u,u)
print("G.number_of_edges():", G.number_of_edges())

CC = nx.connected_components(G)
largest_cc = max(CC, key=len)
print(len(largest_cc))

# Pega o subgrafo da maior componente conexa
print("Pegando a maior componente conexa...")
Gc = G.subgraph(largest_cc)
print("Gc.number_of_nodes:", len(Gc))
print("Gc.number_of_edges():", Gc.number_of_edges())
G = Gc

# Converte os labels para inteiros para facilitar a manipulacao
# Os labels originais sao salvos no node attribute 'url'
# e podem ser acessados com rG.nodes[v]['url']
G = nx.convert_node_labels_to_integers(G, label_attribute='label')

print("G.number_of_nodes():", len(G))
print("G.number_of_edges():", G.number_of_edges())
#for i,v in enumerate(G.nodes):
#    print(i,v, G.nodes[v]['id'], G.nodes[v]['url'], G.degree[v])

print("Pre-processamento de G pronto!")
### PRONTO!!!! ###

import numpy as np
from algoritmos import *
from networkx.algorithms.community import is_partition,modularity

# Representar as comunidades num vetor
n = len(G)
gt = np.zeros(n,dtype=int)
for v in G.nodes:
#    print(v, G.nodes[v]['value'])
    if G.nodes[v]['value'] == 'c': # conservative
        gt[v] = 1
    elif G.nodes[v]['value'] == 'l': # liberal
        gt[v] = 0
    else:
        gt[v] = 2 # 'n' == neutro?

print(r'# 1-labelled nodes (conservatives):', sum(gt == 1))
print(r'# 0-labelled nodes (liberals):', sum(gt == 0))

results = spectral_method(G)
spec_s = results['labels'].astype(int, copy=False)
Cspec = [[], []]
for i,ci in enumerate(spec_s): Cspec[ci].append(i)
print("Accuracy of spectral method:", acc(spec_s,gt))
if is_partition(G,Cspec):
    print("Spectral modularity:", modularity(G,Cspec))
print()

results = mva_method(G, maxiter=200)
mva_s = results['labels'].astype(int, copy=False)
Cmva = [[], []]
for i,ci in enumerate(mva_s): Cmva[ci].append(i)
print("Accuracy of mva method:", acc(mva_s,gt))
if is_partition(G,Cmva):
    print("MVA modularity:", modularity(G,Cmva))
print()

results = gam_method(G, maxiter=200)
gam_s = results['labels'].astype(int, copy=False)
Cgam = [[], []]
for i,ci in enumerate(gam_s): Cgam[ci].append(i)
print("Accuracy of gam method:", acc(gam_s,gt))
if is_partition(G,Cgam):
    print("GAM modularity:", modularity(G,Cgam))
print()

results = bootstrap_method(G,'soft',nrounds=10)
soft_s = results['labels'].astype(int, copy=False)
Csoft = [[], []]
for i,ci in enumerate(soft_s): Csoft[ci].append(i)
print("Accuracy of soft GAMB method:", acc(soft_s,gt))
if is_partition(G,Csoft):
    print("soft GAMB modularity:", modularity(G,Csoft))
print()

results = bootstrap_method(G,'hard',nrounds=10)
hard_s = results['labels'].astype(int, copy=False)
Chard = [[], []]
for i,ci in enumerate(hard_s): Chard[ci].append(i)
print("Accuracy of hard GAMB method:", acc(gam_s,gt))
if is_partition(G,Chard):
    print("hard GAMB modularity:", modularity(G,Chard))
