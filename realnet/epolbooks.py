# coding: utf-8
"""
epolbooks.py
"""
import numpy as np
import networkx as nx
import pandas as pd
import argparse
import time
from algoritmos import *
from networkx.algorithms.community import modularity, is_partition

parser = argparse.ArgumentParser()
parser.add_argument('-R', default=10, type=int,
	                help='numero de repeticoes')
parser.add_argument('-a', type=str, help='algoritmo: mva, gam, hard ou soft')
parser.add_argument('--nrounds', default=10, type=int)
parser.add_argument('--seed', default=1, type=int, help='seed do random')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--latex', action='store_true')
parser.add_argument('--estrategia', type=int)

args = parser.parse_args()
R = args.R
a = args.a
sd = args.seed
verbose = args.verbose
nr = args.nrounds
latex = args.latex
estrategia = args.estrategia

if a not in ['mva', 'gam', 'hard', 'soft']:
    raise ValueError("a not in ['mva', 'gam', 'hard', 'soft']")

# Fixar as fontes de aleatoriedade
numpy_seed = sd 
np.random.seed(numpy_seed)

#### INICIO DA LEITURA E PRE-PROCESSAMENTO DOS DADOS ####
G = nx.read_gml("polbooks.gml")
print("Leu o arquivo polbooks.gml")
# Crio um novo atributo com o id de cada no como esta no .gml
for i,u in enumerate(G.nodes):
    G.nodes[u]['id'] = i

# Converte os labels para inteiros para facilitar a manipulacao
# Os labels originais sao salvos no node attribute 'title'
# e podem ser acessados com G.nodes[v]['title']
G = nx.convert_node_labels_to_integers(G, label_attribute='title')

print("G.number_of_nodes():", len(G))
print("G.number_of_edges():", G.number_of_edges())

# Representar as comunidades num vetor
N = len(G)
gt = np.zeros(N,dtype=int)
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
print(r'# 2-labelled nodes (neutral):', sum(gt == 2))

print("Finished pre-processing.")
#### PRE PROCESSAMENTO PRONTO!!!! ####

#### ESTRATEGIA DE PROJECAO DA TERCEIRA CLASSE ####
if estrategia == 1: # Subgrafo induzido por 0 e 1
    print("Estrategia de projecao 1: subgrafo induzido por 0 ('c') e 1 ('l')")
    proj_nodes = np.nonzero(gt != 2)[0]
elif estrategia == 2:
    print("Estrategia de projecao 2: subgrafo induzido por 0 ('c') e 2 ('n')")
    proj_nodes = np.nonzero(gt != 1)[0]
elif estrategia == 3:
    print("Estrategia de projecao 3: subgrafo induzido por 2 ('n') e 1 ('l')")
    proj_nodes = np.nonzero(gt != 0)[0]

PG = G.subgraph(proj_nodes)
#PG = nx.convert_node_labels_to_integers(PG)
#for i,v in enumerate(PG.nodes):
#    print(i, v, PG.nodes[v]['id'])

#### Tomando a maior componente conexa
CC = list(nx.connected_components(PG))
print("# componentes conexas do grafo projetado:", len(CC))
largest_cc = max(CC, key=len)
print("Tamanho da maior c.c:", len(largest_cc))
Gc = G.subgraph(largest_cc)
Gc = nx.convert_node_labels_to_integers(Gc)
print("Gc.number_of_nodes:", len(Gc))
print("Gc.number_of_edges():", Gc.number_of_edges())

#### Projetando vetor ground truth
pgt = np.zeros(len(Gc), dtype=int)
for i in Gc.nodes:
    v = Gc.nodes[i]['id']
    if estrategia == 1: # 'c': 0, 'l': 1
        pgt[i] = gt[v]
    elif estrategia == 2:
        pgt[i] = int(gt[v] == 2) # 'c': 0, 'n': 1
    elif estrategia == 3:
        pgt[i] = int(gt[v] == 1) # 'n': 0, 'l': 1
#for i,v in enumerate(Gc.nodes):
#    if estrategia == 1: # 'c': 0, 'l': 1
#        pgt[i] = gt[v] 
#    elif estrategia == 2:
#        pgt[i] = int(gt[v] == 2) # 'c': 0, 'n': 1
#    elif estrategia == 3:
#        pgt[i] = int(gt[v] == 1) # 'n': 0, 'l': 1
#for i,v in enumerate(PG.nodes):
#    print(i, v, pgt[i])
#print(len(proj_nodes))

gt = pgt
G = Gc
print("G.number_of_nodes():", len(G))
print("G.number_of_edges():", G.number_of_edges())
print("G.is_connected():", nx.is_connected(G))
print()
##### FIM DA PROJECAO #####


print(r'# 1-labelled nodes:', sum(gt))
print(r'# 0-labelled nodes:', sum(1-gt))

columns = ['acuracia','onelabels','acvfixos','niter','ciclo','pvfixos']

D = {}
for c in columns:
    D[c] = np.zeros(R)

t1 = time.time()
for r in range(R):
    if verbose:
        print(f"r: {r:d}") 

    if a == 'gam':
        results = gam_method(G)
    elif a == 'mva':
        results = mva_method(G)
    elif a in ['hard', 'soft']:
        results = bootstrap_method(G,restart=a,nrounds=nr)

    labels = results['labels'].astype(int, copy=False)
    D['acuracia'][r] = acc(labels,gt)
    D['onelabels'][r] = sum(labels)/N
    D['ciclo'][r] = results['ciclo']
    D['niter'][r] = results['niter']
    vfixos = results['vfixos']; pvfixos = sum(vfixos)/N
    D['pvfixos'][r] = pvfixos 
    if pvfixos > 0: 
        D['acvfixos'][r] = acc(labels[vfixos], gt[vfixos])
    else:
        D['acvfixos'][r] = np.nan

t2 = time.time()

main_time = t2 - t1
print("Wall-clock time: ", main_time)

pd.set_option('float_format', '{:.3f}'.format)
df = pd.DataFrame(data=D)
df['niter'] = df['niter'].astype(int)
df['ciclo'] = df['ciclo'].astype(int)

if R <= 10: print(df)
print()
dfstats = df.describe()
if latex:
    print(dfstats.to_latex())
else:
    print(dfstats)
