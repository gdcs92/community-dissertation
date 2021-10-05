# coding: utf-8
"""
karate.py
"""
import numpy as np
import networkx as nx
import pandas as pd
import argparse
import time

from algoritmos import *
from networkx.algorithms.community import modularity,greedy_modularity_communities,is_partition

parser = argparse.ArgumentParser()
parser.add_argument('-R', default=10, type=int, help='numero de repeticoes')
parser.add_argument('-a', type=str, help='algoritmo: mva ou gam')
parser.add_argument('--nrounds', default=10, type=int)
parser.add_argument('--seed', default=1, type=int, help='seed do random')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--latex', action='store_true')

args = parser.parse_args()
R = args.R
a = args.a
sd = args.seed
verbose = args.verbose
nr = args.nrounds
latex = args.latex

if a not in ['mva', 'gam', 'hard', 'soft']:
    raise ValueError("a not in ['mva', 'gam', 'hard', 'soft']")

# Fixar as fontes de aleatoriedade
numpy_seed = sd 
np.random.seed(numpy_seed)

G = nx.karate_club_graph()

# Definicao do ground truth
N = len(G)
gt = np.zeros(N, dtype=int)
for v in range(N):
    if G.nodes[v]['club'] == 'Mr. Hi':
        gt[v] = 0
    else:
        gt[v] = 1 # 'Officer'

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
