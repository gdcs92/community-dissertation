# coding: utf-8
"""
g3_visu.py
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import random
import pickle

from community import *
from algoritmos import *
from algoritmos import _hillclimb_init

parser = argparse.ArgumentParser()
parser.add_argument('-n', default=50, type=int,
	                help='numero de vertices')
parser.add_argument('-p', type=float,
                    help='probabilidade de aresta intra-comunidade')
parser.add_argument('-d', type=float,
                    help='diferenca p-q')
parser.add_argument('-M', default=10, type=int,
	                help='numero de instancias')
parser.add_argument('-R', default=10, type=int,
	                help='numero de repeticoes para cada instancia')
parser.add_argument('--nrounds', default=5, type=int,
	                help='numero de rounds do bootstrap')
parser.add_argument('--restart', type=str,
	                help='restart to bootstrap: hard ou soft')
parser.add_argument('--seed', type=int,
	                help='seed do random para gerar o SBM')
parser.add_argument('--save', action='store_true',
					help='salvar graficos em arquivos .pdf')

args = parser.parse_args()
n = args.n; 
p = args.p; d = args.d; q = p - d
R = args.R; M = args.M
save = args.save
restart = args.restart
nrounds = args.nrounds
seed = args.seed

## Inicializacao dos parametros do experimento
#nrounds = 15
T = nrounds + 1

columns = ['acuracia','ciclo','niter','pvfixos','acvfixos']
column_labels = {
	'acuracia': 'accuracy', 
    'niter': 'iterations',
    'ciclo': 'cycle length',
	'pvfixos': 'fraction of fixed vertices',
    'acvfixos': 'accuracy of fixed vertices',
}

## Leitura dos dados pickle
D = {}

ps = f'{p:.4f}'[2:]; qs = f'{q:.4f}'[2:]

file_loc = './data/'
filename = f"g3_n{n:d}_p{ps:s}_q{qs:s}_R{R:d}_M{M:d}_{restart:s}_s{seed:d}.data" 

fileObj = open(file_loc + filename, 'rb')
simdata = pickle.load(fileObj)
fileObj.close()

columns = simdata['columns']
for c in columns:
    D[c] = simdata['D'][c]
    
#    for a,c in ((a,c) for a in algorithms for c in columns):
#        D[(a,c)][f] = simdata[f]['D'][(a,c)]
#
#newD = {}
#for a,c in ((a,c) for a in algorithms for c in columns):
#    newD[(a,c)] = [[] for t in range(T)]
#    for t in range(T):
#        aa = []
#        for f in filenames:
#            aft = D[(a,c)][f][t,:,:].flatten()
#            aa.append(aft)
#        newD[(a,c)][t] = np.concatenate(aa) 

newD = {}
for c in columns:
    newD[c] = [[] for t in range(T)]
    for t in range(T):
        newD[c][t] = D[c][t,:,:].flatten() 

## remocao de valores ruins

goodvals = [v>0 for v in newD['ciclo']]

for c in columns:
    for i in range(len(newD[c])):
        newD[c][i] = [x for j,x in enumerate(newD[c][i]) if goodvals[i][j] > 0]

#for a in algorithms:
#    goodvals = [v>0 for v in newD[(a,'ciclo')]]
#    for c in columns:
#        k = (a,c)
#        for i in range(len(newD[k])):
#            newD[k][i] = [x for j,x in enumerate(newD[k][i]) if goodvals[i][j] > 0]


#if c == 'acvfixos':
#    for t,v in enumerate(vals):
#        vals[t] = [x for x in v if x > 0]


## Visualizacao estatistica
#
for c in columns:
        fig, ax = plt.subplots(figsize=(6,5))
        ax.set_title(f'{restart:s} bootstrapping',
            fontsize=14)
        vals = newD[c] #[D_t.flatten() for D_t in D[k]]

        ylabel = column_labels[c]
        ax.boxplot(vals, showmeans=True, whis='range') #whis=1000) 

        ax.set_xlabel('bootstrap round', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xticklabels([str(x) for x in list(range(nrounds+1))])
        ax.tick_params(axis='both', labelsize=12)

        if c in ['acuracia','pvfixos','acvfixos']:
            ax.set_ylim(top=1.03)

        ax.grid()
        fig.tight_layout()
#       fig.subplots_adjust(top=0.88)
        if save:
            file_location = './figures/'
            file_name = f'g3_{restart:s}_n{n:d}_p{ps:s}_q{qs:s}_M{M:d}_R{R:d}_{c:s}.png'
            fig.savefig(file_location+file_name, dpi=300)

if not save:
    plt.show()
