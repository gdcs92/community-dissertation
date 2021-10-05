# coding: utf-8
"""
g4_visu2.py
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
parser.add_argument('-M', default=10, type=int,
	                help='numero de instancias')
parser.add_argument('-R', default=10, type=int,
	                help='numero de repeticoes para cada instancia')
parser.add_argument('--nrounds', default=5, type=int,
	                help='numero de rounds do bootstrap')
parser.add_argument('--seed', type=int,
	                help='seed do random para gerar o SBM')
parser.add_argument('--save', action='store_true',
					help='salvar graficos em arquivos .pdf')

args = parser.parse_args()
n = args.n; 
p = args.p;
R = args.R; M = args.M
save = args.save
nrounds = args.nrounds
seed = args.seed

## Inicializacao dos parametros do experimento
T = nrounds+1

algorithms = ['hard','soft']
columns = ['acuracia','ciclo','niter','pvfixos','acvfixos']
column_labels = {
	'acuracia': 'accuracy', 
    'niter': 'iterations',
    'ciclo': 'cycle length',
	'pvfixos': 'fraction of fixed vertices',
    'acvfixos': 'fixed vertices accuracy',
}
qvals = [0.003, 0.004, 0.005, 0.006, 0.007]

file_loc = './data/'
D = {}

for a in algorithms:
    for j,q in enumerate(qvals):
        ps = f'{p:.4f}'[2:]; qs = f'{q:.4f}'[2:]
        filename = f"g3_n{n:d}_p{ps:s}_q{qs:s}_R{R:d}_M{M:d}_{a:s}_s{seed:d}.data" 

        fileObj = open(file_loc + filename, 'rb')
        simdata = pickle.load(fileObj)
        fileObj.close()

        D[(a,q)] = {}
        for c in columns:
            D[(a,q)][c] = simdata['D'][c]

newD = {}
for a,q in ((a,q) for a in algorithms for q in qvals):
    k = (a,q)
    newD[k] = {}
    for c in columns:
        newD[k][c] = [[] for t in range(T)]
        for t in range(T):
            newD[k][c][t] = D[k][c][t,:,:].flatten() 

    ## remocao de valores ruins
    goodvals = [v>0 for v in newD[k]['ciclo']]

    for c in columns:
        for i in range(len(newD[k][c])):
            newD[k][c][i] = [x for j,x in enumerate(newD[k][c][i]) if goodvals[i][j] > 0]


## Visualizacao estatistica
#
xvals = list(range(nrounds+1))
bboxs = {
    ('hard','acuracia'): (0.7, 0.48),
    ('hard','ciclo'): (0.2,0.2),
    ('hard','niter'): (0.7, 0.58),
    ('hard','pvfixos'): (0.7, 0.52),
    ('hard','acvfixos'): (0.7, 0.52),
    ('soft','acuracia'): (0.7, 0.6),
    ('soft','ciclo'): (0.2,0.2),
    ('soft','niter'): (0.7, 0.64),
    ('soft','pvfixos'): (0.66, 0.2),
    ('soft','acvfixos'): (0.36, 0.58),
}

for c in columns:
    ylabel = column_labels[c]
    for a in algorithms:
        fig, ax = plt.subplots(figsize=(6,5))
#        k = (a,c)
        for j,q in enumerate(qvals):
            k = (a,q)
            Ddk = newD[k][c]
            yvals = np.zeros(T); se = np.zeros(T); yerr = np.zeros(T)
            for i in range(T):
                yvals[i] = np.mean(Ddk[i])
                se[i] = np.std(Ddk[i], ddof=1)
                yerr[i] = 1.96*se[i]/np.sqrt(len(Ddk[i]))

            ax.errorbar(xvals, yvals, yerr=yerr, label=f'{p-q:.3f}',
                color=f'C{j:d}', marker='o')

#        ax.set_title(f'{a:s} bootstrapping', y=1.05, fontsize=14)
        ax.set_xlabel('bootstrap round', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)

#        fig.legend(loc='lower left', bbox_to_anchor=bboxs[(a,c)], fontsize=13)

        fig.legend(loc='lower left', bbox_to_anchor=(0.11,0.86,0.86,0.8), 
            mode='expand', ncol=5, fontsize=11, title='p-q', title_fontsize=12)
        plt.subplots_adjust(top=0.86,right=0.96)

        if c in ['acuracia','pvfixos','acvfixos']:
            ax.set_ylim(top=1.02)

        xticks = list(range(nrounds+1))
        ax.set_xticks(xticks)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True)
#        fig.tight_layout()

        if save:
            file_location = './figures/'
            file_name = f'g4_{a:s}_n{n:d}_p{ps:s}_M{M:d}_R{R:d}_{c:s}.png'
            fig.savefig(file_location+file_name, dpi=300)

if not save:
    plt.show()
