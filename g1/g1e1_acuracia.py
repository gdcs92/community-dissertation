# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import random
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=float,
                    help='probabilidade de aresta intra-comunidade')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--save', action='store_true',
					help='salvar graficos em arquivos .pdf')

args = parser.parse_args()
p = args.p
sd = args.seed
save = args.save

column_labels = {
	'acuracia': 'acurácia', 'niter':r'iterações', 'ciclo':'tamanho do ciclo',
	'pvfixos': r'fração de vértices fixos', 'acvfixos': 'acurácia dos vértices fixos',
	'acvmudam': 'acurácia dos vértices que mudam'
}

## Parametros fixos
n = 1000
#p = 0.02
M = 10
R = 30
#sd = 12

algorithms = ['mva','gam','spectral','hard','soft','hard-mva'] #,'hard'] #'spectral' 
columns = ['acuracia'] # 'niter'

## Leitura dos dados pickle
dvals = {}; D = {}
fileloc = './data/'

a = 'spectral'
ps = f'{p:.4f}'#[2:]
filename = f"g1_n1000_p{ps:s}_R1_M{M:d}_s{sd:d}_{a:s}.data"
fileObj = open(fileloc+filename, 'rb')
simulation_data = pickle.load(fileObj)
fileObj.close()
dvals[a] = simulation_data['dvals']
D[a] = simulation_data['D']

for a in algorithms:
    if a == 'spectral':
        continue
    D[a] = {}
    filename = f"g1_n{n:d}_p{p:.4f}_R{R:d}_M{M:d}_s{sd:d}_{a:s}.data"
    fileObj = open(fileloc+filename, 'rb')
    simulation_data = pickle.load(fileObj)
    fileObj.close()
    dvals[a] = simulation_data['dvals'] # igual para todas as instancias, posso sobrescrever
    # simulation_data['D'][c].shape = (len(dvals), M, R)
    D[a] = simulation_data['D']

## Geracao dos graficos
#
alglabels = {'spectral': 'spectral', 'mva': 'MVA', 'gam': 'GAM', 'soft': 'soft GAMB',
    'hard': 'hard GAMB', 'hard-mva': 'hard MVAB' 
}

for c in columns:

    fig, ax = plt.subplots(figsize=(6,5))

    for j,a in enumerate(algorithms):
        xvals = dvals[a]
        ydata = D[a][c]
        yvals = np.mean(ydata, axis=(1,2))
        se = np.std(ydata, axis=(1,2), ddof=1)
        nn = ydata.shape[1]*ydata.shape[2] 
        yerr = 1.96*se/np.sqrt(nn)

        ax.errorbar(xvals, yvals, yerr=yerr, label=alglabels[a],
            color=f'C{j:d}', linestyle='-', marker='o')

    ylabel = column_labels[c]
    ax.grid(True)
    ax.set_xlabel(r'$p-q$', fontsize=14)
    ax.set_ylabel('accuracy', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(right=p)
    ax.set_ylim(top=1.03)

    #ax.set_title(f'n = {n:d}, p = {p:.1f}, R = {R:d}, M = {M:d}, seed = {sbm_seed:d}', fontsize=13)

    fig.legend(loc='lower left', bbox_to_anchor=(0.15,0.5), fontsize=13)
    fig.tight_layout()

    if save:
        file_location = './figures/'
        ps = f'{p:.4f}'[2:]
        file_name = f'g1_n{n:d}_p{ps:s}_R{R:d}_M{M:d}_s{sd:d}_{c:s}_mvab2_dpi300.png'
        fig.savefig(file_location+file_name, dpi=300)

if not save:
    plt.show()
