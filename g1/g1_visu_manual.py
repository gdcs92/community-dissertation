# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import random
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true',
					help='salvar graficos em arquivos .pdf')

args = parser.parse_args()
save = args.save

column_labels = {
	'acuracia': 'acurácia', 'niter':r'iterações', 'ciclo':'tamanho do ciclo',
	'pvfixos': r'fração de vértices fixos', 'acvfixos': 'acurácia dos vértices fixos',
	'acvmudam': 'acurácia dos vértices que mudam'
}

## Parametros fixos
n = 1000
p = 0.01
M = 10
R = 30

## Leitura dos dados pickle
dvals = {}; D = {}
fileloc = './data/'

a = 'spectral'
filename = f"g1_n1000_p0.0100_R1_M{M:d}_s10_spectral.data"
fileObj = open(fileloc+filename, 'rb')
simulation_data = pickle.load(fileObj)
fileObj.close()
dvals[a] = simulation_data['dvals']
D[a] = simulation_data['D']

a = 'mva'
filename = f"g1_n1000_p0.0100_R{R:d}_M{M:d}_s10_mva.data"
fileObj = open(fileloc+filename, 'rb')
simulation_data = pickle.load(fileObj)
fileObj.close()
dvals[a] = simulation_data['dvals']
D[a] = simulation_data['D']

a = 'gam'
filename = f"g1_n1000_p0.0100_R{R:d}_M{M:d}_s10_gam.data"
fileObj = open(fileloc+filename, 'rb')
simulation_data = pickle.load(fileObj)
fileObj.close()
dvals[a] = simulation_data['dvals']
D[a] = simulation_data['D']


## Geracao dos graficos
#
algorithms = ['spectral','mva','gam']
alglabels = {'spectral': 'spectral', 'mva': 'MVA', 'gam': 'GAM', 'soft': 'soft bootstrap'}
columns = ['acuracia']

for c in columns:

    fig, ax = plt.subplots(figsize=(8,6))

    for j,a in enumerate(algorithms):
        xvals = dvals[a]
        ydata = D[a][c]
        yvals = np.mean(ydata, axis=(1,2))
        se = np.std(ydata, axis=(1,2), ddof=1)
        nn = ydata.shape[1]*ydata.shape[2] 
        yerr = 1.96*se/np.sqrt(nn)

        ax.errorbar(xvals, yvals, yerr=yerr, label=alglabels[a],
            color=f'C{j:d}', linestyle='-', marker='o')

#    a = 'spectral'
#    k = (a,c)
#    xvals = dvals[a]
#    yvals = np.mean(D[k], axis=1)
#    se = np.std(D[k], axis=1, ddof=1)
#    nn = D[k].shape[-1]
#    yerr = 1.96*se/np.sqrt(nn)       
#    ax.errorbar(xvals, yvals, yerr=yerr, label=alglabels[a],
#        color=f'C{j+1:d}', linestyle='-', marker='o')
#
#    a = 'soft'
#    k = (a,c)
#    xvals = bs_dvals
#    yvals = np.mean(bs_vals, axis=1)
#    se = np.std(bs_vals, axis=1, ddof=1)
#    nn = bs_vals.shape[-1]
#    yerr = 1.96*se/np.sqrt(nn)       
#    ax.errorbar(xvals, yvals, yerr=yerr, label=alglabels[a],
#        color=f'C{j+2:d}', linestyle='-', marker='o')


    ylabel = column_labels[c]
    ax.grid(True)
    ax.set_xlabel(r'$p-q$', fontsize=14)
    ax.set_ylabel('accuracy', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(right=p)

    #ax.set_title(f'n = {n:d}, p = {p:.1f}, R = {R:d}, M = {M:d}, seed = {sbm_seed:d}', fontsize=13)

    fig.legend(loc='lower left', bbox_to_anchor=(0.7,0.6), fontsize=13)
    fig.tight_layout()

    if save:
        file_location = './figures/'
        file_name = f'g1_n{n:d}p{p:.1f}M{M:d}R{R:d}_{c:s}.png'
        fig.savefig(file_location+file_name, dpi=100)

if not save:
    plt.show()
