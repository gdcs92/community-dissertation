# coding: utf-8
"""
g1_experimento1_bs-mva.py

Parametros fixos: n = 1000, p = 0.01, q = 0.01, ..., 0.001
"""
import numpy as np
import networkx as nx
import random
import argparse
import pickle
import time
from community import *
from algoritmos import *

parser = argparse.ArgumentParser()
#parser.add_argument('-n', default=50, type=int,
#	                help='numero de vertices')
parser.add_argument('-p', type=float,
                    help='probabilidade de aresta intra-comunidade')
parser.add_argument('-M', default=10, type=int,
	                help='numero de instancias')
parser.add_argument('-R', default=10, type=int,
	                help='numero de repeticoes para cada instancia')
parser.add_argument('--seed', default=None, type=int,
	                help='seed do random para gerar o SBM')
parser.add_argument('--restart', default=None, type=str,
	                help='restart to bootstrap: hard ou soft')
parser.add_argument('--nrounds', default=5, type=int,
	                help='numero de rounds do bootstrap')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()
#n = args.n
p = args.p
R = args.R; M = args.M
sbm_seed = args.seed
verbose = args.verbose
a = args.restart
nrounds = args.nrounds

if a not in ['hard', 'soft']:
    raise ValueError("--restart deve ser igual a 'hard' ou 'soft'")

# Fixar as fontes de aleatoriedade
numpy_seed = sbm_seed
np.random.seed(numpy_seed)
random.seed(sbm_seed) # stochastic block model

# Inicializacao dos parametros do experimento
n = 1000
#p = 0.01
#columns = ['acuracia','niter']
columns = ['acuracia','onelabels','niter','ciclo','pvfixos','acvfixos','acvmudam']
dvals = np.linspace(0,p-0.001,num=10)

# Construcao da estrutura que armazena os dados gerados
D = {}
for c in columns:
    D[c] = np.zeros((len(dvals),M,R))

tstart = time.time()

for i,d in enumerate(dvals):

    q = p-d    

    for m in range(M): # instancias

        sbm = SBM(n, p, q)

        for r in range(R): # repeticoes

            if verbose: 
                print(f"i: {i:d} (d={d:.4f}), m: {m:d}, r: {r:d}")

            results = bootstrap(sbm,nrounds,restart=a,update='mva')

            labels = results['labels']

            D['acuracia'][i,m,r] = sbm.eval_labels(labels)
            D['onelabels'][i,m,r] = sum(labels)/(2*n)
            for c in columns[2:]:
                D[c][i,m,r] = results[c]

tend = time.time()
main_time = tend - tstart
print("Wall-clock time: ", main_time)

simulation_params = {
    'n': n, 'p': p, 'R': R, 'M': M, 'sbm_seed': sbm_seed,
    'columns': columns,
    'dvals': dvals, 'D': D,
    'a': a, 'nrounds': nrounds
}

file_location = './data/'
file_name = f'g1_n{n:d}_p{p:.4f}_R{R:d}_M{M:d}_s{sbm_seed:d}_{a:s}-mva.data'
fileObj = open(file_location + file_name, 'wb')
pickle.dump(simulation_params,fileObj)
fileObj.close()
