# coding: utf-8
"""
g3_experimento.py
"""
import numpy as np
import networkx as nx
import random
import argparse
import pickle
import time
#from timeit import default_timer as timer
from community import *
from algoritmos import *


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
parser.add_argument('--restart', default=None, type=str,
	                help='restart to bootstrap: hard ou soft')
parser.add_argument('--nrounds', default=5, type=int,
	                help='numero de rounds do bootstrap')
parser.add_argument('--seed', default=None, type=int,
	                help='seed do random para gerar o SBM')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()
n = args.n; 
p = args.p; d = args.d; q = p - d
R = args.R; M = args.M
sbm_seed = args.seed
verbose = args.verbose
restart = args.restart
nrounds = args.nrounds

if restart not in ['hard','soft']:
    raise ValueError("--restart deve ser igual a 'hard' ou 'soft'")

# Fixar as fontes de aleatoriedade
numpy_seed = sbm_seed
np.random.seed(numpy_seed)
random.seed(sbm_seed) # stochastic block model

# Inicializacao dos parametros do experimento
#nrounds = 25
N = 2*n
columns = ['acuracia','ciclo','niter','pvfixos','acvfixos']
column_labels = {
	'acuracia': 'acurácia', 'niter':r'iterações', 'ciclo':'tamanho do ciclo',
	'pvfixos': r'fração de vértices fixos', 'acvfixos': 'acurácia dos vértices fixos',
	'acvmudam': 'acurácia dos vértices que mudam'
}

print("n = {:d}, p = {:.4f}, q = {:.4f} ( p-q = {:.4f} )".format(n,p,q,d))
print("columns = ", columns)
print(f"R = {R:d}, M = {M:d}")
print(f"nrounds = {nrounds:d}, restart = {restart:s}")
print()

# Construcao da estrutura que armazena os dados gerados
D = {}
for c in columns:
    D[c] = np.zeros((nrounds+1,M,R))
#
## Construcao da estrutura que armazena os dados gerados
#D = {}
#for a,c in ((a,c) for a in algorithms for c in columns):
#    D[(a,c)] = np.zeros((nrounds+1,M,R))

tstart = time.time()

if restart == 'hard':
    a = 'hard'
    print('Running hard bootstrap simulations')
    for m in range(M):
        sbm = SBM(n,p,q)
        G = sbm.G
        for r in range(R):
            if verbose and r%10 == 0: print(f"m = {m:d}, r = {r:d}")

            # inicializacao
            labels = np.random.binomial(1,0.5,size=N) 
            vfixos = np.zeros(N,dtype=int)

            # iteracoes do bootstrap
            for t in range(nrounds+1):
                if sum(vfixos) > 0:
                    random_labels = np.random.binomial(1,0.5,size=N)
                    for v in range(N):
                        if not vfixos[v]:
                            labels[v] = random_labels[v]

                results = hillclimb(sbm, init_labels=labels, update='global')
                labels = results['labels']
                vfixos = results['vfixos']
                for c in columns:
                    if c == 'acuracia':
                        D[c][t,m,r] = sbm.eval_labels(labels)
                    else:
                        D[c][t,m,r] = results[c]
elif restart == 'soft':
    a = 'soft'
    print('Running soft bootstrap simulations')
    for m in range(M):
        sbm = SBM(n,p,q)
        G = sbm.G
        for r in range(R):
            if verbose and r%10 == 0: print(f"m = {m:d}, r = {r:d}")

            # inicializacao
            labels = np.random.binomial(1,0.5,size=N).astype(int)
            vfixos = np.zeros(N,dtype=int)

            # iteracoes do bootstrap
            for t in range(nrounds+1):
                if sum(vfixos) > 0:
                    for u in G:
                        viz_fixos = [v for v in G.adj[u] if vfixos[v]]
                        deg_u = G.degree(u)
                        if vfixos[u]:
                            p_manter = 0.5
                            if viz_fixos:
                                # No. de vizinhos fixos com a mesma classificacao
                                nvfmc = len([v for v in viz_fixos if labels[v] == labels[u]])
                                p_manter += 0.5 * nvfmc / len(viz_fixos)
                            # moeda enviesada decide se mantem ou troca a cor
                            if np.random.binomial(1,p_manter) == 0:
                               labels[u] = 1 - labels[u]
                        else: # not vfixos[u]
                            labels[u] = np.random.binomial(1,0.5)                                

                results = hillclimb(sbm, init_labels=labels, update='global')
                labels = results['labels']
                vfixos = results['vfixos']
                for c in columns:
                    if c == 'acuracia':
                        D[c][t,m,r] = sbm.eval_labels(labels)
                    else:
                        D[c][t,m,r] = results[c]

tend = time.time()
main_time = tend - tstart
print("Wall-clock time: ", main_time)

simulation_data = {
    'n': n, 'p': p, 'q': q, 'R': R, 'M': M,
    'sbm_seed': sbm_seed,
    'columns': columns,
    'algorithm': restart, 'nrounds': nrounds,
    'D': D
}

file_location = './data/'
ps = f'{p:.4f}'[2:]; qs = f'{q:.4f}'[2:]
file_name = f'g3_n{n:d}_p{ps:s}_q{qs:s}_R{R:d}_M{M:d}_{restart:s}_s{sbm_seed:d}.data'
fileObj = open(file_location + file_name, 'wb')
pickle.dump(simulation_data,fileObj)
fileObj.close()
