# coding: utf-8
"""
g4_experimento.py

Realiza experimento para gerar e salvar os
dados do grafico1.

Entrada
	-n: tamanho das comunidades
	-r: numero de repeticoes
	--seed (opcional): seed usado pelo np.random e geracao do sbm

Para cada repeticao e cada par (p, q), gera uma
instancia do SBM e executa os algoritmos local e global,
calculando as metricas
	'acuracia','niter','ciclo','pvfixos','acvfixos','acvmudam'

Os dados gerados sao salvos como DataFrames em arquivos .csv
	'g4_n{n}_r{R}_p{p}_{local ou global}.csv'
Os DataFrames sao indexados por (p-q,r).

Esses dados deverao ser lidos pelo script grafico1_visualizacao.py
que lê os arquivos .csv e desenha os graficos correspondentes a 
cada métrica.
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
parser.add_argument('-n', default=50, type=int,
	                help='numero de vertices')
parser.add_argument('-p', type=float,
                    help='probabilidade de aresta intra-comunidade')
parser.add_argument('-M', default=10, type=int,
	                help='numero de instancias')
parser.add_argument('-R', default=10, type=int,
	                help='numero de repeticoes para cada instancia')
parser.add_argument('--seed', default=None, type=int,
	                help='seed do random para gerar o SBM')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()
n = args.n; 
p = args.p;
R = args.R; M = args.M
sbm_seed = args.seed
verbose = args.verbose

# Fixar as fontes de aleatoriedade
numpy_seed = sbm_seed
np.random.seed(numpy_seed)
random.seed(sbm_seed) # stochastic block model

# Inicializacao dos parametros do experimento
nrounds = 25
N = 2*n
columns = ['acuracia','ciclo','niter','pvfixos','acvfixos']
column_labels = {
	'acuracia': 'acurácia', 'niter':r'iterações', 'ciclo':'tamanho do ciclo',
	'pvfixos': r'fração de vértices fixos', 'acvfixos': 'acurácia dos vértices fixos',
	'acvmudam': 'acurácia dos vértices que mudam'
}
algorithms = ['hard','soft'] 

# Esses valores sao bons para n = 1000 e p = 0.3
#dvals = [0.01, 0.02, 0.03, 0.04, 0.05]
dvals = [0.02, 0.04, 0.05]

print("n = {:d}, p = {:.4f}".format(n,p))
print("dvals = ", dvals)
print("algorithms = ", algorithms)
print("columns = ", columns)
print(f"R = {R:d}, M = {M:d}")
print()

# Construcao da estrutura que armazena os dados gerados
D = {}
for a,c in ((a,c) for a in algorithms for c in columns):
    D[(a,c)] = [np.zeros((nrounds+1,M,R)) for d in dvals]

tstart = time.time()

a = 'hard'
print('Running hard bootstrap simulations')
for i,d in enumerate(dvals):
    q = p - d
    for m in range(M):
        sbm = SBM(n,p,q)
        G = sbm.G
        for r in range(R):
            if verbose and r%10 == 0: 
                print(f"i = {i:d} ({d:.2f}), m = {m:d}, r = {r:d}")

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
                        D[(a,c)][i][t,m,r] = sbm.eval_labels(labels)
                    else:
                        D[(a,c)][i][t,m,r] = results[c]
print()
a = 'soft'
print('Running soft bootstrap simulations')
for i,d in enumerate(dvals):
    q = p - d
    for m in range(M):
        sbm = SBM(n,p,q)
        G = sbm.G
        for r in range(R):
            if verbose and r%10 == 0:
                print(f"i = {i:d} ({d:.2f}), m = {m:d}, r = {r:d}")

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
                        D[(a,c)][i][t,m,r] = sbm.eval_labels(labels)
                    else:
                        D[(a,c)][i][t,m,r] = results[c]

tend = time.time()
main_time = tend - tstart
print("Wall-clock time: ", main_time)

simulation_data = {
    'n': n, 'p': p, 'R': R, 'M': M,
    'dvals': dvals,
    'sbm_seed': sbm_seed,
    'columns': columns,
    'algorithms': algorithms,
    'D': D
}

file_location = './data/g4/'
file_name = f'g4_n{n:d}_p{p:.2f}_R{R:d}_M{M:d}_s{sbm_seed:d}.data'
fileObj = open(file_location + file_name, 'wb')
pickle.dump(simulation_data,fileObj)
fileObj.close()
