# coding: utf-8
"""
g5_experimento.py
"""
import numpy as np
import networkx as nx
import random
import argparse
import pickle
import time

from community import *
from algoritmos import *
from algoritmos import _hillclimb_filter

parser = argparse.ArgumentParser()

parser.add_argument('-n', default=50, type=int,
	                help='numero de vertices')
parser.add_argument('-p', type=float,
                    help='probabilidade de aresta intra-comunidade')
parser.add_argument('-d', type=float,
                    help='diferenca p-q')
parser.add_argument('-M', default=10, type=int,
	                help='numero de instancias usadas no total')
parser.add_argument('-K', default=10, type=int,
	                help='numero de instancias que aparecem no grafico')
parser.add_argument('--seed', default=None, type=int,
	                help='seed do random para gerar o SBM')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--view', action='store_true')
parser.add_argument('--save', action='store_true')

args = parser.parse_args()
n = args.n; 
p = args.p; d = args.d; q = p - d
M = args.M
K = args.K
sbm_seed = args.seed
verbose = args.verbose
view = args.view
save = args.save

# Fixar as fontes de aleatoriedade
numpy_seed = sbm_seed
np.random.seed(numpy_seed)
random.seed(sbm_seed) # stochastic block model

# Inicializacao dos parametros do experimento
N = 2*n
maxiter = 300
update = 'global'

print("n = {:d}, p = {:.4f}, q = {:.4f} ( p-q = {:.4f} )".format(n,p,q,d))
print("maxiter = ", maxiter)
print(f"M = {M:d}, K = {K:d}")
print()

# Construcao da estrutura que armazena os dados gerados
# D é uma lista com as 'series temporais' de fbar
D = [[] for m in range(M)]

tstart = time.time()

for m in range(M):
    print(f"m = {m:d}")
    sbm = SBM(n,p,q)
    G = sbm.G

    # inicializacao
    labels = np.random.binomial(1,0.5,size=N).astype(int)
    vfixos = np.zeros(N,dtype=int)

    H = [labels]
    new_labels = np.zeros(N)
    vfixos = np.zeros(N)
    ciclo = 0
    nvfixos = 0
    acfixos = -1
    acmudam = -1

    A = nx.adjacency_matrix(G)
    W = np.diag([1/G.degree(i) for i in G])*A

    for t in range(maxiter):
        
        gbar = np.mean(np.dot(W,labels))
        D[m].append(gbar)

        new_labels = _hillclimb_filter(np.dot(W,labels),update)

        labels = np.copy(new_labels)

        # ver se entrou em ciclo comparando com os labels ja obtidos
        for j in range(t,-1,-1):
            if np.all(labels == H[j]):
                # encontrei um ciclo de tamanho t+1-j
                ciclo = t+1-j
                # cada linha de S é o array de labels de um dos estados
                # do ciclo
                S = np.array(H[j:])
                # np.all(S==S[0,:],axis=0) seleciona as colunas em que
                # todos os valores sao iguais ao primeiro -- labels que
                # sao constantes durante todo o ciclo
                vfixos = np.all(S==S[0,:],axis=0)
                vertices_fixos = np.nonzero(vfixos)[0]
                vertices_mudam = np.nonzero(1-vfixos)[0]
                nvfixos = len(vertices_fixos)
                nvmudam = len(vertices_mudam)
                if nvfixos + nvmudam != N:
                    print("\n!!! A soma de vertices fixos e nao fixos nao bate\n")
                if nvfixos > 0:
                    acfixos = sbm.eval_labels(labels,\
                        vertices=vertices_fixos,normalize=True)
                if nvmudam > 0:
                    acmudam = sbm.eval_labels(labels,\
                        vertices=vertices_mudam,normalize=True)
                else: # se nenhum vertice muda, a acuracia "dos que mudam" = 1
                    acmudam = 1
                break
        if ciclo:
            break
        else:
            H.append(labels)
            
tend = time.time()
main_time = tend - tstart
print("Wall-clock time: ", main_time)

#simulation_data = {
#    'n': n, 'p': p, 'd': d, 'M': M,
#    'sbm_seed': sbm_seed,
#    'D': D
#}
#file_location = './data/g5/'
#file_name = f'g5_n{n:d}_p{p:.2f}_d{d:.3f}_R{R:d}_M{M:d}_s{sbm_seed:d}.data'
#fileObj = open(file_location + file_name, 'wb')
#pickle.dump(simulation_data,fileObj)
#fileObj.close()

## Preprocessing
# D é uma lista com as 'series temporais' de fbar

# Escolhe os indices de K instancias ao acaso
sample_ids = np.random.choice(M,K,replace=False)
sD = [D[i] for i in sample_ids]
for x in [(i,len(sD[i])) for i in range(K)]:
    print(x)

# Encontro o maior comprimento dentre as series amostradas
Tmax = 0
for k in range(K):
    if len(sD[k]) > Tmax:
        Tmax = len(sD[k])

# Agora faço a media de todas as series
avgD = np.zeros(Tmax)
for t in range(Tmax):
    v = []
    for m in range(M):
        if len(D[m]) > t:
            v.append(D[m][t])
    avgD[t] = np.mean(v) 

#DD = sorted(list(enumerate(sD)), key=lambda x: len(x[1]))
#smallest_len = [DD[0][0], DD[1][0]]
#print("smallest_len = ", smallest_len)
#
#for x in [(i,len(sD[i])) for i in range(K)]:
#    print(x)
#
#T = len(DD[-1][1])
#meanvals = [0 for t in range(T)]
#for t in range(T):
#    v = []
#    for k in range(K):
#        if len(sD[k]) > t:
#            v.append(sD[k][t])
#    meanvals[t] = np.mean(v)

#fig, ax = plt.subplots(figsize=(6,5))
fig, ax = plt.subplots(figsize=(8,6))
for k in range(K):
    xvals = range(1, len(sD[k])+1)
    yvals = sD[k]
    ax.plot(xvals, yvals, color=f'C{k:d}', alpha=1.0, linewidth=0.8)

ax.plot(range(1,Tmax+1), avgD, color='black', linewidth=1.2, alpha=0.9)

#ax.set_title(f'n = {n:d}, p = {p:.2f}, d = {d:.3f}, K = {K:d}', fontsize=14)
ax.set_xlabel('iterations', fontsize=14)
ax.set_ylabel(r'$\bar{f}$', rotation=0, fontsize=16)
ax.tick_params(axis='both', labelsize=12)
ax.grid()
fig.tight_layout()

if save:
    file_location = './figures/'
    ps = f'{p:.4f}'[2:]
    ds = f'{d:.4f}'[2:]
    file_name = f'g5_n{n:d}_p{ps:s}_d{ds:s}_M{M:d}_K{K:d}_s{sbm_seed:d}.png'
    fig.savefig(file_location+file_name, dpi=300)

if view: 
    plt.show()
