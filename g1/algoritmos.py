# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from community import *


def spectral(sbm):
    n = sbm.n; N = 2*n
    G = sbm.G;
    A = nx.adjacency_matrix(G)
    A = A.toarray()
    w, v = np.linalg.eigh(A)
    ew_2 = v[:,-2]
    labels = np.array([1 if x>0 else 0 for x in ew_2],dtype='int')
    results = {}
    results['labels'] = labels
    return results

def _hillclimb_init(sbm, init='parity', noise=None):
    G = sbm.G; N = G.number_of_nodes()
    if init == 'alternating':
        labels = np.zeros(N)
        degrees = sorted(list(G.degree), key=lambda x: x[1])
        for i,v in enumerate(degrees):
            labels[v[0]] = i%2
    elif init == 'parity':
        labels = np.array([i%2 for i in range(N)])
    elif init == 'gt':
        labels = np.copy(sbm.planted_labels)
    elif init == 'noisy':
        labels = np.copy(sbm.planted_labels)
        try:
            changes = np.random.binomial(1,noise,size=N)
        except:
            raise ValueError
        labels = np.logical_xor(changes,labels).astype(int)
    else:
        labels = np.random.binomial(1,0.5,size=N)
    return labels

def _hillclimb_filter(x, rule):
    if rule in ['global', 'frac', 'gam']:
        c = np.mean(x)
        for u in range(len(x)):
            if x[u] > c:
                x[u] = 1
            elif x[u] < c:
                x[u] = 0
            else:
                x[u] = np.random.binomial(1,0.5)
        # x = x > np.mean(x)
    elif rule in ['local', 'maj', 'mva']:
        for u in range(len(x)):
            if x[u] > 0.5:
                x[u] = 1
            elif x[u] < 0.5:
                x[u] = 0
            else:
                x[u] = np.random.binomial(1,0.5)
    else:
        print("Argumento rule deve ser igual a 'global' ou 'local'.")
        raise ValueError
    return x # x.astype(int)

def labels2string(labels):
    return ''.join(list(map(str,labels)))

def string2labels(string):
    try:
        S = [list(map(int,list(s))) for s in string]
    except:
        S = list(map(int,list(string))) 
    return S

def hillclimb(sbm, maxiter=100, init='random', update='global', 
    init_labels=None, noise=None):
    n = sbm.n; N = 2*n
    G = sbm.G; planted_partition = sbm.planted_partition

    # ISSO TEM QUE SER REFEITO DE MANEIRA MAIS LIMPA
    if init_labels is not None and len(init_labels)==N:
        labels = np.copy(init_labels)
    else:
        labels = _hillclimb_init(sbm, init=init, noise=noise)

    H = [labels]
    new_labels = np.zeros(N)
#    vertices_fixos = []
    vfixos = np.zeros(N)
    ciclo = 0
    nvfixos = 0
    acfixos = -1
    acmudam = -1

    A = nx.adjacency_matrix(G)
    W = np.diag([1/G.degree(i) for i in G])*A

    for t in range(maxiter):

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

    results = {}
    results['labels'] = labels
    results['niter'] = t+1
    results['ciclo'] = ciclo
    results['pvfixos'] = nvfixos / N
    results['acvfixos'] = acfixos
    results['acvmudam'] = acmudam
    results['vfixos'] = vfixos
    return results 

def hard_bootstrap(sbm, nrounds=5, **kwargs):
    n = sbm.n; N = 2*n
    # random init
    labels = np.random.binomial(1,0.5,size=N)
    vfixos = np.zeros(N,dtype=int)
    _niter = 0 

    for t in range(nrounds+1):
        if sum(vfixos) > 0:
            random_labels = np.random.binomial(1,0.5,size=N)
            for i in range(N):
                if not vfixos[i]:
                    labels[i] = random_labels[i]

        results = hillclimb(sbm,init_labels=labels,**kwargs)
        labels = results['labels']
        vfixos = results['vfixos']
        _niter += results['niter']

    results['niter'] = _niter
    return results

def soft_bootstrap(sbm, nrounds=5, **kwargs):
    n = sbm.n; N = 2*n
    G = sbm.G
    # random init
    labels = np.random.binomial(1,0.5,size=N)
    vfixos = np.zeros(N,dtype=int)
    _niter = 0 

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

        results = hillclimb(sbm, init_labels=labels, **kwargs)
        labels = results['labels']
        vfixos = results['vfixos']
        _niter += results['niter']

    results['niter'] = _niter
    return results

def bootstrap(sbm, nrounds=5, restart='hard', **kwargs):
    if restart == 'soft':
        return soft_bootstrap(sbm,nrounds,**kwargs)
    else:
        return hard_bootstrap(sbm,nrounds,**kwargs)

### NEW METHODS ###
def acc(x,y):
    if len(x) != len(y):
        raise ValueError('x and y must be arrays of the same size')
    N = len(x)
    # IMPORTANTE: x,y devem assumir valores apenas 0 ou 1
    matches1 = sum(x == y)
    matches2 = sum(x == 1-y)
    score = max([matches1,matches2]) / N
    return score

def spectral_method(G, matrix='adjacency'):
    if matrix == 'adjacency':
        A = nx.adjacency_matrix(G)
        A = A.toarray()
        w,v = np.linalg.eigh(A)
        ew_2 = v[:,-2]
        labels = np.array([1 if x>0 else 0 for x in ew_2],dtype='int')
        results = {'labels': labels}
    return results

def hillclimb_method(G, maxiter=100, update='gam', init_labels=None):

    N = len(G)
    if init_labels is not None and len(init_labels)==N:
        labels = np.copy(init_labels)
    else:
        labels = np.random.binomial(1,0.5,size=N)

    H = [labels]
    new_labels = np.zeros(N)
    vfixos = np.zeros(N)
    ciclo = 0
    nvfixos = 0

    A = nx.adjacency_matrix(G)
    W = np.diag([1/G.degree(i) for i in G])*A

    for t in range(maxiter):

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
                nvfixos = len(vertices_fixos)
                break
        if ciclo:
            break
        else:
            H.append(labels)

    results = {}
    results['labels'] = labels
    results['niter'] = t+1
    results['ciclo'] = ciclo
    results['pvfixos'] = nvfixos / N
    results['vfixos'] = vfixos
    return results 

def mva_method(G,**kwargs):
    return hillclimb_method(G,update='mva',**kwargs)    

def gam_method(G,**kwargs):
    return hillclimb_method(G,update='gam',**kwargs)

def _hard_bootstrap(G, nrounds=5, **kwargs):
    N = len(G)
    # random init
    labels = np.random.binomial(1,0.5,size=N)
    vfixos = np.zeros(N,dtype=int)
    _niter = 0 

    for t in range(nrounds+1):
        if sum(vfixos) > 0:
            random_labels = np.random.binomial(1,0.5,size=N)
            for i in range(N):
                if not vfixos[i]:
                    labels[i] = random_labels[i]

        results = hillclimb_method(G,init_labels=labels,**kwargs)
        labels = results['labels']
        vfixos = results['vfixos']
        _niter += results['niter']

    results['niter'] = _niter
    return results

def _soft_bootstrap(G, nrounds=5, **kwargs):
    N = len(G)
    # random init
    labels = np.random.binomial(1,0.5,size=N)
    vfixos = np.zeros(N,dtype=int)
    _niter = 0 

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

        results = hillclimb_method(G, init_labels=labels, **kwargs)
        labels = results['labels']
        vfixos = results['vfixos']
        _niter += results['niter']

    results['niter'] = _niter
    return results


def bootstrap_method(G, restart, nrounds=5, **kwargs):
    if restart == 'hard':
        return _hard_bootstrap(G,nrounds,**kwargs)
    elif restart == 'soft':
        return _soft_bootstrap(G,nrounds,**kwargs)
    else:
        raise ValueError("restart must be equal to 'hard' or 'soft'.")
