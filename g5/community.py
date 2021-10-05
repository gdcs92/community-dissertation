# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class SBM:
	"""Stochastic Block Model com 2 comunidades"""

	def __init__(self, n, p_in, p_out,\
		connected=True,\
		planted_partition=None, sbm_seed=None):
		self.n = n
		self.p_in = p_in
		self.p_out = p_out
		
		planted_labels = np.zeros(2*n, dtype='int')
		if planted_partition is None:
			planted_partition = dict()
			planted_partition[0] = list(range(n))
			planted_partition[1] = list(range(n,2*n))
		for k in planted_partition:
			for v in planted_partition[k]:
				planted_labels[v] = k

		self.planted_partition = planted_partition
		self.planted_labels = planted_labels
		self.G = nx.planted_partition_graph(2,n,p_in,p_out,sbm_seed)
		while connected and not nx.is_connected(self.G):
			self.G = nx.planted_partition_graph(2,n,p_in,p_out,sbm_seed)

	def eval_labels(self, labels, vertices=None, normalize=True):
		reference = self.planted_labels
		if vertices is not None:
			reference = reference[vertices]
			labels = labels[vertices]
		# permuta labels 0 e 1 e utiliza a permutacao
		# com mais acertos
		matches1 = sum(reference == labels)
		matches2 = sum(reference == 1-labels)
		score = max([matches1,matches2])
		# score = matches1
		if normalize == True:
			score = score / len(reference)
		return score

	def graph_minority(self):
		"""Calcula o numero de vertices sem
		maioria na planted bisection do sbm."""
		G = self.G
		planted_labels = self.planted_labels
		min_count = 0
		for u in G:
			k1 = sum([planted_labels[v] for v in G[u]])
			k0 = len(G[u]) - k1
			if (k1 > k0 and planted_labels[u] == 0) or \
			(k0 > k1 and planted_labels[u] == 1):
				min_count += 1
		return min_count

	def graph(self):
		return self.G

	def info(self):
		print(self.__doc__)
		print("n =", self.n)
		print("p_in = {:.6e}, p_out = {:.6e}".format(self.p_in,self.p_out))
		print("planted partition:")
		print(self.planted_partition)

def layout_planted_bisection(G,partition):
	n = int(G.number_of_nodes()/2)
	V1 = partition[0]
	V2 = partition[1]
	G1 = G.subgraph(V1); G2 = G.subgraph(V2)
	spring_k = n/(5*np.sqrt(n))
	pos1 = nx.spring_layout(G1, k=spring_k)
	pos2 = nx.spring_layout(G2, k=spring_k)
	pos = dict(pos1); pos.update(pos2)
	# Ajuste das posicoes
	r = 1.5
	for i in range(n):
		pos[i] = pos[i] + np.array([-r,0])
		pos[i+n] = pos[i+n] + np.array([r,0])
	return pos

def draw_planted_bisection(G,partition,pos):
	nx.draw_networkx_nodes(G, pos,	
	                       nodelist=partition[0],
	                       node_color='xkcd:slate grey',
	                       edgecolors='black',
	                       alpha=1)
	nx.draw_networkx_nodes(G, pos,
	                       nodelist=partition[1],
	                       node_color='white',
	                       edgecolors='black',
	                       alpha=1)
	nx.draw_networkx_edges(G, pos, edgelist=list(G.edges), width=1.0, alpha=0.5)
	plt.axis('off')

def plot_planted_bisection(G,partition):
	fig, ax = plt.subplots(figsize=(8,6))
	pos = layout_planted_bisection(G,partition)
	draw_planted_bisection(G,partition,pos)
	n = int(len(G)/2)
	plt.title('n = {:d}'.format(n), fontsize=14)
	plt.tight_layout()
	plt.show()

def labels2partition(labels):
	N = len(labels)
	partition = [set(), set()]
	for i,s in enumerate(labels):
		# assume labels em {0,1}
		partition[s].add(i)
	return partition

if __name__ == '__main__':
	
	n = 50
	p_in = 0.2
	p_out = 0.02
	sbm = SBM(n, p_in, p_out)
	sbm.info()
	plot_planted_bisection(sbm.G,sbm.planted_partition)