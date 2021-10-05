# coding: utf-8

import networkx as nx

G = nx.MultiDiGraph()
G.add_edge(1,2)
G.add_edge(2,1)
G.add_edge(2,3)
G.add_edge(2,3) # multiaresta
G.add_edge(2,3) # multiaresta
G.add_edge(4,2)
G.add_edge(4,2) # multiaresta
G.add_edge(2,4)
G.add_edge(4,4) # selfloop
print("G.nodes:", list(G.nodes))
print("G.edges:", list(G.edges))
print("G.number_of_edges():", G.number_of_edges())

print()
Gu = nx.Graph(G)
#Gu = G.to_undirected()
print("Gu.nodes:", list(Gu.nodes))
print("Gu.edges:", list(Gu.edges))
print("Gu.number_of_edges():", Gu.number_of_edges())

def num_self_loops(G):
    count = 0
    for u in G.nodes:
        if G.has_edge(u,u):
            count += 1
    return count

print("num_self_loops(G):", num_self_loops(G))

def num_multiedges(G):
    count = 0
    me = {}
    for u,v,k in G.edges:
        if len(G[u][v]) > 1:
            count += 1
            if (u,v) not in me:
                me[(u,v)] = 1
                print((u,v), G[u][v])
            else:
                me[(u,v)] += 1

    print("total multiedges:", count)
    print("total distinct multiedges:", len(me))

num_multiedges(G)
