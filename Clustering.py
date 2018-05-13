#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:57:35 2017

@author: akkasi
"""

import networkx as nx
import random


from utility import *
from operator import itemgetter
from collections import OrderedDict

def Chinese_whispers(Graph,iterations=20):
    
    for z in range(0, iterations):
        gn = Graph.nodes()

        random.shuffle(gn)
        for node in gn:
            neighs = Graph[node] # Get neighbors of node
            classes = {}     # class list of each node with their weights
            for ne in neighs:
                if isinstance(ne, int): # assumed that nodes' type are Integer
                    key=Graph.node[ne]['class']
                    if key in classes:
                        classes[key] += Graph[node][ne]['weight']
                    else:
                        classes[key] = Graph[node][ne]['weight']
       # find the class with the highest edge weight sum
            max = 0
            maxclass = 0
            for c in classes:
                if classes[c] > max:
                    max = classes[c]
                    maxclass = c
       # set the class of target node to the winning local class
            Graph.node[node]['class'] = maxclass
    n_clusters = []
    for node in Graph.nodes():
        n_clusters.append(Graph.node[node]['class'])
    clusters=set(n_clusters)
    All_Clusters=[]
    for e in clusters:
        cl=[]
        for n in Graph.nodes():
            if (nx.get_node_attributes(Graph,'class')[n]==e):
                cl.append(n)
        All_Clusters.append(cl)
    
    """
     Find clusters' representatives based on Beiman's approach. the edges between 
     different clusters are excluded.
    """

    return All_Clusters

def Leader_Follwer_Clustering(Graph): # Graph must be weighted undirected graph
    """ Phase 1: Detect Leaders and Followers
    """
    Leaders=[]
    Followers=[]
    Clusters=[]
    Nodes=Graph.nodes()
    Distance_Set=nx.all_pairs_dijkstra_path_length(Graph)
    # print(Distance_Set)
    Distance_centrality={} # the elements will be like nodei:dci for all nodes=1...m
    for node in Nodes:
        s=0
        for a in Distance_Set[node]:
            s+=Distance_Set[node][a]
            
        Distance_centrality[node]=s
    
    Followers=Nodes[:]
    for node in Nodes:
        
        n_dist=Distance_centrality[node]
        n_neig=Graph.neighbors(node)
        for n in n_neig:
            if n_dist < Distance_centrality[n]:
                Leaders.append(node)
                Followers.remove(node)
                break
            
    """ Phase 2 Community Assignment
    """
    leaders_distance={}
    for a in Leaders:
        leaders_distance[a]=Distance_centrality[a]
        
    Sorted_leaders=list(OrderedDict(sorted(leaders_distance.items(), key=itemgetter(1))))
    
    M={}
    for a in Sorted_leaders:
        M[a]=a
    for a in Followers:
        M[a]=-1
    C={}
   
    i=0
    while(i<len(Sorted_leaders)):
        n_neig=Graph.neighbors(Sorted_leaders[i])
        intersect=list(set(n_neig) & set(Followers))
        FVi=[]
        for a in intersect:
            if M[a]==-1:
                FVi.append(a)
        for a in FVi:
            M[a]=Sorted_leaders[i]
        C[Sorted_leaders[i]]=[Sorted_leaders[i]]+FVi
        i+=1
#    print(C)
    i=0
    while(i<len(Sorted_leaders)):
        if C[Sorted_leaders[i]]==[Sorted_leaders[i]]:
            del C[Sorted_leaders[i]]
            HVi=[]
            n_neig=Graph.neighbors(Sorted_leaders[i])
            
            intersect=list(set(n_neig) & set(Followers))
            for a in intersect:
                HVi.append(M[a])
              
            if len(HVi)!=0:
                while(True):
                    u=Mode(HVi)
                    if u in C:
                        break
                M[Sorted_leaders[i]]=u
                C[u]=C[u]+[Sorted_leaders[i]]
            else:
                IVi=n_neig[:]
                while(True):
                    u=Mode(IVi)
                    if u in C:
                        break
                M[Sorted_leaders[i]]=u
                C[u]=C[u]+[Sorted_leaders[i]]
               
        i+=1
    
    for a in C:
        Clusters.append(C[a])
    DN=[]
    for a in Nodes:
        if a not in Flatten2dList(Clusters):
            DN.append(a)
    NewClusters=[]
    for a in M:
        if M[a]==-1:
            NewClusters.append(a)
    Clusters.append(NewClusters)
    Clusters= [x for x in Clusters if x != []]

    return Clusters
