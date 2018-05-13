import tempfile
import os
import random
import networkx as nx
import numpy as np
#from stanfordcorenlp import StanfordCoreNLP
#from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tokenize import WordPunctTokenizer
from numpy import  dot
from numpy.linalg import norm
from collections import Counter
from operator import itemgetter
import time
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

random.seed=3
np.seterr(divide='ignore', invalid='ignore')
###############################################################################################
def Join(List,Delimiter):
    e=List[0]
    for p in List[1:]:
        e=str(e)+Delimiter+str(p)
    return e
######################################################################
def FindRepresentative(Graph, ClustersList):
    Representatives = []
    for cluster in ClustersList:
        if len(cluster) == 1:
            Representatives.append(cluster[0])
        else:
            MaxW = 0
            MaxNode = cluster[0]
            for n in cluster:
                sw = 0
                for m in cluster:
                    if m != n and Graph.has_edge(m, n):
                        sw += Graph.get_edge_data(n, m, 'weight')['weight']

                if sw > MaxW:
                    MaxW = sw
                    MaxNode = n

            Representatives.append(MaxNode)
    i = 0
    while (i < len(Representatives)):
        ClustersList[i].remove(Representatives[i])
        ClustersList[i].insert(0, Representatives[i])
        i += 1

    return Representatives, ClustersList


################################################################################
def CosineSimilarity(Vec1, Vec2):
    cos_sim = dot(Vec1, Vec2) / (norm(Vec1) * norm(Vec2))
    return cos_sim

#################################################################################

def Mode(List):
    counter = Counter(List)
    max_count = max(counter.values())
    mode = [k for k, v in counter.items() if v == max_count]
    if len(mode) > 1:
        return random.choice(mode)
    if type(mode) == list:
        return mode[0]
    return mode
################################################################################################
def FindSubsNumber(ListOfsubstitute,Threshold=0):
    Dict={}
    SubSList=[]
    for a in ListOfsubstitute:
        subs=a.split(',')
        for s in subs:
            if s not in SubSList:
                SubSList.append(s)
    for s in SubSList:
        i=0
        for a in ListOfsubstitute:
            subs=a.split(',')
            if s in subs:
                i+=1
        if i > Threshold:
            Dict.update({s:i})
    return Dict
###########################################################################
def CompareSubsLists(SourceDict,DestDict,Threshold=3):
    Intersection = list(set(SourceDict) & set(DestDict))
    w=0
    if len(Intersection) >= Threshold:
        for a in Intersection:
            w+=SourceDict[a]+DestDict[a]
        return w
    return w
##############################################################################
def CreateEdgeList(Infile,Outfile,InterSectionThreshold=5):
    f=open(Infile,'r')
    l=f.readlines()
    f.close()
    Words=[]
    for e in l:
        a=e.strip().split('\t')
        w=a[0].split('.')[0]
        if w not in Words:
            Words.append(w)
    o = open(Outfile, 'w')
    for w in Words:
        L=[]
        for e in l:
            if e.startswith(w+'.'):
                L.append(e)
        i=0
        while(i < len(L)-1):
            a=L[i].strip().split('\t')
            words = a[0].split('.')[0]
            source = a[0]
            SourceDict = FindSubsNumber(a[2:])
            j=i+1
            while(j<len(L)):
                b = L[j].strip().split('\t')
                wordd = b[0].split('.')[0]
                Dest = b[0]
                DestDict = FindSubsNumber(b[2:])
                if words==wordd:
                    d=CompareSubsLists(SourceDict,DestDict,Threshold=InterSectionThreshold)
                    if d!=0:
                        o.write(Join([source,Dest,str(d)],' '))
                        o.write('\n')
                j+=1
            i+=1
    o.close()
    return

####################################################################################
def FindWords(File):
    f=open(File,'r')
    l=f.readlines()
    f.close()
    Words=[]
    for e in l:
        w=e.split('\t')[0].split('.')[0]
        if w not in Words:
            Words.append(w)
    return  Words
##################################################################################
def FindWordIns(InputFinalLS,w):
    f = open(InputFinalLS, 'r')
    l = f.readlines()
    f.close()
    Instances=[]
    for e in l:
        id=e.split('\t')[0]
        if id.startswith(w+'.'):
            Instances.append(id)
    return list(set(Instances))
###################################################################################
def SelectRandomNodes(List,number):
    OutList=[]
    while(len(OutList)!=number):
        OutList.append(random.choice(List))
        OutList=list(set(OutList))
    return OutList
##################################################################################
def CreateLocalEdgeFile(EdgeFile,Nodes,LocalFile):
    f=open(EdgeFile,'r')
    l=f.readlines()
    f.close()
    o=open(LocalFile,'w')
    for e in l:
        a=e.strip().split()
        if (a[0] in Nodes) and (a[1] in Nodes):
            o.write(e)
    o.close()
    return
###################################################################################
def GetNodesAndWeightedEdges(EdgeListFile, AllNodes):
    f = open(EdgeListFile, 'r')
    lines = f.readlines()
    f.close()
    WeightedEdges = []
    for line in lines:
        e = line.strip().split()
        WeightedEdges.append((e[0], e[1], float(e[2])))
    return WeightedEdges
#####################################################################################
def CreateGraph(EdgeFile,Nodes):
    f=tempfile.NamedTemporaryFile(delete=False)
    CreateLocalEdgeFile(EdgeFile,Nodes,f.name)
    edges = GetNodesAndWeightedEdges(f.name,Nodes)
    os.remove(f.name)
    graph = nx.Graph()
    graph.add_nodes_from(Nodes)
    for n, v in enumerate(graph.nodes()):
        graph.node[v]['class'] = n
    graph.add_weighted_edges_from(edges)
    return graph

#####################################################################################
def FindEmbedding(Sentence,window_size,TW,SubList,model):
    Sentence=Sentence.lower()
    Tokens = WordPunctTokenizer().tokenize(Sentence)
    Sentence_Vec=np.array([0.0]*300, dtype=np.float64)
    Subs_Vec=np.array([0.0]*300, dtype=np.float64)
    if window_size == 0:

        i=0
        for t in Tokens:
            if t in model:
                np.add(Sentence_Vec, model[t], out=Sentence_Vec, dtype=np.float64)
                i+=1
        Sentence_Vec=Sentence_Vec/i
    else:
        if TW in model:
            np.add(Sentence_Vec, model[TW], out=Sentence_Vec, dtype=np.float64)
        TW_index = Tokens.index(TW)
        if TW_index == 0:
            i = 1
            c = 0
            ce =0
            while(i < window_size +1 ):
                if Tokens[i] in model:
                    np.add(Sentence_Vec, model[Tokens[i]], out=Sentence_Vec, dtype=np.float64)
                    ce +=1
                i+=1
                c += 1
                if i >= len(Tokens):
                    break
            Sentence_Vec = Sentence_Vec / ce
        elif TW_index == len(Tokens)-1:
            i = len(Tokens)-2
            c = 0
            ce = 0
            while(i>=0):
                if Tokens[i] in model:
                    np.add(Sentence_Vec, model[Tokens[i]], out=Sentence_Vec, dtype=np.float64)
                    ce +=1
                i -= 1
                c += 1
                if c == window_size -1:
                    break
            Sentence_Vec = Sentence_Vec / ce
        else:
            c1 = 1
            i = TW_index -1
            ce1 = 0
            while (i>=0):
                if Tokens[i] in model:
                    np.add(Sentence_Vec, model[Tokens[i]], out=Sentence_Vec, dtype=np.float64)
                    ce1 +=1
                i -=1
                c1 += 1
                if c1 == window_size:
                    break
            i = TW_index + 1
            c2 = 1
            ce2 =0
            while (i < len(Tokens)):
                if Tokens[i] in model:
                    np.add(Sentence_Vec, model[Tokens[i]], out=Sentence_Vec, dtype=np.float64)
                    ce2 +=1
                i += 1
                c2 += 1
                if c2 == window_size:
                    break
            Sentence_Vec = Sentence_Vec / (ce1+ce2)
    i = 0
    for t in SubList:
        if t in model:
            np.add(Subs_Vec, model[t], out=Subs_Vec, dtype=np.float64)
            i += 1
    Subs_Vec = Subs_Vec/i
    return Sentence_Vec, Subs_Vec
#####################################################################################
def Flatten2dList(List):
    flist=[]
    for l in List:
        for a in l:
            flist.append(a)
    return flist
####################################################################################
def Find_Representative(OverallEfgeListfile, ListOfNodes):
    Graph = CreateGraph(OverallEfgeListfile,ListOfNodes)
    Edges_weighis = list(Graph.edges_iter(data='weight'))
    Weight_Sum={}
    for n in ListOfNodes:
        s=0
        for e in Edges_weighis:
            if n==e[0] or n==e[1]:
                s+=e[2]
        Weight_Sum.update({n:s})
    Rep=ListOfNodes[0]
    for n in ListOfNodes[1:]:
        if Weight_Sum[Rep] < Weight_Sum[n]:
            Rep = n
    return Rep
#############################################################################
def FindTWinSentence(Sentence,givenW,POS):
    TW = givenW
    Tokens = WordPunctTokenizer().tokenize(Sentence.lower())
    for token in Tokens:
        if wordnet_lemmatizer.lemmatize(givenW,pos=POS) == wordnet_lemmatizer.lemmatize(token,pos=POS):
           TW,index = token, Tokens.index(token)
    return TW

############################################################################

def InstanceAssignment_Similarity_Based(All_dict,PrimitiveClusters, Remaining,model,OverallEdgeListfile,window_size,W,POS,SimThreshold=0 ):
    C_Sent= 0.5
    C_Sub = 0.5
    i = 0
    print('len Remaining is:',len(Remaining))
    for id in Remaining:
        Sentence = All_dict[id][0].lower()

        TWS = FindTWinSentence(Sentence,W.lower(),POS)

        All_Subs = list(FindSubsNumber(All_dict[id][1:], Threshold=1))
        id_Em_Sent, id_Em_Subs = FindEmbedding(Sentence,window_size, TWS, All_Subs, model)
        Simmilarity = []
        for cluster in PrimitiveClusters:
            Representative = Find_Representative(OverallEdgeListfile, cluster)
            TWR = FindTWinSentence(All_dict[Representative][0].lower(), W.lower(),POS)

            SUBS = list(FindSubsNumber(All_dict[Representative][1:], Threshold=1))
            R_Sent, R_Sub = FindEmbedding(All_dict[Representative][0],window_size, TWR, SUBS, model)
            Sim = C_Sent * CosineSimilarity(id_Em_Sent, R_Sent) + C_Sub * CosineSimilarity(id_Em_Subs, R_Sub)
            Simmilarity.append(Sim)
        MaxValue = max(Simmilarity)

        MaxIndex = Simmilarity.index(MaxValue)
        if MaxValue >= SimThreshold:
            PrimitiveClusters[MaxIndex].append(id)
        else:
            PrimitiveClusters.append([id])
        if i% 50 :
            print('The Number of Processed Instances is: ',i)
        i+=1
        # print(i)
    # PrimitiveClusters=MergeClusters(PrimitiveClusters, OverallEdgeListfile, All_dict, model, Thereshold=0.75)
    return PrimitiveClusters
######################################################################################
def InstanceAssignment_Intersection_Based (All_dict,PrimitiveClusters, Remaining,OverallEdgeListfile,Intersection_Threshold=0 ):
    random.shuffle(Remaining)
    i = 0
    for id in Remaining:
        Sentence = All_dict[id][0]
        All_Subs = list(FindSubsNumber(All_dict[id][1:], Threshold=1))

        Simmilarity = []
        for cluster in PrimitiveClusters:
            Representative = Find_Representative(OverallEdgeListfile, cluster)
            SUBS = list(FindSubsNumber(All_dict[Representative][1:], Threshold=1))
            Sim = len (set(SUBS)& set(All_Subs))
            Simmilarity.append(Sim)
        MaxValue = max(Simmilarity)
        MaxIndex = Simmilarity.index(MaxValue)
        if MaxValue >= Intersection_Threshold:
            PrimitiveClusters[MaxIndex].append(id)
        else:
            PrimitiveClusters.append([id])
        i += 1
    return PrimitiveClusters

def ClustersSimilarity(Clusters,OverallEdgeListfile,All_dict,model):
    SimList=[]
    # print('Len cluster:',len(Clusters))
    i=0
    while(i<len(Clusters)-1):
        start=time.time()
        if len(Clusters[i])==1:
            Repi=Clusters[i][0]
        else:
            Repi=Find_Representative(OverallEdgeListfile, Clusters[i])

        SUBSi = list(FindSubsNumber(All_dict[Repi][1:], Threshold=1))
        R_Senti, R_Subi = FindEmbedding(All_dict[Repi][0], SUBSi, model)

        j=i

        while(j<len(Clusters)-1):
            j+=1
            if len(Clusters[i])>1 and len(Clusters[j])>1:
                # print(i,j)
                continue

            if len(Clusters[j]) == 1:
                Repj=Clusters[j][0]
            else:
                Repj = Find_Representative(OverallEdgeListfile, Clusters[j])
            SUBSj = list(FindSubsNumber(All_dict[Repj][1:], Threshold=1))
            R_Sentj, R_Subj = FindEmbedding(All_dict[Repj][0], SUBSj, model)
            Sim = 0.5 * CosineSimilarity(R_Senti, R_Sentj) + 0.5 * CosineSimilarity(R_Subi, R_Subj)
            SimList.append((i,j,Sim))
            end=time.time()
            # print(i,j,end-start)
            # j+=1

        i+=1

    SimList=sorted(SimList, key=itemgetter(2), reverse=True)
    return SimList
######################################################################################
def Merge(Clusters,i,j):
    NewCluster=[]

    NewCluster.append(Clusters[i]+Clusters[j])
    k=0
    while(k< len(Clusters)):
        if k not in [i,j]:
            NewCluster.append(Clusters[k])
        k+=1
    return NewCluster
######################################################################################
def MergeClusters(Clusters,OverallEdgeListfile,All_dict,model,Thereshold=0.75):
    SortedSimilarity=ClustersSimilarity(Clusters,OverallEdgeListfile,All_dict,model)
    # print(SortedSimilarity)
    MaxSim=SortedSimilarity[0][2]
    while (MaxSim >=Thereshold):
        # print(len(Clusters))
        # print(Clusters)
        # print(SortedSimilarity[0][0],SortedSimilarity[0][1])
        Clusters= Merge(Clusters,SortedSimilarity[0][0],SortedSimilarity[0][1])
        SortedSimilarity = ClustersSimilarity(Clusters, OverallEdgeListfile, All_dict, model)
        MaxSim = SortedSimilarity[0][2]
    return Clusters
######################################################################################
def EvaluationResult(path_to_jar_files_goldfile, PredictionFile):
    f = tempfile.NamedTemporaryFile(delete=False)

    os.system('java -jar '+path_to_jar_files_goldfile+'/vmeasure.jar '+path_to_jar_files_goldfile+'/trial.gold-standard_OneSense.key '+PredictionFile+' all > '+f.name)
    file = open(f.name,'r')
    l=file.readlines()
    file.close()
    v_measure_all = l[-1].strip().split(':')[-1]
    os.remove(f.name)
    f = tempfile.NamedTemporaryFile(delete=False)
    os.system('java -jar ' + path_to_jar_files_goldfile + '/fscore.jar ' + path_to_jar_files_goldfile + '/trial.gold-standard_OneSense.key ' + PredictionFile + ' all > ' + f.name)
    file = open(f.name, 'r')
    l = file.readlines()
    file.close()
    F_measure_all = l[-1].strip().split(':')[-1]
    os.remove(f.name)
    return  float(v_measure_all),float(F_measure_all)

def FindMaxNumberOfClusters(AllCulsterList):
    l=0
    for e in AllCulsterList:
        if len(e) > l:
            l = len(e)
    return l

def WriteToFile(FinalClusters,OutFile):
    Instances = Flatten2dList(FinalClusters)
    Word = Join(FinalClusters[0][0].split('.')[:2],'.')
    o=open(OutFile,'a')
    for a in Instances:
        for c in FinalClusters:
            if a in c:
                id = FinalClusters.index(c)
                break
        o.write(Join([Word,a,Word+'.'+str(id)],' '))
        o.write('\n')
    o.close()
    return
#############################################
def FindWords1(All_dict):
    Words = []
    InsList = list (All_dict)
    for ins in InsList :
        if ins.split('.')[0] not in Words:
            Words.append(ins.split('.')[0])
    return Words
def FindWords2(All_dict):
    Words = []
    InsList = list (All_dict)
    for ins in InsList :
        if (ins.split('.')[0],ins.split('.')[1]) not in Words:
            Words.append((ins.split('.')[0],ins.split('.')[1]))
    return Words
def FindWordIns1(All_dict, w):
    Instances = []
    Ins = list (All_dict)
    for ins in Ins:
        if ins.startswith(w+'.'):
            Instances.append(ins)


    return Instances

#This is a change 

