from Clustering import *
import gensim
import sys
import tempfile
import math
ModelPath='/storage/GoogleNews-vectors-negative300.txt'
# ModelPath='/data/dsm/google-news-vectors/GoogleNews-vectors-negative300.txt'
model=gensim.models.KeyedVectors.load_word2vec_format(ModelPath,binary=False)
print('Model Loaded!')


def main(InputFinalLS,OutFile,ClusteringType = 'LF',InterSectionThreshold = 4,BasicInstances = 0.50,SimThreshold = 0.75,InstanceAssignment = 'Similarity',window_size = 8):
    f = open(InputFinalLS, 'r')
    l = f.readlines()
    f.close()
    All_dict = {}
    for e in l:
        a = e.strip().split('\t')
        All_dict.update({a[0]: a[1:]})

    # Words = FindWords1(All_dict)
    Words = FindWords2(All_dict)
    print(Words)
    f = tempfile.NamedTemporaryFile(delete=False)
    o = open(OutFile,'w')
    CreateEdgeList(InputFinalLS, f.name, InterSectionThreshold=InterSectionThreshold)
    Max_Len = 0

    for word in Words:
        w = word[0]
        pos = word[1]
        print(w)

        Inst = FindWordIns(InputFinalLS, w)
        print('The Number of Instances is: ', len(Inst))
        GraphCandidateNodes = SelectRandomNodes(Inst, int(len(Inst) * BasicInstances))
        Remainig = list(set(Inst) - set(GraphCandidateNodes))
        Graph = CreateGraph(f.name, GraphCandidateNodes)
        if ClusteringType == 'LF':
            PrimitiveClusters = Leader_Follwer_Clustering(Graph)
        elif ClusteringType == 'CW':
            PrimitiveClusters = Chinese_whispers(Graph)
        else:
            print('Clustering Algorithm is not valid')
            return
        if InstanceAssignment == 'Similarity':
            FinalClusters = InstanceAssignment_Similarity_Based(All_dict, PrimitiveClusters, Remainig, model, f.name, window_size, w, pos, SimThreshold=SimThreshold)
        elif InstanceAssignment == 'InterSection':
            FinalClusters = InstanceAssignment_Intersection_Based(All_dict, PrimitiveClusters, Remainig, f.name,Intersection_Threshold=InterSectionThreshold)
        # M_Len = FindMaxNumberOfClusters(FinalClusters)
        M_Len = len(FinalClusters)
        if M_Len > Max_Len :
            Max_Len = M_Len
        WriteToFile(FinalClusters, OutFile)

    os.remove(f.name)
    o.close()
    return
if __name__=="__main__":
    main(sys.argv[1],sys.argv[2])
