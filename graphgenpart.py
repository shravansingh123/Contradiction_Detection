import os
import sys
from collections import defaultdict
import json
import pickle
import matplotlib.pyplot as plt
from matplotlib import pylab
import pandas as pd
from textblob import TextBlob as tb
import networkx as nx

def GraphGeneration(datadir):
    adjdict={}
    tweetlst={}

    twtfiles = os.listdir(datadir)
    for j,filename in enumerate(twtfiles):
        print("Reading file: {0}".format(filename))
        file = open(datadir+filename)
        for i,line in enumerate(file):
            try:
                rawstring = json.loads(line)
            except:
                print(filename, line)
            
            txt = rawstring["text"]
            if('#' not in txt):
                continue
            rwtcount = int(rawstring["retweet_count"])
            if(rwtcount >0 and txt[0:2]!="RT" and 'retweeted_status' not in rawstring): #original tweet
                #print("inside if "+txt)
                usersec=rawstring["user"]
                node=int(usersec['id'])
                if(node not in tweetlst):
                    tweetlst[node]=list() # set can be used to check duplicates?
                else:
                    tweetlst[node].append(int(rawstring['id']))
                if(node not in adjdict):
                    adjdict[node]=list()
                #print("inside IF Node:{0} and tweetid:{1}".format(node,int(rawstring['id'])))
            elif(rwtcount >0 and txt[0:2]=="RT" and 'retweeted_status' in rawstring): #retweet
                node=int(rawstring["retweeted_status"]["user"]["id"])
                node1=int(rawstring["user"]["id"])
                if(node in tweetlst and int(rawstring["retweeted_status"]["id"]) in tweetlst[node]):
                    stat=0
                    for index,l in enumerate(adjdict[node]):
                        if(l[0]==node1):#if already present in adjdict then increase the weight
                            stat=1;
                            adjdict[node][index][1]+=1;
                            break;
                    if(stat==0):#otherwise add a new node in the list with weight 1
                        adjdict[node].append(list([node1,1]))            
                #print("inside ELIF Node:{0} Node1:{1} and tweetid:{2}".format(node,node1,int(rawstring["retweeted_status"]["id"])))
                #print("inside elif "+txt)
        file.close()        
    #running it again for retweets that were missed out because of ordering of original tweets after retweet in Dump
    for j,filename in enumerate(twtfiles):
        #print("Reading file: {0}".format(filename))
        file = open(datadir+filename)
        for i,line in enumerate(file):
            try:
                rawstring = json.loads(line)
            except:
                print(filename, line)
            
            txt = rawstring["text"]
            if('#' not in txt):
                continue
            rwtcount = int(rawstring["retweet_count"])
            
            if(rwtcount >0 and txt[0:2]=="RT" and 'retweeted_status' in rawstring): #retweet
                node=int(rawstring["retweeted_status"]["user"]["id"])
                node1=int(rawstring["user"]["id"])
                if(node in tweetlst and int(rawstring["retweeted_status"]["id"]) in tweetlst[node]):
                    stat=0
                    for index,l in enumerate(adjdict[node]):
                        if(l[0]==node1):#if already present in adjdict then increase the weight
                            stat=1;
                            adjdict[node][index][1]+=1;
                            break;
                    if(stat==0):#otherwise add a new node in the list with weight 1
                        adjdict[node].append(list([node1,1]))            
        file.close()                
    for l,v in adjdict.iteritems():
        if(len(v)>0):                                
            for i,a in enumerate(v):
                if(a[1]%2==0):
                    adjdict[l][i][1]=adjdict[l][i][1]/2 #reduce the weight by half to void double counting

    #for using connections that have weight 2(u->v or v->u or including both ways) 
    adj=list() #adjacency list for graph partitioning index starts from 1
    adj.append([])
    nodedict=dict() # mapping between actual userid and userid used for graph partionning(starts from 1 )
    count=0
    for l,v in adjdict.iteritems():
        if(len(v)>0):
            stat=0
            temp=list()
            for a in v:
                if(a[1]>1): # when weight is already greater than 1 then an edge will exist
                    stat=1                        
                    temp.append(a[0])                        
                else: # otherwise see whether a[0] also retweeted l's tweet
                    if(a[0] in adjdict):
                        v1=adjdict[a[0]]
                        if(len(v1)>0):
                            for a1 in v1:
                                if(a1[0]==l):
                                    temp.append(a[0]) # because weight is now at least 2 even if this one has only weight 1
                                    stat=1
                                    break
            if(stat==1):# adding nodes that will be used in graph
                count+=1
                nodedict[l]=count
                adj.append(temp)
    
    # for converting nodes to index(starting from 1)that are still in userid format(for partitioning algo)
    for i,item in enumerate(adj):
        for k,item1 in enumerate(item):
            if(item1 in nodedict):
                adj[i][k]=nodedict[item1]
                #print(item1,adj[i][k])
            else:
                count+=1
                adj[i][k]=count
                nodedict[item1]=count
                #print(item1,count)
    
    #for i,item in nodedict.iteritems():
    #    print(i,item)       

    #formating adjacency list for partitioning(metis) algo
    finaladjlist=adj + [ [] for _ in range(count-len(adj)+1)]
    #print(len(finaladjlist),count-len(adj))
    for i,item in enumerate(finaladjlist):
        if(i>0):                
            for k,item1 in enumerate(item):
                if(i in finaladjlist[item1]):# for partitioning it is ok                    
                    continue
                else:# when corresponding entry in item1 node for ith node is not there then add(partitioning algo)
                    finaladjlist[item1].append(i)                 
    #adjacency list generation for networkx graph format
    lines=[]
    for i,item in enumerate(finaladjlist):
            if(i>0):
                ed=''+str(i)            
                for k,item1 in enumerate(item):
                    if(i==item1):
                        finaladjlist[i].pop(k)
                    else:
                        ed=ed+' '+str(item1)
                    if(i<item1):
                        finaladjlist[item1].remove(i)
                lines.append(ed)
                #if(i<100):
                    #print(ed)
    
    G = nx.parse_adjlist(lines, nodetype = int)
    print(len(G.nodes()),len(G.edges()))
    Gc = max(nx.connected_component_subgraphs(G), key=len)
    #for line in nx.generate_adjlist(Gc):
    #    print(line)
     
    file = open("graphfile.txt","w")     
    
    del finaladjlist[:]
    print(len(finaladjlist))
    nodes={}
    count=0
    for line in nx.generate_adjlist(Gc):
        x = [int(i) for i in line.split()]
        count+=1
        nodes[x[0]]=count                    
    
    finaladjlist.append([])
    finaladjlist[0].append(len(Gc.nodes()))
    finaladjlist[0].append(len(Gc.edges()))
    for line in nx.generate_adjlist(Gc):
        x = [int(i) for i in line.split()]
        temp=list()
        for i,x1 in enumerate(x):
            if(i>0):
                temp.append(nodes[x1])                    
        finaladjlist.append(temp)        
    for i,x in enumerate(finaladjlist):        
        if(i>0):
            for k,x1 in enumerate(x):
                if(i not in finaladjlist[x1]):
                    finaladjlist[x1].append(i)
            
    file = open("graphfile.txt","w")     
    for x in finaladjlist:
        for x1 in x:
            file.write("%s " % x1)            
        file.write("\n")    
    file.close()
    
    file=open("graphfile.txt.part.2","r")    
    color=[int(i) for i in file.readlines()]
    print color
    nx.draw_networkx(Gc,node_color=color,node_size=2,alpha=0.8,with_labels=False)
    #plt.axis('equal')            
    plt.axis('off')
    plt.savefig('grimg.png',dpi=1000)
    plt.savefig('grimg.pdf')

GraphGeneration("./data/")
