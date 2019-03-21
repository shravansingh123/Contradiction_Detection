import subprocess
import gensim
from gensim.models import word2vec
import os
import collections
import smart_open
import random
import sys
from collections import defaultdict
import json
import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib import pylab
import pandas as pd
from textblob import TextBlob as tb
import networkx as nx
import numpy as np
import operator
import nltk
from nltk.corpus import stopwords
import fileinput
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from gensim.test.utils import get_tmpfile
from gensim.corpora import Dictionary
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import re
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import sklearn.svm as svm
from gensim.scripts.glove2word2vec import glove2word2vec
from shutil import copyfile
#Counter({'other_useful_information': 437, 'treatment': 364, 'not_related_or_irrelevant': 271, 
#'disease_signs_or_symptoms': 258, 'prevention': 247, 'disease_transmission': 222, 'deaths_reports': 185})
#Counter({4: 437, 6: 364, 3: 271, 1: 258, 5: 247, 2: 222, 0: 185})
datadir="./data/ebola/"
nodedict="nodedict.bin"
topnig="topnIG.bin"
rtstatus=1 #1 states that retweet will be considered 0 otherwise

def finalprobscore(categorydir):
    GG=TwtstanceIndication(categorydir) #uncomment later    
    with open("Output.txt","a") as f:
        for i,x in enumerate(GG):
            tweets=fetchtweets(categorydir,x.nodes())#tweets structure tuple(user,txt)            
            '''with open(categorydir+"/tweets.bin","wb") as f:
                pickle.dump(tweets,f)
            with open(categorydir+"/tweets.bin","rb") as f:
                tweets=pickle.load(f)    '''
            artscore=ArticulationScore(tweets,categorydir)        
            n=7
            index=sorted(range(len(artscore)),key=lambda x:artscore[x],reverse=True)[0:n]
            print ("for category {0} graph number {1}, top {2} articulate tweets:\n".format(categorydir,i,n))        
            f.writelines("\nfor category {0} graph number {1}, top {2} articulate tweets:\n".format(categorydir,i,n))
            for x in index:
                print (tweets[x][1],artscore[x])
                f.writelines(tweets[x][1]+"\n")
    '''
    #needs to check both relevance and articulation score sorted order and check tweets
    relscore=TopicRelevanceScore()
    index=sorted(range(len(relscore)),key=lambda x:relscore[x],reverse=True)[0:n]
    print("top 5 relevant tweets:\n")
    for x in index:
        print (tweets[x][0],relscore[x])

    #print ("rel index:",len(relscore))
    #print ("arti index:",len(artscore))        
    indexA= [x[0] for x in stanceA]    
    scoreA=[[3/(1/a[1]+1/b+1/c),d] for a,b,c,d in zip(stanceA,[artscore[i] for i in indexA],[relscore[i] for i in indexA],indexA)]
    indexB= [x[0] for x in stanceB]    
    scoreB=[[3/(1/a[1]+1/b+1/c),d] for a,b,c,d in zip(stanceB,[artscore[i] for i in indexB],[relscore[i] for i in indexB],indexB)]
    
    scoreA.sort(reverse=True)
    scoreB.sort(reverse=True)
    summaryA1=[tweets[i][0] for x,i in scoreA[0:n/2]]
    summaryB1=[tweets[i][0] for x,i in scoreB[0:n/2]]
    print("top {0} tweets based on sumSAT:\n".format(n))
    for x in summaryA1:
        print x
    for x in summaryB1:
        print x    
    
    print("top {0} tweets based on HASHTAGsumSAT:\n".format(n))
    tophta=set() # only different hashtags even if top scores are of same hashtags
    tophtb=set()
    summaryA2=list()
    summaryB2=list()
    for score,i in scoreA:
        for ht in tweets[i][1]:
            if ht in Ha and ht not in tophta:
                tophta.add(ht)
                if(len(tophta)>n/2):break
                summaryA2.append(tweets[i][0])
        else:
            continue
        break            
    for sc,i in scoreB:
        for ht in tweets[i][1]:
            if ht in Hb and ht not in tophtb:
                tophtb.add(ht)            
                if(len(tophtb)>n/2):break
                summaryB2.append(tweets[i][0])
        else:
            continue
        break                    
    print tophta
    print tophtb      
    for x in summaryA2:
        print x
    for x in summaryB2:
        print x         
    ''' 
def TopicRelevanceScore():
    # training command will be executed on command prompt not everytime
    #ngram-count -kndiscount -interpolate -text preprocesstrain.txt -lm shravan.lm     #will be run on whole data
    #ngram -lm shravan.lm -ppl preprocesstest.txt -debug 1 >> relevancscore.txt
    with open("relevancescore.txt","w") as f:
        subprocess.call(["ngram","-lm", "shravan.lm","-ppl","preprocesstest.txt","-debug", "1"],stdout=f)    
    relscore=list()
    num=2
    with open("relevancescore.txt","r") as f,open("relevancecheck.txt","w") as f1:
        for i,line in enumerate(f):            
            if(i==num):
                num+=4
                st=line.find("logprob=")
                st+=9
                end=line.find("ppl=")                                
                relscore.append(np.exp(float(line[st:end-1])))#np.exp(line[st:end-1]))
            elif(i%4==0):
                f1.write(line)                
    return relscore
    
def ArticulationScore(tweets,categorydir):
    #make a dictionary of all dictionary words and check the existence(use 66of12, 3of62+2+3lem 
    #and check which one works best or their combinations works best)
    with open("bigDictionary.txt","r") as f:
        oov={x.rstrip() for x in f.readlines()}        
    print ("bigDictionary length:",len(oov))
    #make a dictionary of bad words and count
    with open("badwords.txt","r") as f:
        badwords={x.rstrip():1 for x in f.readlines()}
        
    dftrain=xvalsetter("non_articulate_tweets.txt",badwords,oov,categorydir)        #training data
    ylist=[0 if x<158 else 1 for x in range(0,658) ]
    yseries=pd.Series(ylist)
    #dftrain["y"]=ylist#yseries
    #print dftrain    
    from sklearn.utils import shuffle
    dftrain, yseries = shuffle(dftrain, yseries, random_state=0) # just for completeness incase not shuffled 
    #print dftrain    
    regr = RidgeCV(alphas=(0.001,0.01,0.1,1,10),scoring="neg_mean_squared_error", store_cv_values=True,cv=None)
    regr.fit(dftrain, yseries)
    #print("cross val values:",regr.cv_values_)
    #print("alpha val:",regr.alpha_)
    """scores = cross_val_score(regr, dftrain, yseries, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    print("rmse_score:",rmse_scores)
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())"""
    #trainArticulation(df)
    with open(categorydir+"/tweets.txt","w") as f:
        for t in tweets:
            x=t[0:-1]
            f.write("%s\n" %  x[0])    
    dftest=xvalsetter(categorydir+"/tweets.txt",badwords,oov,categorydir)        #testing   
    # run prediction on the training set to get a rough idea of how well it does
    y_pred = regr.predict(dftest)
    #print("Lasso ", y_pred)    
    return y_pred    

def xvalsetter(filename,badwords,oov,categorydir):
    #pd.set_option('display.max_columns', None) # for showing all columns on screen even if linebreaks are needed
    #pd.set_option('display.expand_frame_repr', False) #will show all the columns on frame(not linebreaks)
    #pd.set_option('display.max_rows', 500) #for showing all the rows(no ...)
    tottwts=0
    with open(filename,"r") as f:
        tottwts=sum(1 for _ in f.readlines())
    with open(categorydir+"/POStagtwt.txt","w") as f:
        subprocess.call(["postagger/runTagger.sh","--output-format","conll", filename],stdout=f) #calling tokenizer and POS tagger
    
    with open(categorydir+"/POStagtwt.txt","r") as f:
        token=[x.split('\t') for x in f.readlines()]    
    stwd = set(stopwords.words('english'))  # for faster searching set  
    col=['N',',','O','^','S','Z','V','L','M','A','R','!','D','P','&','T','X','Y','#','@','~','U','E','$','G','badwords','oov','wordratio','charcnt','stopwords']
    df=pd.DataFrame(0,index=range(0,tottwts),columns=col)
    count=0
    tkncnt=0 #avg number of words length(include url? or emoticon ? or hashtags): exclude them from tkncount
    charcnt=0
    stopcnt=0
    #TBD(extra): proportion of hashtags,number of usernames,number of enoticons,number of ellipsis,
    for t in token:
        if(t[0]=='\n'):# new tweet starting
            df.loc[count,'wordratio']=float(charcnt)/tkncnt
            df.loc[count,'charcnt']=charcnt
            df.loc[count,'stopwords']=float(stopcnt)/tkncnt
            df.loc[count,'badwords']=float(df.loc[count,'badwords'])/tkncnt
            df.loc[count,'oov']=float(df.loc[count,'oov'])/tkncnt
            for x in range(0,25):
                df.iloc[count,x]=float(df.iloc[count,x])/tkncnt
            tkncnt=0
            charcnt=0
            stopcnt=0
            count+=1
        else:
            t[0]=t[0].lower()
            tkncnt+=1
            charcnt+=len(t[0])
            if t[0] in stwd:
                stopcnt+=1                
            #print(t[0],t[1],tkncnt,charcnt)
            # '#' is not removed from hashtag for finding badword ?  url and emoticons and ht are not excluded in OOV
            if(t[0] in badwords or (t[0][0:1]=='#' and len(t[0])>2 and t[0][1:] in badwords)):
                #print ("badwords ",t[0])
                df.loc[count,'badwords']+=1
            if(t[0] not in oov):                
                df.loc[count,'oov']+=1
            df.loc[count,t[1]]+=1
    #print df
    return df
#called only once for setting up data for tweet2vec, after this tweet2vec program will be called to get the required(predictionsProb, labeldict) files
#recall and precision sucks rightnow, needs to be improved by training on atleast 2 million tweets and removing too frequent and rare hashtags
def tweet2vecsetup(Allht):    
    #run this part when new model needs to be trained
    #after activating cenv run #THEANO_FLAGS='device=cuda,floatX=float32' ./tweet2vec_trainer.sh    
    #Overwrite the file, removing empty lines that are removed by preprocessing(RT lines).         
    with open("preprocesstrain.txt") as in_file, open("preprocesstrain.txt", 'r+') as out_file:
        out_file.writelines(line for line in in_file if line.strip())
        out_file.truncate()
    with open("preprocesstrain.txt","r") as f:
        totlines = sum(1 for _ in f)
    vallines=totlines/20
    with open("preprocesstrain.txt","r") as rd,open("tweetstrain.txt","w") as wr,open("tweetsval.txt","w") as wr1: # for storing (hashtag,tweet) tuple so as to use it for tweet2vec        
        for i,line in enumerate(rd):            
            for x in Allht[i]:
                if(i<vallines):wr1.write("%s\t%s" % (x,line))     #validation part                
                else : wr.write("%s\t%s" % (x,line))  

#tweets structure (txt,ht list),ht is hashtag list 
def tweet2vec(categorydir,ht,tweets):    
    #testing set
    with open(categorydir+"/twttestwoprepros.txt","w") as f: 
        f.writelines("%s\n" % (x[0].replace('\n','').replace('\r','')) for x in tweets ) #created by tweets
    subprocess.call([r"python","preprocess.py",categorydir+"/twttestwoprepros.txt", categorydir+"/twttest0.txt"]) #calling preprocessing for test set tweet2vec
    with open(categorydir+"/twttest0.txt") as in_file, open(categorydir+"/twttest0.txt", 'r+') as out_file:
        out_file.writelines(line for line in in_file if line.strip())
        out_file.truncate()   
    with open(categorydir+"/twttest0.txt","r") as rd,open(categorydir+"/tweetstest.txt","w") as wr: # for storing (hashtag,tweet) tuple so as to use it for tweet2vec        
        for i,line in enumerate(rd):            
            temp=str()
            for x in tweets[i][1]:
                temp=temp+str(x)+","
            wr.write("%s\t%s" % ((temp[:-1].replace(',,',',')).lower(),line))                            
    #run this in bhonda #THEANO_FLAGS='device=cuda,floatX=float32' ./tweet2vec_tester.sh #for each category
    copyfile(categorydir+"/tweetstest.txt","tweet2vec-master/misc/tweetstest.txt")
    os.chdir("tweet2vec-master/tweet2vec/")
    subprocess.call(["./tweet2vec_tester.sh",r"THEANO_FLAGS='device=cuda,floatX=float32'"]) #calling preprocessing for test set tweet2vec
    os.chdir(sys.path[0])
    print sys.path[0]
    save_path=categorydir+"/tweet2vec"
    if not os.path.exists(save_path):
        os.mkdir(save_path)            
    copyfile("tweet2vec-master/tweet2vec/result/predictionsProb.npy",save_path+"/predictionsProb.npy")
    copyfile("tweet2vec-master/tweet2vec/best_model/label_dict.pkl",save_path+"/label_dict.pkl")
    copyfile("tweet2vec-master/tweet2vec/result/originalindices.pkl",save_path+"/originalindices.pkl")    
    #with open('%s/data.pkl'%save_path,'r') as f:        
    #    out_data=pickle.load(f)
    #with open('%s/predictions.npy'%save_path,'r') as f: #just gives the sorted order of prediction     
    with open('%s/predictionsProb.npy'%save_path,'r') as f:
        out_pred_prob=np.load(f)        # output prob of test tweets
    #save_path="tweet2vec-master/tweet2vec/model/tweet2vec"
    with open('%s/label_dict.pkl' % save_path, 'rb') as f:
        labeldict = pickle.load(f)# hashtag:index in model softmax layer
    with open('%s/originalindices.pkl' % save_path, 'rb') as f:
        orgindex = pickle.load(f) #order in which out_pred_prob is given out in terms of original index
    tweetprob=list()
    for i,t in enumerate(tweets):
        #print(t[0]) #print(out_data[(orgindex.tolist()).index(i)])
        templs=list()
        for h in ht:            
            templs.append(out_pred_prob.item((orgindex.tolist()).index(i),labeldict[h.lower()]))
        tweetprob.append(templs)        
    return tweetprob            

#returns tweets (tuple(user,hashtag list)), scoreA(index,score),scoreB(index,score)
def TwtstanceIndication(categorydir):
    GG=GraphGeneration(categorydir)        
    #currently tweets that are authored by community member is only included(same ht could have been by other user also);    
    
    return GG
    
#fetch tweets based on hashtags # tweets structure is tuple(user,txt)
def fetchtweetsHs(hs,tweets):    
    twts=list()
    for i,tup in enumerate(tweets):        
        txt = tup[1]        
        temphs=hashtags(txt)
        templist=list()                    
        for y in temphs:
            if(y in hs):#if x is part of either A or B hashtags // check lower case difference ??
                templist.append(y)                      
        if(len(templist)>0):
            twts.append([txt,templist])            
    #print twts        
    print ("tweets selected after IG: {0}".format(len(twts)))
    return twts

def GraphGeneration(categorydir):
    adjdict={} #act as ajacency list with weights node:(adjnode,weights),(adjnode1,weights)
    tweetlst={} # each node(user) and its corresponding tweets
    tempdir=categorydir
    categorydir=categorydir+"/data"
    twtfiles = os.listdir(categorydir)
    for j,filename in enumerate(twtfiles):
        print("Reading file: {0}".format(filename))
        with open(categorydir+"/"+filename) as file:
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
            
    #running it again for retweets that were missed out because of ordering of original tweets after retweet in Dump
    for j,filename in enumerate(twtfiles):
        #print("Reading file: {0}".format(filename))
        with open(categorydir+"/"+filename) as file:
            for i,line in enumerate(file):
                rawstring = json.loads(line)
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
            
    for l,v in adjdict.iteritems():
        if(len(v)>0):                                
            for i,a in enumerate(v):
                if(a[1]%2==0):
                    adjdict[l][i][1]=adjdict[l][i][1]/2 #reduce the weight by half to void double counting

    #for using connections that have weight 2(u->v or v->u or including both ways) 
    adj=list() 
    for l,v in adjdict.iteritems():
        if(len(v)>0):
            stat=0
            temp=list()
            temp.append(l)                        
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
                adj.append(temp)
        
    #adjacency list generation for networkx graph format
    lines=[]
    for item in adj:
        ed=''            
        for item1 in item:
            ed=ed+' '+str(item1)
        lines.append(ed)        
    
    G = nx.parse_adjlist(lines, nodetype = int)
    print(len(G.nodes()),len(G.edges()))
    
    nodes={}
    nx.draw_networkx(G,node_size=2,alpha=0.8,with_labels=False)    
    plt.axis('off')
    plt.savefig(tempdir+'/grimg_'+tempdir+'.png',dpi=1000)
    plt.savefig(tempdir+'/grimg_'+tempdir+'.pdf')
    plt.clf()
    GG=sorted(nx.connected_component_subgraphs(G),key=lambda x:len(x.nodes()), reverse=True)[0:3]
    return GG
    '''Gc = max(nx.connected_component_subgraphs(G), key=len)
    #for line in nx.generate_adjlist(Gc):
    #    print(line)
     
    nodes={}# mapping between actual userid and userid used for graph partionning(starts from 1 )
    count=0
    for line in nx.generate_adjlist(Gc):
        x = [int(i) for i in line.split()]
        count+=1
        nodes[x[0]]=count                    
    
    finaladjlist=list()#adjacency list for graph partitioning(index starts from 1 )
    finaladjlist.append([])
    finaladjlist[0].append(len(Gc.nodes()))
    finaladjlist[0].append(len(Gc.edges()))
    #formatting for graph partitioning algo.(removing first node)
    for line in nx.generate_adjlist(Gc):
        x = [int(i) for i in line.split()]
        temp=list()
        for i,x1 in enumerate(x):
            if(i>0):
                temp.append(nodes[x1])                    
        finaladjlist.append(temp)        
    #for edge u->v, writing corresponding entry v->u(partitioning algo requirement)
    for i,x in enumerate(finaladjlist):        
        if(i>0):
            for k,x1 in enumerate(x):
                if(i not in finaladjlist[x1]):
                    finaladjlist[x1].append(i)
    # writing adjacency list in a file for partitioning process input        
    with open(tempdir+"/graphfile.txt","w") as file:
        for x in finaladjlist:
            for x1 in x:
                file.write("%s " % x1)            
            file.write("\n")    
    
    subprocess.call([r"./metis-5.1.0/exec/bin/gpmetis", tempdir+"/graphfile.txt", "2"]) #calling partitioning executable(metis)
    with open(tempdir+"/graphfile.txt.part.2","r") as file:    #partitioned file having number 1 and 0 for 2 partitioning
        color=[int(i) for i in file.readlines()]    
    nx.draw_networkx(Gc,node_color=color,node_size=2,alpha=0.8,with_labels=False)    
    plt.axis('off')
    plt.savefig(tempdir+'/grimg_'+tempdir+'.png',dpi=1000)
    plt.savefig(tempdir+'/grimg_'+tempdir+'.pdf')        
    return nodes
    '''
def preprocesstwts(directory):
    Allht=list() #for storing all hashtags for tweet2vec processing    
    twtfiles = os.listdir(directory)
    for j,filename in enumerate(twtfiles):
        print("Reading file: {0}".format(filename))
        with open(directory+"/"+filename) as file, open("tweetstrainwoprepros.txt","w") as wr, open("RTtrainwoprepros.txt","w") as wr1:
            rawdata.extend(file.readlines())            
            file.seek(0,0)
            for i,line in enumerate(file):
                try:
                    rawstring = json.loads(line)
                except:
                    print(filename, line)                
                txt = rawstring["text"].lower()
                user = rawstring["user"]["id"]                
                wr1.write("%s\n" % (txt.replace('\n',' ').replace('\r',' ')))    #will be used in gensim doc2vec model                
                if('#' not in txt or txt.strip()=="" or (rtstatus==0 and txt[0:2]=="RT")): #ignore if no hashtag
                    continue
                #for storing all hashtags for tweet2vec(currenly frequency is not considered like in tweet2vec paper(less than 500 and greater than 10k post hashtags are ignored)
                temphs=hashtags(txt)
                temp=list()
                for y in temphs:
                    temp.append(y)                
                # excluding the testing set(categories)? doesn't matter because language model(srilm) and tweet2vec are unsupervised model                
                wr.write("%s\n" % (txt.replace('\n',' ').replace('\r',' ')))    
                Allht.append(temp) #for each tweet in training set this stores the hashtags                         
    subprocess.call([r"python","preprocess.py","tweetstrainwoprepros.txt","preprocesstrain.txt"]) #calling preprocessing for train set for tweet2vec    
    return Allht

#returns the tweets(tuple:tweet txt,user) and hashtags(each row for one tweet) for each category
def fetchtweets(categorydir,users):
    tweets=list()
    tempdir=categorydir
    categorydir=categorydir+"/data"
    twtfiles = os.listdir(categorydir)
    for j,filename in enumerate(twtfiles):
        print("Reading file: {0}".format(filename))
        with open(categorydir+"/"+filename) as file:
            for i,line in enumerate(file):
                try:
                    rawstring = json.loads(line)
                except:
                    print(filename, line)                
                txt = rawstring["text"].lower()
                user = rawstring["user"]["id"]
                #rtstatus is global variable
                if('#' not in txt or txt.strip()=="" or (rtstatus==0 and txt[0:2]=="RT") ): #ignore if no hashtag or not original tweet
                    continue                
                if(int(user) in users):
                    tweets.append((user,txt.replace('\n',' ').replace('\r',' ')))    # adding tuple                                    
    #print tweets
    print ("inside fetchtweets: tweets:{0} users:{1}".format(len(tweets),len(users)))    
    return tweets

#calculate information gain of a tweet for two classes A and B
def TopNInformationGain(ht,n,nodes,color):
    ca=float(len([x for x in color if x==0]))
    cb=float(len(color)-ca)    
    print("ca {0} cb {1}".format(ca,cb))    
    if(ca==0 or cb==0):
        entropy=1 #approximation of IG 
    else:
        entropy=ca/(ca + cb)*np.log((ca + cb)/ca) +cb/(ca + cb)*np.log((ca + cb)/cb)
    print ("entropy",entropy)
    htig=list()
    tothscount=0.0
    for hs,tup in ht.iteritems():
        tothscount+=tup[1]+tup[2]  #total number of times hashtags are used in tweets in com A and B
    for hs,tup in ht.iteritems():
        ca=0.0
        cb=0.0
        users=tup[0]
        hscount=float(tup[1]) +float(tup[2]) #number of times htag is used
        for x in users:#finding users are in com A or B
            if(color[nodes[x]-1]==0):
                ca+=1
            else:
                cb+=1        
        if(ca==0 or cb==0):htig.append([hs,entropy,ca+cb]) #-( ((tothscount-hscount)/tothscount)*cb/(ca + cb)*np.log((ca + cb)/cb))
        else: 
            htig.append([hs,entropy-((hscount/tothscount)*ca/(ca + cb)*np.log((ca + cb)/ca) +((tothscount-hscount)/tothscount)*cb/(ca + cb)*np.log((ca + cb)/cb)),ca+cb])
            #print(ca,cb,htig[hs],first,ca/(ca + cb),np.log((ca + cb)/ca) ,cb/(ca + cb),np.log((ca + cb)/cb))                
        #print(ca,cb,htig[hs],hs,len(htig))        
    #first2pairs = {k: mydict[k] for k in mydict.keys()[:2]}    nice way for dictionary comprehension            
    htig.sort(key = operator.itemgetter(1, 2),reverse=True)    # sorting based on IG and frequency of tweets each ht has
    nthval=htig[n-1][1]
    nhtig=dict()
    count=0
    for ig in htig:
        if(ig[1]>=nthval):
            nhtig[ig[0]]=(ig[1],ht[ig[0]][1],ht[ig[0]][2]) # adding tuple(ig,freqA,freqB) for frequency calculation
            count+=ig[2]
        if(len(nhtig)==n):break    
    print ("total ca+cb:",count)
    return nhtig    

#returns hashtag dictionary where key is hashtag and value is a set having userid(who has tweeted with that hashtag),
#number of times tweeted in community A and B   ht[hashtag]=[set(userid),occurenceA,occurenceB]
def fetchhashtags(categorydir,tweets,nodes,color):    
    ht=dict()
    for i,tup in enumerate(tweets):
        #line=rd.readline()        
        txt = tup[1]
        user = tup[0]
        temphs=hashtags(txt)
        for y in temphs:
            if(y not in ht): # check lower case difference ??
                ht[y]=[set(),0,0] # first item is users and second is number of times hashtag is in tweets by communtyA member and third for communityB( same htag used more than once in a tweet is counted only once)
            ht[y][0].add(user)
            if(color[nodes[user]-1]==0):
                ht[y][1]+=1 # if community A member
            else:
                ht[y][2]+=1                    
    print ("A and B hashtags:".format(len(ht)))
    return ht

def hashtags(txt):
    #UTF_CHARS = ur'a-z0-9_\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff'
    #TAG_EXP = ur'(^|[^0-9A-Z&/]+)(#|\uff03)([0-9A-Z_]*[A-Z_]+[%s]*)' % UTF_CHARS
    TAG_REGEX = re.compile("(?:\#+[\w_]+[\w\'_\-]*[\w_]+)")#re.compile(TAG_EXP, re.UNICODE | re.IGNORECASE)
    #print set(TAG_REGEX.findall(txt))
    return set([x.lower() for x in TAG_REGEX.findall(txt)])
#returns the pshscore(key is hashtag) :normalised IGain  
def HsStanceIndication(ht,tweets,nodes,color,com):   
    #only called once for finding IG(although should be used as singleton class)
    if(HsStanceIndication.state==0):
        nhtig=TopNInformationGain(ht,min(20,len(ht)),nodes,color)  #common for community A and B      
        with open(topnig, 'wb') as f:
            pickle.dump(nhtig,f)
        HsStanceIndication.state=1    
    else:
        with open(topnig, 'r') as f:
            nhtig = pickle.load(f)
    #print(nhtig)    
    pshscore=dict()
    for hs in nhtig.keys():
        # taking community 0 for greater than only
        if(com==0):# for community A
            if(nhtig[hs][1]>nhtig[hs][2]):# if ht is more frequent in comA than comB
                pshscore[hs]=nhtig[hs][0]
        else:
            if(nhtig[hs][1]<=nhtig[hs][2]):
                pshscore[hs]=nhtig[hs][0]
    norsum=float(sum(pshscore.values()))
    for hs in pshscore:
        pshscore[hs]=pshscore[hs]/norsum
    
    print("pshscore :",pshscore)
    return pshscore
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
def multimodel(testfile,ytrain):
    test_corpus = list(read_corpus(testfile,"testgensim.txt", tokens_only=True))    
    count_vect = CountVectorizer(analyzer = "word",ngram_range=(1,2))
    X_train_counts = count_vect.fit_transform([' '.join(x) for x in test_corpus])    
    #tfidf_transformer = TfidfTransformer()
    X_train_tfidf = pd.DataFrame(X_train_counts.todense())#pd.DataFrame((tfidf_transformer.fit_transform(X_train_counts)).todense())
    #print X_train_tfidf
    models = [ RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0), svm.SVC(kernel='rbf'), MultinomialNB(),
    LogisticRegression(random_state=0),]
    folds = 5
    folds = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
    for trn_idx, val_idx in folds.split(X_train_tfidf, ytrain):                
        for model in models: 
            print  model.__class__.__name__       
            model.fit(X_train_tfidf.iloc[trn_idx], [ytrain[i] for i in trn_idx])
            y_pred = model.predict(X_train_tfidf.iloc[val_idx])
            print(classification_report([ytrain[i] for i in val_idx], y_pred))    
    
def multiNaiveBayes(testfile,ytrain):
    test_corpus = list(read_corpus(testfile,"testgensim.txt", tokens_only=True))    
    X_train, X_test, y_train, y_test = train_test_split(test_corpus, ytrain, random_state = 0,test_size=0.1)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform([' '.join(x) for x in X_train])
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    #print type(X_train_tfidf)    
    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    pred=clf.predict(tfidf_transformer.transform(count_vect.transform([' '.join(x) for x in X_test])))    
    print(classification_report(y_test,pred))   
def multiNaiveBayes1(testfile,ytrain):    
    pi = {} #pi is the fraction of each class
    #Set a class index for each document as key
    for i in range(0,ytrain.max()+1):
        pi[i] = 0.0        
    #Extract values from training labels
    lines = ytrain
    total = len(lines)
    #Count the occurence of each class
    for line in lines:
        val = int(line)
        pi[val] += 1        
    #Divide the count of each class by total documents 
    for key in pi:
        pi[key] /= total        
    print("Probability of each class:")
    print("\n".join("{}: {}".format(k, v) for k, v in pi.items()))
    
    #Training data
    train_data = list(read_corpus(testfile,"testgensim.txt", tokens_only=True))        
    dct = Dictionary(train_data)  # initialize a Dictionary
    #print dct
    #print dct.doc2bow(["expert", "cautions"])
    df = pd.read_csv(train_data, delimiter=' ', names=['docIdx', 'wordIdx', 'count'])
    #Training label
    label = []
    train_label = open('/home/sadat/Downloads/HW2_210/20news-bydate/matlab/train.label')
    lines = train_label.readlines()
    for line in lines:
        label.append(int(line.split()[0]))

    #Increase label length to match docIdx
    docIdx = df['docIdx'].values
    i = 0
    new_label = []
    for index in range(len(docIdx)-1):
        new_label.append(label[i])
        if docIdx[index] != docIdx[index+1]:
            i += 1
    new_label.append(label[i]) #for-loop ignores last value

    #Add label column
    df['classIdx'] = new_label

    df.head()
# Function to average all of the word vectors in a given paragraph
def makeFeatureVec(words, model,index2word_set, num_features):    
    featureVec = np.zeros((num_features,),dtype="float32")  # Pre-initialize an empty numpy array (for speed)
    nwords = 0
    # Loop over each word in the tweet and, if it is in the model's vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            #print(word)
            try:
                featureVec = np.add(featureVec,model[word])  
            except:
                print("Oops!",sys.exc_info()[0],"occured.")    
    if(nwords==0):
        nwords=1
    featureVec = np.divide(featureVec,nwords)# Divide the result by the number of words to get the average
    return featureVec
# Given a set of tweets (each one a list of words), calculate the average feature vector for each one and return a 2D numpy array 
def getAvgFeatureVecs(tweets, model, num_features):    
    # Index2word is a list that contains the names of the words in the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    print len(index2word_set)    
    counter = 0
    FeatureVecs = np.zeros((len(tweets),num_features),dtype="float32") # Preallocate a 2D numpy array, for speed
    # Loop through the tweets
    for t in tweets:
       # Print a status message every 1000th tweet
       #if counter%1000. == 0.:
       #    print "tweet %d of %d" % (counter, len(tweets))
       # Call the function that makes average feature vectors
       #print t
       #print len(FeatureVecs),counter
       FeatureVecs[int(counter)] = makeFeatureVec(t, model,index2word_set, num_features)
       counter = counter + 1.
    return FeatureVecs

def read_corpus(filename,output, tokens_only=False):
    subprocess.call([r"python","preprocessGensim.py",filename, output]) #calling preprocessing for train set for Gensim
    print(filename)
    #should remove stop words(done)punctuation(may be not), number(done), emoticons, ellipsis
    #stwd = set(stopwords.words('english'))  # for faster searching set      
    with smart_open.smart_open(output, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

def traindoc2vec(trainfile,testfile):
    test_corpus = list(read_corpus(testfile,"testgensim.txt", tokens_only=True))    
    '''train_corpus = list(read_corpus(trainfile,"traingensim.txt"))        
    print("vector_size=80, min_count=5, epochs=40")
    model = gensim.models.doc2vec.Doc2Vec(vector_size=80, min_count=5, epochs=40)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)    
    model.save("doc2vec_model")
    '''
    model = gensim.models.doc2vec.Doc2Vec.load("doc2vec_model")  # you can continue training with the loaded model!    
    trainvec = [model.infer_vector(t) for t in test_corpus] #train vectors for classifier    
    #train_corpus = list(read_corpus(trainfile,"traingensim.txt",tokens_only=True))    
    #testvec = [model.infer_vector(t) for t in train_corpus] # test vector for classifiers
    testvec=[]
    return trainvec,testvec,model
    '''doc_id=12
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    print(u'Test Document {0}: {1}'.format(doc_id," ".join(test_corpus[doc_id])))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s:%s'%(label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
    '''    
def avgWord2Vec(testfile,model,num_features):
    test_corpus = list(read_corpus(testfile,"testgensim.txt", tokens_only=True))    
    #print test_corpus[:4]        
    # Initialize the "CountVectorizer" object, which is scikit-learn's  for bag of words tool.  
    #vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, preprocessor = None,stop_words = 'english', ngram_range=(1,2)) 
    # fit_transform() does two functions: First, it fits the model and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of # strings.    
    #testvec = vectorizer.fit_transform([' '.join(x) for x in test_corpus])
    # Numpy arrays are easy to work with, so convert the result to an array
    #testvec = testvec.toarray()
    testvec = getAvgFeatureVecs( test_corpus, model, num_features )
    return testvec   

def callclassifier(trainx,ytrain,filename,embedml):
    trainx = pd.DataFrame(trainx)
    # A parameter grid for XGBoost
    params = {
        'min_child_weight': [0.25,0.5,1],
        'gamma': [1,1.5,1.75,2],
        'subsample': [0.4,0.6, 0.8],
        'colsample_bytree': [0.6,0.8, 1.0],
        'max_depth': [8,10,12,14]
        }
    folds = 5
    #Cs = [0.001, 0.01, 0.1, 1, 10]
    #gammas = [0.001, 0.01, 0.1, 1]
    param_grid = { 'C': np.power(10.0, np.arange(-5, 5))
         , 'solver': ['newton-cg','lbfgs'] }
    #param_grid = {'C': Cs, 'gamma' : gammas}
    #param_comb = 30
    '''xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.8, gamma=0.75, learning_rate=0.01,
       max_delta_step=0, max_depth=8, min_child_weight=0.5, missing=None,
       n_estimators=200, n_jobs=1, nthread=1, objective='multi:softprob',
       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=True, subsample=0.4)
    XGBClassifier(learning_rate=0.01, n_estimators=200, silent=True, nthread=1)'''
    #grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=folds)
    #grid_search = GridSearchCV(LogisticRegression(penalty='l2', random_state=777, max_iter=1000), param_grid, cv=folds)
    #grid_search.fit(trainx, ytrain)
    #print grid_search.best_params_
    lg=LogisticRegression(C=1,solver='newton-cg',penalty='l2', class_weight='balanced',random_state=777, max_iter=1000,multi_class='multinomial')
    #print(pd.DataFrame(grid_search.cv_results_))
    #svc=svm.SVC(kernel='rbf',C=10,gamma=0.005)
    folds = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
    for trn_idx, val_idx in folds.split(trainx, ytrain):
        lg.fit(trainx.iloc[trn_idx], [ytrain[i] for i in trn_idx])
        y_pred = lg.predict(trainx.iloc[val_idx])
        print(classification_report([ytrain[i] for i in val_idx], y_pred))
        print(confusion_matrix([ytrain[i] for i in val_idx], y_pred))
        with open(filename,"r") as f:
            twt=[x for x in f.readlines()]
        for i in range(0,len(y_pred)):            
            if(ytrain[val_idx[i]]==2 and y_pred[i]==4):
                print("\npredicted is 4 actual is 2:")
                print twt[val_idx[i]]            
            #elif (ytrain[val_idx[i]]==3 and y_pred[i]==4):
            #    print("\npredicted is 4 actual is 3:")
            #    print twt[val_idx[i]]    


    #random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='accuracy', cv=skf.split(trainx,ytrain), verbose=3, random_state=1001 )
    
    # Here we go
    '''start_time = timer(None) # timing starts from this point for "start_time" variable
    random_search.fit(trainx, ytrain)
    timer(start_time) # timing ends here for "start_time" variable
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)
    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv('xgb-random-grid-search-results-01.csv', index=False)'''
    #random_search.predict()
    '''
    #model = RandomForestClassifier(n_estimators=80)#,min_samples_split=15,max_features=21,criterion="entropy")#,class_weight="balanced")
    #folds = StratifiedKFold(n_splits=5, shuffle=True, random_state = 45)
    # Go through folds    
    for trn_idx, val_idx in folds.split(trainx, ytrain):
        #print val_idx
        model.fit([trainx[i] for i in trn_idx], [ytrain[i] for i in trn_idx])
        y_pred = model.predict([trainx[i] for i in val_idx])
        print(classification_report([ytrain[i] for i in val_idx], y_pred))    
    #confusion_mtx = confusion_matrix(y_vald, y_pred) 
    #print(confusion_mtx)    
    '''
    return lg
#returns classifier
def classify():    
    my_csv = pd.read_csv("data/trainebola.csv",sep='\t')
    tweets = my_csv.tweet_text    
    print collections.Counter(pd.Series(my_csv.choose_one_category))
    encoder=LabelEncoder()
    ytrain=encoder.fit_transform(pd.Series(my_csv.choose_one_category))    
    lemap = dict(zip(encoder.transform(encoder.classes_),encoder.classes_))    
    print collections.Counter(pd.Series(ytrain))
    with open("data/tempebola.txt","w") as f:
        f.writelines("%s\n" %(twt) for twt in tweets)
    #considered only tweets not retweets : done
    #temebola.txt is the training vector for classifier
    #trainx,testx,doc2vecml=traindoc2vec("RTtrainwoprepros.txt","data/tempebola.txt")    #using doc2vec for getting tweet vectors
    #model1=callclassifier(trainx,ytrain,"data/tempebola.txt",doc2vecml)    
    #multiNaiveBayes("data/tempebola.txt",ytrain)
    #multimodel("data/tempebola.txt",ytrain)
    #ytest = model1.predict(testx)
    #print(len(ytest))
    #with open("RTtrainwoprepros.txt","r") as f:
    #    for i,line in enumerate(f):
    #        print(line,lemap[ytest[i]])

    #using averaging of word2vec word vector for finding classification accuracy    
    #glove_input_file = 'glove.840B.300d.txt'#'glove.twitter.27B.200d.txt'
    '''word2vec_output_file = 'word2vec.txt'
    #glove2word2vec(glove_input_file, word2vec_output_file)
    modelw2v = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file,binary=False)#('crisisNLP_word_vector.bin', binary=True)
    #modelw2v=0
    trainx=avgWord2Vec("data/tempebola.txt",modelw2v,300)        
    print("trainx ",len(trainx))
    model2=callclassifier(trainx,ytrain,"data/tempebola.txt",modelw2v)        
    testx=avgWord2Vec("RTtrainwoprepros.txt",modelw2v,300) #so that preprocessgensim.py can process this           
    print("testx ",len(testx))
    print [x for x in testx.flatten() if np.isnan(x)]
    ytest = model2.predict(testx)
    with open("ytest.bin","wb") as f:
        pickle.dump(ytest,f)
    '''
    with open("ytest.bin","rb") as f:
        ytest=pickle.load(f)    
    print collections.Counter(pd.Series(ytest))
    with open("RTtrainwoprepros.txt","r") as f:
        tweettext=f.readlines()    
    for i in range(max(lemap.keys())+1):
        index=[k for k,x in enumerate(ytest) if x==i]        
        catdata=pd.Series(rawdata).iloc[index]        
        cattwttext=pd.Series(tweettext).iloc[index]        
        if not os.path.exists(str(i)):
            os.mkdir(str(i))
            os.mkdir(str(i)+"/data")
        with open(str(i)+"/data/rawdata.txt","w") as f,open(str(i)+"/tweettext.txt","w") as f1:
            f.writelines(catdata)
            f1.writelines(cattwttext)
        finalprobscore(str(i))
    
rawdata=list()    #holds the json data of all the files(one big file)
Allht=preprocesstwts(datadir)
#tweet2vecsetup(Allht)    
classifier=classify()
