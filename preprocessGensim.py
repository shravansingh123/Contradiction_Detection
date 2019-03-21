import subprocess
import re
import sys
import io
from nltk.corpus import stopwords

# input and output files
infile = sys.argv[1]
outfile = sys.argv[2]

stwd = set(stopwords.words('english'))  # for faster searching set  
with open("POStagtwt.txt","w") as f:
    subprocess.call(["postagger/runTagger.sh","--output-format","conll", infile],stdout=f) #calling tokenizer and POS tagger
with io.open("POStagtwt.txt","r",encoding='utf-8') as f:
    tokens=[x.split('\t') for x in f.readlines()]    
#print tokens
stwd = set(stopwords.words('english'))  # for faster searching set  
#col=['N',',','O','^','S','Z','V','L','M','A','R','!','D','P','&','T','X','Y','#','@','~','U','E','$','G','badwords','oov','wordratio','charcnt','stopwords']
temp=list()
with io.open(outfile, 'w') as tweet_processed_text:        
    for t in tokens:
        if(t[0]=='\n'):# new tweet starting
            tweet_processed_text.write(unicode(u''.join(temp)+'\n'))
            del temp[:]
        else:
            t[0]=t[0].lower()            
            if t[0] in stwd:
                continue
            elif(t[1]=='#' and len(t[0])>2):
                temp.append(unicode(t[0][1:]))
            elif(t[1]=='@'):                
                temp.append("@user")
            elif(t[1]=='U'):                
                temp.append("!url")    
            elif(t[1]=='E'):                
                continue
            elif(t[1]=='$'):                
                temp.append("num")        
            elif(t[1]==','):                
                continue
            elif(t[1]=='G'):                
                continue    
            elif(t[1]=='~'):   # : ... rt >>>>             
                continue
            else:
                temp.append(unicode(t[0]))

