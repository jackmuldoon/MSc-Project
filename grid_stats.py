# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:17:57 2018

Second Script 

- this creates the grid over the area that wants to be statistically checked and reads in the files
- it transforms the grid to a localised grid 
- it uses the grid spacing to create a matrix of the grid squares, with the grid 'ID's' stored in the lookup table, (basically creating a unique ID)
- A list is created from each of values in that square and the STDEV, COEFF and MEAN calculated 
- Histograms plotted of stats


--- NOTES ----- 
- potentially encorporate nav_ran_g\en.py into this to generate random nav points for each iteration????
@author: jackm
"""

#get_ipython().magic('reset -sf') 

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import utm
import time
#import threading
import datetime
import multiprocessing as mp

start, end = 0,2000
gridSize = 0.25
length = end-start
Totals = {}
output = mp.Queue()
#for k in np.arange(0,len(dfs)):

downE, downN = 633043.4,5634150.6
upE, upN = 633244.9,5634319.7



mu, sigma = 0.,.5
data = pd.read_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\nav_files\\navTrackNE.csv')
count = 0 
dfs = {}

"""
def nav_random(pointNos):
    for points in pointNos:
        xx = []
        yy = []
        offX = []
        offY = []
        for i in np.arange(start,end+1):
            parX = data.iloc[i,2]
            parY = data.iloc[i,1]       
            offsetX = np.random.normal(mu, sigma)
            newX = parX+offsetX
            offsetY = np.random.normal(mu, sigma)
            newY = parY+offsetY
            #if np.sqrt((offsetX*offsetX)+(offsetY*offsetY)) >= 5:
            #   newX, newY = ("ABORT!","ABORT!")
            if points == 0:
                x,y = 0,0
            xx.append(newX)
            yy.append(newY)
            offX.append(offsetX)
            offY.append(offsetY)      
            if i >= end:                   # if you just want to plot a few points change this
                break        
        #cols = ('newx','newy','px','py')
        df = pd.DataFrame()
        df['offx'] = offX
        df['offy'] = offY
        dfs[points] = df
        df.to_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\nav_files\\random_nav_'+str(points)+'points.csv')
                                                      
def squaring(minE,minN,maxE,maxN):
    nheight = maxN-minN
    elength = maxE-minE
    if (nheight < elength):
        maxN = minN+ elength
        maxE = maxE
    if (nheight > elength):
        maxE = minE+ nheight
        maxN = maxN
    nheight = maxN-minN
    elength = maxE-minE
    return minE, minN, elength, nheight
    
def grid_size(el,nh,gridS):
    #nGridSize = nheight/200                                                         # sets the grid size
    #eGridSize = elength/200
    eRange = np.arange(0,nh,gridS)
    nRange = np.arange(0,el,gridS)
    return eRange,nRange
     
def plot(cof):    
    x = cof
    hist, bins = np.histogram(x, bins=100)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist,align='center',  width=width) 
    plt.ylabel("Frequency")
    plt.xlabel("Coefficient of Variation")
    #plt.savefig('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\Figures\\50_VariationPlot_'+str(k)+'.png', dpi=1000)
    plt.show()   
    total0 = sum(cof)
    print(4)
                                                                ### read the lat and long from the xyz data and convert to eastings and northings ###
def data_parse(data,mE,mN,dfs,k,t): #output
    dct = {}     
    every = 2
    for i in np.arange(0,len(data),every):  #len(data)
        norths =[]
        easts = []
        zs = []
        x = list(utm.from_latlon(data.iloc[i,1], data.iloc[i,2]))[0]                # parses the x,y,z data
        y = list(utm.from_latlon(data.iloc[i,1], data.iloc[i,2]))[1]
        """
        if pp == 0:
            x = x + dfs[k]['offx'][t-start]                                           # adds the offsets to the data
            y = y + dfs[k]['offy'][t-start]
        else:
            x = x + dfs[k]['offx'][t-start] + dfs[pp]['offx'][t-start]                                             # adds the offsets to the data
            y = y + dfs[k]['offy'][t-start] + dfs[pp]['offy'][t-start]
        """
      
        x = x + dfs[k]['offx'][t-start]                                           # adds the offsets to the data
        y = y + dfs[k]['offy'][t-start]
        
        z = data.iloc[i,3]      
        offX = x-minE                                                               # takes to local grid
        offY = y-minN
        if (offY/gridSize) % 1 > 0:
            yInd = (math.ceil(offY/gridSize)-1)                                    # finds the index of the grid
        if (offX/gridSize) % 1 > 0:
            eInd = (math.ceil(offX/gridSize)-1)
        ee = eRange[eInd]                                                           # pulls the relevant bin
        nn = nRange[yInd]
        lu = lookup[ee][nn]       
        if 'lst_%s' % lu in dct:                                                    # compiles the z heights of the grid square
            dct['lst_%s' % lu].append(z)      
        else:
            dct['lst_%s' % lu] = []
            dct['lst_%s' % lu].append(z)     
    #output.put(z)      
    #stdevs = []
    #means = []
    cof = []
    for key, value in dct.items():        
    #    stdevs.append(np.std(value))
    #    means.append(np.mean(value))
        cof.append(np.std(value)/np.mean(value))     
    tot = sum(cof)   
    
    """
    end_time = time.time() 
    loopT = end_time - start_time 
    tLeft = str(datetime.timedelta(seconds=((end-t)/5)*loopT))   
    perc = (t-start)/(end-start)
    print(perc*100,"\t Remaining time:",tLeft) 
    """
    return(tot, dct)

def nav_select(ds,statType,iN):   
    dat = {}
    for i in np.arange(0,ds):
        d = pd.read_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\stats\\'+str(statType)+'\\p_'+str(i)+'.csv', index_col = 0)
        df = pd.DataFrame(d)
        dat[i] = df 
    sums = []  
    keys = []
    mns = []
    df = pd.DataFrame()
    count = 0
    for key, value in dat.items():
        keys.append(count)
        sums.append(sum(value.iloc[:,0]))
        mns.append(np.mean(value.iloc[:,0]))
        count += 1
    df['k'] = keys
    df['s'] = sums
    df['m'] = mns
    opti = df.loc[df['m'].idxmin()]
  
    return(opti,df.iloc[0,2],df)
   
    
########################   working section ############################
minE, minN, elength, nheight = squaring(downE,downN,upE,upN)
eRange,nRange = grid_size(elength, nheight, gridSize)
lookup = pd.DataFrame(columns = eRange, index= nRange)
lookCount = 0
for i in eRange:
    for j in nRange:
        lookup[i][j] = lookCount
        lookCount += 1           
times = []

### ping parse
con = 10
stats = []
itNum = 0
#parts = 5

#nav_random(np.arange(0,end+25,1))
samNum = 10
procStats = {}
tims = {}
amountOfSamples = np.arange(0,end+25)
dffs = {}
for dictEnt in amountOfSamples:
    dffs[dictEnt] = pd.read_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\nav_files\\random_nav_'+str(dictEnt)+'points.csv')
print('dicts made')

#### below creates the stats files for each particle #####
for kk in amountOfSamples:
    start_time = time.time()
    #nav_random(pointNos, itNum)  
    c,s,ms = {},{},{}
    if itNum == 0:
        parent = 0
    #for kk in pointNos:
    for t in np.arange(start,end,5):
        dat = pd.read_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\xyz_files\\ping'+str(t)+'.csv') 
        output,dictt = data_parse(dat, minE, minN,dffs,kk,t)  #output
        #print("it:\t",itNum,"part:\t",kk,"ping:\t",t)
    stdevs = []
    means = []
    cof = []
    for key, value in dictt.items():        
        stdevs.append(np.std(value))
        means.append(np.mean(value))
        cof.append(np.std(value)/np.mean(value))     
        #times.append(ttt)
        c[kk] = pd.DataFrame(cof)
        s[kk] = pd.DataFrame(stdevs)
        ms[kk] = pd.DataFrame(means)
        c[kk].to_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\stats\\coff\\jeff.csv')
        s[kk].to_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\stats\\stdevs\\jeff1.csv')
        ms[kk].to_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\stats\\means\\jeff2.csv')
    print('Particle: ',kk, 'time:',time.time() - start_time,'\n')
    
        #av = np.mean(times)     
#print('Stats csvs compiled \t','time taken:',time.time() - start_time,'\n')
 """          
       
"""    
stats = []
itNum = 0
samNum = 10
dataSummary = pd.DataFrame()
col1,col2,col3 = [],[],[]
while samNum < 2024:         
    route,zeroPart,dataframe = nav_select(samNum, "coff", itNum)
    stats.append(route[2])                                                      #making the list of cof sums
    parent = route[0]
    #print(parent)
    if itNum > 0:
        con = abs(stats[itNum-1]-stats[itNum])
    itNum += 1
    procStats[samNum] = stats  
    print("Particle Complete: \t" , samNum, route[2])
    tims[samNum] = (time.time() - start_time)
    col1.append(samNum)
    col2.append(route[2])
    col3.append(route[0])
    samNum += 25
dataSummary['PartNo'] = col1
dataSummary['AvCoff'] = col2    
dataSummary['optipart'] = col3 
print(zeroPart)  

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dataSummary['PartNo'],dataSummary['AvCoff'], label = 'AvCoff',linestyle='-', marker='|')
ax.axhline(y=zeroPart,  color ='red',xmin = 0.05,xmax= 0.95, label = 'Parent')
ax.set_ylabel('Average Coefficient of Variation')
ax.set_xlabel('Number of Spawned Particles')
ax.legend()


fig.savefig('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\LATEX\Figures\\particles_100.jpg',dpi=300)
"""
"""
points = np.arange(10,2025,25)
fig = plt.figure()
ax = fig.add_subplot(111)
ax2 =  ax.twinx()

ax.plot(dataSummary['PartNo'],dataSummary['AvCoff'], label = 'AvCoff',linestyle='-', marker='|')
ax2.plot(points,points*16, label = 'Processing time in seconds',color = 'green')
ax.axhline(y=zeroPart,  color ='red',xmin = 0.05,xmax= 0.95, label = 'Parent')
ax.set_ylabel('Average Coefficient of Variation')
ax2.set_ylabel('Processing Time in seconds')
ax.set_xlabel('Number of Spawned Particles')
ax.legend()
ax2.legend(loc = (0.1,0.75))
fig.savefig('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\LATEX\Figures\\particles_100_timed.jpg',dpi=300)



###### plotting histograms ######
"""
pt = 0
df = pd.read_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\stats\\coff\\p_'+str(pt)+'.csv', index_col = 0)
x = df.iloc[:,0]
fig = plt.figure()
ax = fig.add_subplot(111)
numBins = 50
ax.hist(x,numBins,color='green',alpha=0.8,edgecolor='black')
ax.set_xlabel('Coefficient of Variation')
ax.set_ylabel('Frequency')
plt.show()
fig.savefig('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\LATEX\Figures\\histo_p'+str(pt)+'.jpg',dpi=300)



