# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:58:32 2018

Third to run
 
Generates random navigation points around the original

- need to make this so it spawns more automatically
- how many iterations for the particle filter???
@author: jackm
"""


#get_ipython().magic('reset -sf') 
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

samNum = 50

mu, sigma = 0.,.5
start, end = 0,1500
data = pd.read_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\nav_files\\navTrackNE.csv')
count = 0 
dfs = {}
pointNos = np.arange(0,samNum,1)
for points in pointNos:
    xx = []
    yy = []
    offX = []
    offY = []
    for i in np.arange(start,end):
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
    
    #print("Run Number:\t",points,df)






""""
lh = {}
hm = {}
n, _ = (len(pointNos),len(pointNos))
colors = mpl.cm.rainbow(np.linspace(0, 1, n))
fig, ax1 = plt.subplots()
for color, PP in zip(colors, pointNos):
    lh[PP] = ax1.plot(dfs[PP].newx, dfs[PP].newy, 'r.',label=PP,markersize=2,color=color) 
    #hm[PP] = ax1.plot(dfs[PP].px,dfs[PP].py,'rx')
    plt.title("")
    
ax1.set_xlabel('X')
#ax1.set_ylabel('Y', color='b')
#ax1.tick_params('y', colors='b')
ax1.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
#ax1.plot(data.iloc[:2,1],data.iloc[:2,2],'bx',markersize=20)                              #to plot the parent points
#ax1.set_ylim([80,100])
#ax1.set_xlim([80,140])


ax1.legend(loc="upper left", bbox_to_anchor=[0, 1],
           ncol=2, shadow=True, title="Track", fancybox=True,prop={'size': 7})

#ax1.set_facecolor("grey")
plt.savefig('myfig2.png', dpi=1000)
"""