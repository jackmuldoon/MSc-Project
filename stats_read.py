# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 20:16:30 2018

@author: jackm

##### dfs is the length of the dict containing the altered nav points #####

- squaring and grid_size are the same as grid_stats
- nav_select chooses the optimum route
- bath_edit runs through all of the xyz files and offsets them to the optimum route
- surf_create creates the surface based upon the averages
        - would be good to create a cube style system- based upon radiuses
- for statType choose between coff, means, stdevs
"""
#get_ipython().magic('reset -sf') 
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import utm
import time
import datetime
from latlon_wgs84 import metres_to_latlon

start, end = 0,1500
downE, downN = 633114.4,5634185.1
upE, upN = 633264.8,5634346.4
gridSize = 0.25

#statType = coff

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
    eRange = np.arange(0,nh,gridS)
    nRange = np.arange(0,el,gridS)
    return eRange,nRange
"""
def nav_select(ds,statType):   
    dat = {}
    for i in np.arange(0,ds):
        d = pd.read_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\stats\\'+str(statType)+'\\particle_num_'+str(i)+'.csv', index_col = 0)
        df = pd.DataFrame(d)
        dat[i] = df 
    sums = []  
    keys = []
    df = pd.DataFrame()
    count = 0
    for key, value in dat.items():
        keys.append(count)
        sums.append(sum(value.iloc[:,0]))
        count += 1
    df['k'] = keys
    df['s'] = sums
    opti = df.loc[df['s'].idxmin()]
    return(opti)
"""
def bath_edit(r,samNum,s,e):
    nav = pd.read_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\nav_files\\random_nav_'+str(samNum)+'points.csv')
    lat, long, z = [],[],[]
    for p in np.arange(s,e,1):   #len(nav)
        offx = nav.loc[p,'offx']
        offy = nav.loc[p,'offy']
        dat = pd.read_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\xyz_files_small_ATTITUDE\\ping'+str(p)+'.csv')
        for i in np.arange(0,len(dat)):
            newLat, newLong = metres_to_latlon(dat.loc[i,'lat'],dat.loc[i,'long'],offx,offy)
            lat.append(newLat)
            long.append(newLong)
            z.append(dat.loc[i,'z'])
    xyzData = pd.DataFrame()
    xyzData['lat'] = lat
    xyzData['long'] = long
    xyzData['z'] = z
    return(xyzData)    
    
def surf_create(data,minE,minN,gridSize,lookup):
    surf_dct = {}     
    for i in np.arange(0,len(data)):  #len(data)
        norths =[]
        easts = []
        zs = []
        x,y,reaon,ekofkd = list(utm.from_latlon(data.iloc[i,0], data.iloc[i,1]))               # parses the x,y,z data
        z = data.iloc[i,2]      
        offX = x-minE                                                               # takes to local grid
        offY = y-minN
        if (offY/gridSize) % 1 > 0:
            yInd = (math.ceil(offY/gridSize)-1)                                    # finds the index of the grid
        if (offX/gridSize) % 1 > 0:
            eInd = (math.ceil(offX/gridSize)-1)
        ee = eRange[eInd]                                                           # pulls the relevant bin
        nn = nRange[yInd]
        lu = lookup[ee][nn]       
        if 'lst_%s' % lu in surf_dct:                                                    # compiles the z heights of the grid square
            surf_dct['lst_%s' % lu].append(z)      
        else:
            surf_dct['lst_%s' % lu] = []
            surf_dct['lst_%s' % lu].append(ee+minE)    
            surf_dct['lst_%s' % lu].append(nn+minN)    
            surf_dct['lst_%s' % lu].append(z)     
    return(surf_dct)

###############   working code #######################
minE, minN, elength, nheight = squaring(downE,downN,upE,upN)
eRange,nRange = grid_size(elength, nheight, gridSize)
lookup = pd.DataFrame(columns = eRange, index= nRange)
lookCount = 0
for i in eRange:
    for j in nRange:
        lookup[i][j] = lookCount
        lookCount += 1
for samNum in (0,6,86,1960):
    rNum = route['k']
    xyz = bath_edit(rNum,samNum,start,end)                  ### samNum = amount of tracks
    chartDat = surf_create(xyz,minE,minN,gridSize,lookup)                       ### data is the total xyz 
    
    ###### stats #####
    gridDat = pd.DataFrame()
    lat, long, zmean, zstdev,xs,ys = [],[],[],[],[],[]
    for i in chartDat.keys():
        x = chartDat[i][:][0]
        y = chartDat[i][:][1]
        values = chartDat[i][:][2:]
        mean = np.mean(values)
        stdev = np.std(values)
        lt,lg = utm.to_latlon(x,y,30,"u")
        lat.append(lt)
        long.append(lg)
        zmean.append(mean)
        zstdev.append(stdev)
        xs.append(x)
        ys.append(y)
        #print(mean, stdev)
    gridDat['lat'] = lat
    gridDat['long'] = long
    gridDat['z'] = zmean
    gridDat['stdev'] = zstdev
    gridDat['x'] = xs
    gridDat['y'] = ys
    gridDat.to_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\stats\\plots_part_'+str(samNum)+'plot_'+str(gridSize)+'m_grid.csv')
import winsound
duration = 4000  # millisecond
freq = 600  # Hz
winsound.Beep(freq, duration)


