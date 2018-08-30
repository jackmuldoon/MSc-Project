# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:47:32 2018

@author: jackm
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.interpolate import UnivariateSpline

df = pd.read_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\recieved\\from_alker\\ACII_20171012_raw_WGS84_attitude.txt',parse_dates=[0])
#d.iloc[:,0]d.iloc[:,0].dt.time

def toTimestamp(d):
  return calendar.timegm(d.timetuple())

things = [] 
for i in np.arange(0,len(df)):
   thing = df.iloc[i,0]#.timetuple()
   thing = thing.replace(year = 2017, month=10,day=12).timetuple()
   thing = calendar.timegm(thing)
   things.append(thing)
df['tstamp'] = things


#df = df.iloc[:80,:]

x = df.tstamp
y = df.iloc[:,1]
p = lagrange(x,y)
spl = UnivariateSpline(x,y)
x2 = np.arange(min(x),max(x),0.2)

"""
spl.set_smoothing_factor(0.0)
plt.plot(x, y, "-", label="Function")
plt.plot(x2, spl(x2), label="Polynom")
"""


#f, axarr = plt.subplots(1, sharex=True)

#.plot(x2, np.polyval(p, x2))
#axarr.set_title('Sharing X axis')
#axarr[1].plot(df.index, df.iloc[:,2])
#axarr[2].plot(df.index, df.iloc[:,3])
#axarr[0].set_xlim(500,505)

