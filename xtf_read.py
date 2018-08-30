# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 12:45:03 2018

@author: jackm

First Script to run
- This reads the XTF files that are created from the MB equipment
- Parses the Binary 
- Converts rx angle, time, sound velocity into xyz of seafloor
- creates csv's of the xyz  for each ping
"""
#get_ipython().magic('reset -sf') 

import pyxtf
import datetime, numpy as np
import matplotlib.pyplot as plt
from pyxtf import xtf_read, concatenate_channel, XTFHeaderType
from bitstring import *
import pandas as pd
from latlon_wgs84 import metres_to_latlon
import utm
import calendar
import time

def toTimestamp(d):
  return calendar.timegm(d.timetuple())

d = pd.read_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\recieved\\from_alker\\ACII_20171012_raw_WGS84_attitude.txt',parse_dates=[0])
#f, axarr = plt.subplots(3, sharex=True)

tim, rolls = x2, spl(x2) 

"""
things = [] 
for i in np.arange(0,len(d)):
   thing = d.iloc[i,0]#.timetuple()
   thing = thing.replace(year = 2017, month=10,day=12).timetuple()
   thing = calendar.timegm(thing)
   things.append(thing)
"""   

lt = []
lg = []
bts = []
bnums = []
jeffs = pd.read_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\random\\beam_angs.csv')
jeffs['angs'] = np.linspace(-1.3090005,1.3090005,512)
count = 1
num = 1 
for j in np.arange(1,3):
    input_file = 'C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\mission\\0002 - Line_ASV_01 - '+str(j)+'.xtf'    
    (fh, p) = pyxtf.xtf_read(input_file)
    print('The following (supported) packets are present (XTFHeaderType:count): \n\t' +
          str([key.name + ':{}'.format(len(v)) for key, v in p.items()]))
    n_channels = fh.channel_count(verbose=True)
    actual_chan_info = [fh.ChanInfo[i] for i in range(0, n_channels)]
    print('Number of data channels: {}\n'.format(n_channels))
    print("LINE NUMBER:", j)
    sonar_ch = p[pyxtf.XTFHeaderType.reson_7125]
    jeff = p[pyxtf.XTFHeaderType.pos_raw_navigation]
    dfPoints = pd.DataFrame(columns = ('lat','long','z'))
    nav = pd.DataFrame(columns = ('lat','long'))   
    
    for i in np.arange(0,len(sonar_ch),5): #np.arange(0,len(sonar_ch)):
        zp = []
        latp = []
        longp = [] 
        alt = jeff[i].RawAltitude
        #print(alt)
        sonar_ch_ping = sonar_ch[i]
        s = BitString(sonar_ch_ping.data)
        
        hour = sonar_ch[i].Hour
        minute = sonar_ch[i].Minute
        second = sonar_ch[i].Second
        hsecs = sonar_ch[i].HSeconds
        
        then = datetime.datetime(2017,10,12,hour,minute,second,hsecs*10000) 
        micTime = (calendar.timegm(then.timetuple())*1e3 + then.microsecond/1e3)/1000
                
        #roll = np.interp(toTimestamp(datetime.datetime(2017,10,12,hour,minute,second,hsecs*10000)),tim,rolls)
        #pitch = np.interp(toTimestamp(datetime.datetime(2017,10,12,hour,minute,second,hsecs*10000)),tim,rolls)
        
        roll = np.interp(micTime,tim,rolls)
        pitch = np.interp(micTime,tim,rolls)
        
        #print(roll) 
        #### DRF ####
        protVers = s.read(16).uintle
        offset = s.read(16).uintle
        syncPat = s.read(32).uintle
        size = s.read(32).uintle*8
        offset = s.read(32).uintle
        ident = s.read(32).uintle
        skTime = s.read(80).uintle     #TIME
        #print(skTime)
        recVers = s.read(16).uintle
        recId = s.read(32).uintle
        s.read(32).uintle
        s.read(16).uintle
        s.read(16).uintle
        s.read(32).uintle
        s.read(16).uintle
        s.read(16).uintle
        s.read(32).uintle
        s.read(32).uintle
        s.read(32).uintle
        #print("rec Id:\t\t",recId)
        ###  7027 rec header ####
        sonID = s.read(64).uintle
        pingNo = s.read(32).uintle
        #print("ping no:\t",pingNo)
        MPS =  s.read(16).uintle
        N = s.read(32).uintle                   #number of reciever beams
        #print("number of beams:\t",N)
        DFS = s.read(32).uintle                 #data field size in bytes
        detectionAlgo = s.read(8).uintle        #detection algo
        flags = s.read(32).uintle
        samRate = s.read(32).floatle            #sample rate
        txAng = s.read(32).floatle
        appRoll = s.read(32).floatle
        s.read(32).uintle
        s.read(32).uintle
        s.read(32).uintle
        s.read(32).uintle
        s.read(32).uintle
        s.read(32).uintle
        s.read(32).uintle
        s.read(32).uintle
        s.read(32).uintle
        s.read(32).uintle
        s.read(32).uintle
        s.read(32).uintle
        s.read(32).uintle
        s.read(32).uintle
        s.read(32).uintle
        lat = sonar_ch_ping.ShipYcoordinate
        long = sonar_ch_ping.ShipXcoordinate    
        lt.append(lat)
        lg.append(long)

        ### 7027 RD ###
        bCount = 1
        while bCount <= N:        
            beamNum = s.read(16).uintle
            #print("beam no:",beamNum)
            detP = s.read(32).floatle
            twt = detP/samRate
            rang = twt*sonar_ch[i].SoundVelocity        
            #print("range:\t",rang)  
            #print("detP:\t",detP)
            
            beta =   s.read(32).floatle         #beam steering angle in radians]       
            #beta = jeffs.iloc[bCount-1,3]
            if bCount == 1:
                #print(rxAng, rang)
                """
            if bCount == 512/2:
                print(rxAng,rang)
            if bCount == 512:
                print(rxAng,rang)"""
            #if bCount == 512:
                #print("ang:\t",rxAng)
            #print("recieve angle:\t",rxAng)
            flgs =  s.read(32).uintle           #flags
            qualDet =  s.read(32).uintle        # quality of detection see table 70
            unc =  s.read(32).floatle           #detection uncertainty
            sigStr =  s.read(32).floatle        #signal strength
            alpha = sonar_ch_ping.ShipGyro

            #bDist = abs((z**2)-(rang**2))**0.5  #abs(np.cos(phi)*rang)          #horizontal distance from centroid to detection point     
            sin = np.sin
            cos = np.cos
            pi = np.pi
            tan = np.tan            
            R = rang
            """
            if (roll <= 0.001) and (roll >= -0.001) and (beamNum == 1):
                print(roll, beta, micTime)
                p#rint(datetime.datetime(2017,10,12,hour,minute,second,hsecs*10000))
            """
            """
            roll and pitch lever arm offsets   
            alpha = heading
            beta = beam angle =rxAng
            tT = pitch
            tR = roll                        
            """
            alpha = alpha*pi/180
            tR = roll*pi/180                                                ## these will become the roll 
            tT =  pitch*pi/180                                                 ## and heave values (from the pospac)
            tR = 0
            #tT = 0
            x = ( cos(alpha)*cos(tR)+sin(alpha)*sin(tT)*sin(tR)) *R*sin(beta) +(-cos(alpha)*sin(tR)+sin(alpha)*sin(tT)*cos(tR))*R*cos(beta)
            y = (-sin(alpha)*cos(tR)+cos(alpha)*sin(tT)*sin(tR)) *R*sin(beta) + (sin(alpha)*sin(tR)+cos(alpha)*sin(tT)*cos(tR))*R*cos(beta)
            z = cos(tT)*sin(tR)*R*sin(beta)+cos(tT)*cos(tR)*R*cos(beta)    
            jeff = p[pyxtf.XTFHeaderType.pos_raw_navigation]
            z = jeff[i].RawAltitude + z
            
            alpha = alpha*180/pi
                
            """
            if beamNum > 256:
                x = np.sin((alpha+90)*np.pi/180)*bDist
                y = np.cos((alpha+90)*np.pi/180)*bDist
                
            elif beamNum <= 256:
                x = np.sin((alpha-90)*np.pi/180)*bDist
                y = np.cos((alpha-90)*np.pi/180)*bDist             
            """
            
            latNew,longNew= metres_to_latlon(lat, long, x, y)
            latp.append(latNew)
            longp.append(longNew)
            zp.append(z)     
            bCount += 1
            
            """
            if (roll < 0.001) and (roll > -0.001):
               print("bs",beta,"roll: \t",roll, "ping:", count)
               bts.append(beta)
               bnums.append(beamNum)
               
               num += 1
            """
                  
        """
        if beamNum == 511:
            print("511: z:",z, "x:", x, "y:2", y)
        """
           
        dfPoints['lat'] = latp
        dfPoints['long'] = longp
        dfPoints['z'] = zp
        
        dfPoints.to_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\xyz_files_small_ATTITUDE\\ping'+str(count)+'.csv') #, append= True    
        count += 1

        #print(count)
        """
        if count >= 368:
            break
    if count >= 368:
        break
        """
#jeffs['bNum'] = bnums
#jeffs['angle'] = bts
#jeffs.to_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\random\\beam_angs.csv')   

"""        
nav['lat'] = lt
nav['long'] = lg
nav.to_csv('C:\\Users\\jackm\\Google Drive\\Uni\\Masters\\Semester 2\\Research Project\\XTF_PLOT\\random\\navTrack_test_1.csv')  
"""
import winsound
duration = 1000  # millisecond
freq = 440  # Hz
winsound.Beep(freq, duration)



"""
# Get multibeam/bathy data (xyza) if present
if XTFHeaderType.bathy_xyza in p:
    np_mb = [[y.fDepth for y in x.data] for x in p[XTFHeaderType.bathy_xyza]]

    # Allocate room (with padding in case of varying sizes)
    mb_concat = np.full((len(np_mb), max([len(x) for x in np_mb])), dtype=np.float32, fill_value=np.nan)
    for i, line in enumerate(np_mb):
        mb_concat[i, :len(line)] = line

    # Transpose if the longest axis is vertical
    is_horizontal = mb_concat.shape[0] < mb_concat.shape[1]
    mb_concat = mb_concat if is_horizontal else mb_concat.T
    plt.figure()
    plt.imshow(mb_concat, cmap='hot')
    plt.colorbar(orientation='horizontal')
"""


"""
# Get sonar if present
if XTFHeaderType.reson_7125 in p:
    upper_limit = 2 ** 16
    np_chan1 = concatenate_channel(p[XTFHeaderType.reson_7125], file_header=fh, channel=0, weighted=True)
    np_chan2 = concatenate_channel(p[XTFHeaderType.reson_7125], file_header=fh, channel=1, weighted=True)

    # Clip to range (max cannot be used due to outliers)
    # More robust methods are possible (through histograms / statistical outlier removal)
    np_chan1.clip(0, upper_limit - 1, out=np_chan1)
    np_chan2.clip(0, upper_limit - 1, out=np_chan2)

    # The sonar data is logarithmic (dB), add small value to avoid log10(0)
    np_chan1 = np.log10(np_chan1 + 1, dtype=np.float32)
    np_chan2 = np.log10(np_chan2 + 1, dtype=np.float32)

    # Transpose so that the largest axis is horizontal
    np_chan1 = np_chan1 if np_chan1.shape[0] < np_chan1.shape[1] else np_chan1.T
    np_chan2 = np_chan2 if np_chan2.shape[0] < np_chan2.shape[1] else np_chan2.T

    # The following plots the waterfall-view in separate subplots
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(np_chan1, cmap='gray', vmin=0, vmax=np.log10(upper_limit))
    ax2.imshow(np_chan2, cmap='gray', vmin=0, vmax=np.log10(upper_limit))
    fig.tight_layout()

    # The following plots a waterfall-view of the 100th ping (in the file)
    # fig, (ax1, ax2) = plt.subplots(2,1)
    # ax1.plot(np.arange(0, np_chan1.shape[1]), np_chan1[196, :])
    # ax2.plot(np.arange(0, np_chan2.shape[1]), np_chan2[196, :])
"""

"""
if XTFHeaderType.custom_vendor_data in p:
    pings = p[XTFHeaderType.custom_vendor_data]  # type: List[XTFAttitudeData]
    heave = [ping.Heave for ping in pings]
    pitch = [ping.Pitch for ping in pings]
    roll = [ping.Roll for ping in pings]
    heading = [ping.Heading for ping in pings]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(range(0, len(heave)), heave, label='heave')
    ax1.plot(range(0, len(pitch)), pitch, label='pitch')
    ax1.plot(range(0, len(roll)), roll, label='roll')
    ax1.legend()
    ax2.plot(range(0, len(heading)), heading, label='heading')
    ax2.legend()
    fig.tight_layout()
    """

"""
if XTFHeaderType.pos_raw_navigation in p:  #pos_raw_navigation
    pings = p[XTFHeaderType.pos_raw_navigation]  # type: List[XTFHeaderNavigation]
    alt = [ping.RawAltitude for ping in pings]
    x = [ping.RawXcoordinate for ping in pings]
    y = [ping.RawYcoordinate for ping in pings]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(range(0, len(alt)), alt, label='altitude')
    ax1.set_title('altitude')
    ax2.plot(range(0, len(x)), x, label='x')
    ax2.set_title('x')
    ax3.plot(range(0, len(y)), y, label='y')
    ax3.set_title('y')
    fig.tight_layout()

plt.show()
"""