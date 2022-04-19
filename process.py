
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 09:40:40 2021

@author: jparent1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog

from scipy import interpolate
from scipy import signal

#interpolate for ressampling, median filter is also applied
def interploateNan(x,y,kind=None,axis=None,fill_value=None) :
    #get index where y is not a Nan and apply a logical or between column is one column is nan mak the line
    mask = np.logical_or.reduce(np.isnan(y), axis=1)
    #interpoloate data where data are not nan
    return interpolate.interp1d(x[~mask],y[~mask,:],kind=kind,axis=axis,fill_value=fill_value)

#IGNORE MAGNETIC FIELD
ignore_mag = False

#OPEN FILE DIALOG
root = tk.Tk()
#test if variable exist 
try: filename
except: filename = None
#open filedialog, with initialfile set with previous open file
#in spyder go to run / configuration per file enable fonction : Run in console's namespace instead of an empty one
filename = filedialog.askopenfilename(initialfile=filename)
#destroy tk window
root.destroy() 

#open csv with panda reader
data = pd.read_csv(filename,sep=None, engine='python')
print("Data loaded")
#Remove offset in timestamps
data["timestamp[us]"] -= data["timestamp[us]"].min()

#Get numpy array from panda
timestamps = data["timestamp[us]"].to_numpy()
forces = data[["F1[N]","F2[N]","F3[N]","F4[N]","F5[N]","F6[N]"]].to_numpy()
#remove outliners avec a median filter of size 3 and smooth data
forces = signal.medfilt(forces,(3,1))
acc = data[["accX[m/s2]","accy[m/s2]","accZ[m/s2]"]].to_numpy()
gyro = data[["gyroX[degree/s]","gyroY[degree/s]","gyroZ[degree/s]"]].to_numpy()*np.pi/180
mag = data[["magX[uT]","magY[uT]","magZ[uT]"]].to_numpy()

#interpolate data
method = "linear"
fF = interploateNan(timestamps,forces,kind=method,axis=0,fill_value="extrapolate")
facc = interploateNan(timestamps,acc,kind=method,axis=0,fill_value="extrapolate")
fgyro = interploateNan(timestamps,gyro,kind=method,axis=0,fill_value="extrapolate")
fmag = interploateNan(timestamps,mag,kind=method,axis=0,fill_value="extrapolate")

#Resample data @ 50hz
frequency = 50; #resampling frequency
#new time vector
time = np.arange(0,data["timestamp[us]"].max(),1e6/frequency)
#resample data
forcesRsp = fF(time)
accRsp = facc(time)
gyroRsp = fgyro(time)
magRsp = fmag(time)

time /= 1e6
print('durée en s',time[-1]-time[0])

#remove outliners avec a median filter of size 3 and smooth data
accRsp = signal.medfilt(accRsp,(5,1))
gyroRsp = signal.medfilt(gyroRsp,(5,1))
magRsp = signal.medfilt(magRsp,(5,1))

#plot raw data vs resample
plt.figure('data')
plt.subplot(4,1,1)
plt.title("Forces Z")
plt.plot(data["timestamp[us]"],forces,'-')
plt.plot(time,forcesRsp,',')
plt.ylim(np.floor(forcesRsp.min()),np.ceil(forcesRsp.max()))

plt.subplot(4,1,2)
plt.title("Acceleration")
plt.plot(data["timestamp[us]"],acc,'-')
plt.plot(time,accRsp,'.')

plt.subplot(4,1,3)
plt.title("Gyroscope")
plt.plot(data["timestamp[us]"],gyro,'-')
plt.plot(time,gyroRsp,'.')

plt.subplot(4,1,4)
plt.title("Magnetic field")
plt.plot(data["timestamp[us]"],mag,'*')
plt.plot(time,magRsp,'.')

#compute force
F = forcesRsp
#sum front and back force
F_front = np.sum(F[:,0:3],axis=1)
F_back = np.sum(F[:,3:6],axis=1)
#compute moment
g1 = 151.5  
g2 = 121.2  
e1 = 276.5  
e3 = 294    
e2 = 539    
e4 = 464
M_tot_X = (g1*F[:,1]+g2*F[:,5]-g1*F[:,2]-g2*F[:,4]) / 1000
M_tot_Y = (e1*np.sum(F[:,1:3],axis=1)+e2*F[:,0]-e3*np.sum(F[:,4:6],axis=1)-e4*F[:,3]) / 1000

#Compute front moment / back moment
h1 = 100
h2 = 140
j1 = 175
alpha = 60 * np.pi / 180
threshold = 100

M_front_X = (np.sin(alpha)* j1 * F[:,1] - np.sin(alpha) * j1 * F[:,2]) / 1000
M_front_Y = (j1*F[:,0] - np.cos(alpha)*j1*np.sum(F[:,1:3],axis=1)) / 1000
#define a mask to compute reaction point only when force is applied on board
mask = np.abs(F_front>threshold)
P_front_X = M_front_Y[mask] * 1000 / F_front[mask]
P_front_Y = -M_front_X[mask] * 1000 / F_front[mask]

M_back_X = (np.sin(alpha)*h2 * F[:,5] - np.sin(alpha)*h2 * F[:,4]) / 1000
M_back_Y = (np.cos(alpha)*h2*np.sum(F[:,4:6],axis=1) - h1*F[:,3]) / 1000
#define a mask to compute reaction point only when force is applied on board
mask = np.abs(F_back>threshold)
P_back_X = M_back_Y[mask] * 1000 / F_back[mask]
P_back_Y = -M_back_X[mask] * 1000 / F_back[mask]

P_front_X_sensor = [0, j1 , -j1*np.cos(alpha), -j1*np.cos(alpha)] 
P_front_Y_sensor = [0, 0 , -j1*np.sin(alpha), j1*np.sin(alpha)] 

P_back_X_sensor = [0, -h1 , h2*np.cos(alpha), h2*np.cos(alpha)] 
P_back_Y_sensor = [0, 0 , h2*np.sin(alpha), -h2*np.sin(alpha)] 

plt.figure("Forces")
plt.subplot(4,1,1)
plt.title("Forces")
plt.plot(time,F_front)
plt.plot(time,F_back)
# plt.plot(time,np.ones(time.shape)*11.5)
plt.legend(("front","back"))

plt.subplot(4,1,2)
plt.title("Moment total")
plt.plot(time,M_tot_X)
plt.plot(time,M_tot_Y)
plt.legend(("Mx","My"))

plt.subplot(4,1,3)
plt.title("Moment front")
plt.plot(time,M_front_X)
plt.plot(time,M_front_Y)
plt.legend(("Mx","My"))

plt.subplot(4,1,4)
plt.title("Moment back")
plt.plot(time,M_back_X)
plt.plot(time,M_back_Y)
plt.legend(("Mx","My"))


plt.figure("Position")
plt.subplot(1,2,2)
plt.title("position front")
plt.scatter(P_front_X,P_front_Y)
plt.scatter(P_front_X_sensor,P_front_Y_sensor,marker='x',color='red')
plt.axis('equal')
plt.xlim(-200,200)
plt.ylim(-200,200)

plt.subplot(1,2,1)
plt.title("position back")
plt.scatter(P_back_X,P_back_Y)
plt.scatter(P_back_X_sensor,P_back_Y_sensor,marker='x',color='red')
plt.axis('equal')
plt.xlim(-200,200)
plt.ylim(-200,200)

#Attitude estimator : https://ahrs.readthedocs.io/en/latest/filters.html
if ignore_mag : 
    magRsp = None
    gain = 0.033
else :
    gain = 0.041

# from ahrs.filters import EKF
# result = EKF(gyr=gyroRsp, acc=accRsp, mag=magRsp,frequency=frequency)

from ahrs.filters import Madgwick
madgwick = Madgwick(gyr=gyroRsp, acc=accRsp, mag = magRsp, frequency=frequency, gain = gain)

# from ahrs.filters import Mahony
# result = Mahony(gyr=gyro80hz, acc=acc80hz, frequency=80.0)

# from ahrs.filters.aqua import AQUA
# result = AQUA(gyr=gyro80hz, acc=acc80hz, frequency=80.0)

#Good performance
from ahrs.filters  import Complementary
complementary = Complementary(gyr=gyroRsp, acc=accRsp, mag=magRsp, frequency=frequency)

#convert quaternion to Euler angle more specifically Tait-Bryan angles
from scipy.spatial.transform import Rotation as R
r_comp = R.from_quat(np.roll(complementary.Q,-1))
r_madg = R.from_quat(np.roll(madgwick.Q,-1))
#Tait-Bryan angles
angles_comp = r_comp.as_euler('zxy', degrees=False)
angles_madg = r_madg.as_euler('zxy', degrees=False)
#unwrap data if necessary
# angles_comp  =np.unwrap(angles_comp,axis=0)
# angles_madg  =np.unwrap(angles_madg,axis=0)

angles_comp[:,0] = np.mod((angles_comp[:,0] + np.pi+np.pi/2 ),2*np.pi) - np.pi
angles_comp *= 180/np.pi

angles_madg *= 180/np.pi

plt.figure("Tait-Bryan angles")
plt.subplot(3,1,1)
plt.title("Roulis")
#plt.plot(angles_comp[:,0])
plt.plot(angles_madg[:,0])
plt.legend(('complementary','madgwick'))

plt.subplot(3,1,2)
plt.title("Tangage")
#plt.plot(angles_comp[:,1])
plt.plot(angles_madg[:,1])

plt.subplot(3,1,3)
plt.title("Lacet")
#plt.plot(angles_comp[:,2])
plt.plot(angles_madg[:,2])

plt.show()