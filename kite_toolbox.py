# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:16:13 2021

@author: jparent1
"""
import tkinter as tk
from tkinter import filedialog
from scipy import interpolate
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal

from ipywidgets import widgets
from traitlets import traitlets

class LoadedButton(widgets.Button):
    """A button that can holds a value as a attribute."""
    def __init__(self, value=None, *args, **kwargs):
        super(LoadedButton, self).__init__(*args, **kwargs)
        # Create the value attribute.
        self.add_traits(value=traitlets.Any(value))

def get_directory(button) :
    root = tk.Tk()  #create tk root object
    root.wm_attributes('-topmost', 1) #place root on top
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    #test if directory exist, use to define initialdir
    #in spyder go to run / configuration per file enable fonction : Run in console's namespace instead of an empty one
    try: button.value
    except: button.value = None
    #open filedialog, with initialfile set with previous open file
    #filename = filedialog.askopenfilename(initialfile=filename, parent=root)
    button.value = filedialog.askdirectory()
    #destroy tk window
    root.destroy() 

#interpolate for ressampling, median filter is also applied
def interploateNan(x,y,kind=None,axis=None,fill_value=None) :
    #get index where y is not a Nan and apply a logical or between column is one column is nan mak the line
    mask = np.logical_or.reduce(np.isnan(y), axis=1)
    #interpoloate data where data are not nan
    return interpolate.interp1d(x[~mask],y[~mask,:],kind=kind,axis=axis,fill_value=fill_value)

def compute_board(folder) :
    #load board data
    data_board = data = pd.read_csv(folder+'/board.csv',sep=None, engine='python')
    #Remove offset in timestamps
    data_board["timestamp[us]"] -= data_board["timestamp[us]"].min()
    #Get numpy array from panda
    timestamps = data_board["timestamp[us]"].to_numpy()
    forces = data_board[["F1[N]","F2[N]","F3[N]","F4[N]","F5[N]","F6[N]"]].to_numpy()
    #remove outliners avec a median filter of size 3 and smooth data
    forces = signal.medfilt(forces,(3,1))
    acc = data_board[["accX[m/s2]","accy[m/s2]","accZ[m/s2]"]].to_numpy()
    gyro = data_board[["gyroX[degree/s]","gyroY[degree/s]","gyroZ[degree/s]"]].to_numpy()*np.pi/180
    mag = data_board[["magX[uT]","magY[uT]","magZ[uT]"]].to_numpy()
    #interpolate data
    method = "linear"
    fF = ktb.interploateNan(timestamps,forces,kind=method,axis=0,fill_value="extrapolate")
    facc = ktb.interploateNan(timestamps,acc,kind=method,axis=0,fill_value="extrapolate")
    fgyro = ktb.interploateNan(timestamps,gyro,kind=method,axis=0,fill_value="extrapolate")
    fmag = ktb.interploateNan(timestamps,mag,kind=method,axis=0,fill_value="extrapolate")
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

    #remove outliners avec a median filter of size 3 and smooth data
    accRsp = signal.medfilt(accRsp,(5,1))
    gyroRsp = signal.medfilt(gyroRsp,(5,1))
    magRsp = signal.medfilt(magRsp,(5,1))

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
