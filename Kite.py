import tkinter as tk
from tkinter import filedialog

from os import path

import pandas as pd
import numpy as np

from scipy import interpolate
from scipy import signal

#from bqplot import pyplot as plt
#import bqplot as bq
import plotly.graph_objects as go
import chart_studio.plotly as py
from IPython.display import display

#plt.rcParams['figure.figsize'] = [8, 8]

#dimensional parameters in mm
g1 = 151.5  
g2 = 121.2  
e1 = 276.5  
e3 = 294    
e2 = 539    
e4 = 464
h1 = 100
h2 = 140
j1 = 175
alpha = 60 * np.pi / 180 #in degree
threshold = 100

#interpolate for ressampling, median filter is also applied
def interploateNan(x,y,kind=None,axis=None,fill_value=None) :
    #get index where y is not a Nan and apply a logical or between column is one column is nan mak the line
    mask = np.logical_or.reduce(np.isnan(y), axis=1)
    #interpoloate data where data are not nan
    return interpolate.interp1d(x[~mask],y[~mask,:],kind=kind,axis=axis,fill_value=fill_value)

class Kite :
    sampling = 50 #re-sampling-frequnecy in Hz
    folder = None
    board_data = None
    gps_data = None
    harness_data = None
    range = None

    def __init__(self,out) :
        #out widget to display info
        self.debug_view = out
        pass

    def set_folder(self,_) :
        with self.debug_view :
            root = tk.Tk()  #create tk root object
            root.wm_attributes('-topmost', 1) #place root on top
            root.withdraw() # we don't want a full GUI, so keep the root window from appearing
            self.folder = filedialog.askdirectory() 
            print('selected folder :',self.folder)
            #destroy tk window
            root.destroy()
            #define path to csv file
            board_path = self.folder+'/board.csv'
            gps_path = self.folder+'/gps.csv'
            harness_path = self.folder+'/harness.csv'
            #test if file exist 
            if path.isfile(board_path) :
                #open csv with panda reader
                self.board_data = pd.read_csv(board_path,sep=None, engine='python')
                print('board data loaded')
            else :
                print('board.csv file missing')
            if path.isfile(gps_path) :
                self.gps_data = pd.read_csv(gps_path,sep=None, engine='python')
                print('gps data loaded')
            else :
                print('gps.csv file missing')
            if path.isfile(harness_path) :
                self.harness_data = pd.read_csv(harness_path,sep=None, engine='python')
                print('harness data loaded')
            else :
                print('harness.csv file missing')
        
        self.analyze()
    
    def analyze(self) :
        with self.debug_view :
            #Remove offset in timestamps
            self.board_data["timestamp[us]"] -= self.board_data["timestamp[us]"].min()
            self.gps_data['time'] -= self.gps_data['time'].min()

            #Get numpy array from panda
            timestamps_ms = self.board_data["timestamp[us]"].to_numpy() / 1e3 #timestamps in ms
            forces = self.board_data[["F1[N]","F2[N]","F3[N]","F4[N]","F5[N]","F6[N]"]].to_numpy()
            acc = self.board_data[["accX[m/s2]","accy[m/s2]","accZ[m/s2]"]].to_numpy()
            gyro = self.board_data[["gyroX[degree/s]","gyroY[degree/s]","gyroZ[degree/s]"]].to_numpy()*np.pi/180
            mag = self.board_data[["magX[uT]","magY[uT]","magZ[uT]"]].to_numpy()
            #remove outliners with a median filter of size 3 and smooth data
            forces = signal.medfilt(forces,(3,1))
            #get numpy from pandas
            timestamps_gps_ms = self.gps_data['time'].to_numpy()
            location = self.gps_data[['lon','lat']].to_numpy()
            speed = self.gps_data['speed'].to_numpy()
            bearing = self.gps_data['bearing'].to_numpy()

            #convert gps time to ms and round it
            timestamps_gps_ms = np.round(timestamps_gps_ms * 1e3, decimals=0)
            #add offset to gps timestamps
            timestamps_gps_ms -= 2200       #WARNING CUSTOM PARAMETERS


            #interpolate data
            method = "linear"
            fF = interploateNan(timestamps_ms,forces,kind=method,axis=0,fill_value="extrapolate")
            facc = interploateNan(timestamps_ms,acc,kind=method,axis=0,fill_value="extrapolate")
            fgyro = interploateNan(timestamps_ms,gyro,kind=method,axis=0,fill_value="extrapolate")
            fmag = interploateNan(timestamps_ms,mag,kind=method,axis=0,fill_value="extrapolate")

            flocation = interpolate.interp1d(timestamps_gps_ms,location,kind=method,axis=0,fill_value="extrapolate")
            fspeed = interpolate.interp1d(timestamps_gps_ms,speed,kind=method,axis=0,fill_value="extrapolate")
            fbearing = interpolate.interp1d(timestamps_gps_ms,bearing,kind=method,axis=0,fill_value="extrapolate")

            #Resample data at defined frequency
            #new time vector
            self.timestamps_ms = np.arange(0,timestamps_ms.max(),1e3/self.sampling)
            #resample data
            self.board_forces = fF(timestamps_ms)
            self.board_acc = facc(timestamps_ms)
            self.board_gyro = fgyro(timestamps_ms)
            self.board_mag = fmag(timestamps_ms)
            self.gps_location = flocation(timestamps_ms)
            self.gps_speed = fspeed(timestamps_ms)
            self.gps_bearing = fbearing(timestamps_ms)

    def define_range(self,start,end) :
        self.range = slice(start,end)
        out_range = np.ma.array(self.gps_location, mask=False)
        out_range.mask[start:end,:] = True
        in_range = self.gps_location[start:end,:]
        #fig = px.scatter(x=in_range[:,0], y=in_range[:,1])
        fig = go.Figure(data=go.Scatter(x=in_range[:,0], y=in_range[:,1], mode='markers'))
        py.iplot(fig, filename='jupyter-basic_bar')
        #plt.figure()
        #plt.scatter(out_range[:,0],out_range[:,1])
        #plt.scatter(in_range[:,0],in_range[:,1],color=['red'])
        #plt.show()
        #ax_x = bq.Axis(label='Test X', scale=bq.LinearScale(), tick_format='0.0f')
        #ax_y = bq.Axis(label='Test Y', scale=bq.LinearScale(), orientation='vertical', tick_format='0.2f')
        #scatt_in = bq.Scatter(x=in_range[:,0],y=in_range[:,1],scales={'x': bq.LinearScale(), 'y': bq.LinearScale()},default_size=1)
        #scatt_out = bq.Scatter(x=out_range[:,0],y=out_range[:,1],scales={'x': bq.LinearScale(), 'y': bq.LinearScale()},default_size=1)
        #fig = bq.Figure(axes=[ax_x, ax_y], marks=[scatt_in,scatt_out])
        #display(fig)
        self.compute_forces()

    def add_plot(self,out) :
        self.plot_area = out

    def compute_forces(self) :
        if self.range : 
            F = self.board_forces[self.range,:]
            self.F_front = np.sum(F[:,0:3],axis=1)
            self.F_back = np.sum(F[:,3:6],axis=1)
            self.M_tot_X = (g1*F[:,1]+g2*F[:,5]-g1*F[:,2]-g2*F[:,4]) / 1000
            self.M_tot_Y = (e1*np.sum(F[:,1:3],axis=1)+e2*F[:,0]-e3*np.sum(F[:,4:6],axis=1)-e4*F[:,3]) / 1000

    def display_plots(self,button) :
        self.plot_area.clear_output(True)
        time = self.timestamps_ms[self.range] / 1e3
        with self.plot_area :
            plt.figure("GPS data")
            plt.subplot(2,1,1)
            plt.title("Speed [m/s]")
            plt.plot(time,self.gps_speed[self.range])
            plt.subplot(2,1,2)
            plt.title("bearing")
            plt.plot(time,self.gps_bearing[self.range])

            plt.figure("Forces")
            plt.subplot(4,1,1)
            plt.title("Forces")
            plt.plot(time,self.F_front)
            plt.plot(time,self.F_back)
            # plt.plot(time,np.ones(time.shape)*11.5)
            plt.legend(("front","back"))
            
            plt.subplot(4,1,2)
            plt.title("Moment total")
            plt.plot(time,self.M_tot_X)
            plt.plot(time,self.M_tot_Y)
            plt.legend(("Mx","My"))

            plt.show()