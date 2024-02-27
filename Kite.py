
from math import degrees
import struct
from pathlib import Path

from datetime import datetime
from tkinter import Wm
from urllib.parse import DefragResultBytes
#from tkinter import Button

import numpy as np
import pandas as pd
from pandas import DataFrame, Timedelta

#import xlsxwriter as xlsx

from scipy import interpolate
from scipy import signal
from scipy.spatial.transform import Rotation as R

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Button

#sampling rate
#sampling_hz = 80 

#dimensional parameters in mm
d= 728
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
threshold = 200     #threshold in N to find pressure point

#data format to decode binary files
data_struct = struct.Struct('I6f9f7fh9B?3c21x')

#interpolate for ressampling, median filter is also applied
def interploateNan(x,y,kind=None,axis=None,fill_value=None) :
    #get index where y is not a Nan and apply a logical or between column is one column is nan mak the line
    if len(y.shape) > 1 : 
        mask = np.logical_or.reduce(np.isnan(y), axis=1)
        out = interpolate.interp1d(x[~mask],y[~mask,:],kind=kind,axis=axis,fill_value=fill_value)
    else :
        mask = (np.isnan(y))
        out = interpolate.interp1d(x[~mask],y[~mask],kind=kind,axis=axis,fill_value=fill_value)
    
    #interpoloate data where data are not nan
    return out

class Kite :
    #bin files list 
    files = []
    folder = wind_file = protocol_file = None 
    #selected range for export mainly
    range = None
    #pandas data_frame
    raw_datas = raw_wind_datas = None
    datas = pd.DataFrame()
    
    #set experiement folder
    def set_folder(self,path,sampling_hz=80) :
        folder = Path(path)
        count = 0
        #find bin files
        for file in folder.iterdir() :
            if file.suffix == ".bin" :
                print("Data file : "+str(file))
                count += 1
        self.files = []
        for i in range(count) :
            self.files.append(folder/"log_{:04d}.bin".format(i))
        #find wind and protocol file in parent folder
        wind_file = folder.parent/'wind.xlsx'
        self.wind_data = False
        #wind_file = Path('Station2 - 2022-5-10 - 2022-5-13 23_59_59.csv')
        protocol_file = folder.parent/'protocol.txt'
        if wind_file.exists() :
            self.wind_file = wind_file
            print("Wind file : "+str(wind_file))
        else :
            print("No Wind file in parent directory")
        if protocol_file.exists() :
            self.protocol_file = protocol_file
            print("Protocol file : "+str(protocol_file))
        else :
            print("Protocol file NOT FOUND ")
        self.folder = folder

        #check if sampling rate is a string
        if isinstance(sampling_hz,str) :
            #get number before space
            sampling_hz = int(sampling_hz.split(' ')[0])
            print("Sampling rate : ",sampling_hz)
        self.sampling_hz = sampling_hz

    def read_bin_files_to_pandas(self) : 
        #data_struct = struct.Struct('I6f9f4B?fcfc2f3Bfc2B2f12b')
        #names   = 't','F1','F2','F3','F4','F5','F6','Ax','Ay','Az','Gx','Gy','Gz','Mx','My','Mz','hour','min','sec','msec','fix','lat','lat_NS','lon','lon_EW','speed','angle','day','month','year','mag_var','mag_var_EW','fix_quality','satellites','pecision','alitude','empty'
        #offsets =  0 ,  4 ,  8 , 12 , 16 , 20 , 24 , 28 , 32 , 36 , 40 , 44 , 48 , 52 , 56 , 60 ,   64 ,  65 ,  66 ,   67 ,  68 ,  72 ,     76 ,  80 ,     84 ,    88 ,    92 ,  96 ,    97 ,   98 ,      100 ,        104 ,         105 ,        106 ,      108 ,     112 ,   116
        #formats ='u4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4',   'B',  'B',  'B',   'B',   '?', 'f4',    'U1', 'f4',    'U1',   'f4',   'f4',  'B',    'B',   'B',     'f4',        'U1',          'B',         'B',      'f4',     'f4',  'a12'
        
        #data_struct = struct.Struct('I6f9f7fh9B?3c21x')
        names   = 't','F1','F2','F3','F4','F5','F6','Ax','Ay','Az','Gx','Gy','Gz','Mx','My','Mz','lat','lon','speed','angle','mag_var','precision','altitude','msec','year','month','day','hour','min','sec','satellites','battery','fix_quality','fix','lat_NS','lon_EW','mag_EW','empty'
        offsets =  0 ,  4 ,  8 , 12 , 16 , 20 , 24 , 28 , 32 , 36 , 40 , 44 , 48 , 52 , 56 , 60 ,  64 ,  68 ,    72 ,    76 ,      80 ,       84 ,      88 ,   92 ,   94 ,    95 ,  96 ,   97 ,  98 ,  99 ,        100 ,     101 ,         102 ,  103,    104,     105 ,    106 , 107
        formats ='u4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4', 'f4', 'f4',   'f4',   'f4',     'f4',      'f4',     'f4',  'u2',   'B',    'B',  'B',   'B',  'B',  'B',         'B',      'B',          'B',  '?',   'V1',     'V1',    'V1','a21'
        dt = np.dtype({'names': names, 'offsets': offsets, 'formats': formats}, align=True)
        dataframes = []
        for file in self.files :
            with open(str(file),mode='rb') as f :
                temp = DataFrame(np.fromfile(f, dt))
                temp = temp.astype({'lat_NS':'str','lon_EW':'str','mag_EW':'str'}) #type changed to avoid concat error
                #print(temp.dtypes)
                #print(temp.head())
                dataframes.append(temp)

        #concatenate all dataframes into one
        self.raw_datas = pd.concat(dataframes,ignore_index=True)
        #remove offset in time
        self.raw_datas['t'] -= self.raw_datas['t'][0]
        #convert first gps date to epoch
        offset_to_epoch_ns = pd.Timestamp(year=self.raw_datas['year'][0]+2000,month=self.raw_datas['month'][0],day=self.raw_datas['day'][0],hour=self.raw_datas['hour'][0],minute=self.raw_datas['min'][0],second=self.raw_datas['sec'][0],microsecond=self.raw_datas['msec'][0]*1000)#,tz="Europe/Zurich")
        self.offset_to_epoch_us = offset_to_epoch_ns.value / 1000
        #add a timestamp series
        time_index = pd.to_datetime(self.raw_datas['t']+offset_to_epoch_ns.value/1000,unit='us') + pd.DateOffset(hours=2,nanoseconds=offset_to_epoch_ns.value)
        self.raw_datas['Timestamp'] = time_index
        self.raw_datas.set_index('Timestamp',inplace=True)
        #remove max value used as NAN on byte informations
        self.raw_datas = self.raw_datas.replace(to_replace={'msec':65535,'year':255,'month':255,'day':255,'hour':255,'min':255,'sec':255,'satellites':255,'fix_quality':255},value=np.nan)
        #self.filled_data = self.raw_datas.asfreq('12500US',method='pad')
        self.raw_datas.drop(columns=['msec','year','month','day','hour','min','sec','satellites','fix_quality','lat_NS','lon_EW','mag_EW','empty','mag_var','precision','altitude'],inplace=True)
        print(self.raw_datas.head())
        print(self.raw_datas.tail())

        #read wind file
        if self.wind_file :
            wind_data = pd.read_excel(self.wind_file)
            #wind_data = pd.read_csv(self.wind_file)
            wind_data.set_index('Date',inplace=True)
            wind_data = wind_data.resample(pd.Timedelta(1e6/sampling_hz, "micros")).interpolate('linear')
            start_date = time_index.min()
            end_date = time_index.max()
            mask = (wind_data.index >= start_date) & (wind_data.index <= end_date)
            wind_data = wind_data.loc[mask]
            #wind_data.set_index(np.arange(len(wind_data)),inplace=True)
            wind_data.reset_index(level='Date',inplace=True)
        else :
            wind_data = pd.DataFrame()

        self.raw_wind_datas = wind_data

        print(wind_data.head())
        print(wind_data.tail())

        #read segment file
        if self.protocol_file :
            txt_file = open(self.protocol_file, "r")
            segments_list = txt_file.read().split(',')
            txt_file.close()
            print("The segments are: ", segments_list)
        else :
            segments_list = [" "]
            print("No segments definesd ", segments_list)
        
        self.segments_list = segments_list
    
    def write_raw_data(self) :
        self.raw_datas.to_excel(self.folder/"raw_data.xlsx",freeze_panes=(1,0))

    def write_datas(self,segment) :
        range = slice(self.range[0],self.range[1])
        #self.filter_datas.loc[range].to_excel(self.folder/"segmented_data.xlsx",freeze_panes=(1,0))
        filename = segment+'.xlsx'
        path = Path(self.folder.parent/filename)
        if path.exists() :
            mode = 'a'
        else :
            mode = 'w'
        try :
            with pd.ExcelWriter(self.folder.parent/filename,mode=mode,engine='auto',datetime_format='dd/mm/yy hh:mm:ss.000') as writer:  
                self.datas.loc[range].to_excel(writer, sheet_name=self.folder.name,freeze_panes=(1,0))
        except :
            print("Excel writer skipped, sheet already exist")
        

    def analyze(self,) :
        #re-init datas dataframe
        self.datas = pd.DataFrame()
        #Get numpy array from panda
        raw_datas = self.raw_datas
        timestamps_us = raw_datas['t']
        forces = raw_datas[['F1','F2','F3','F4','F5','F6']].to_numpy()

        #remove outliners with a median filter of size 3 and smooth data
        forces = signal.medfilt(forces,(5,1))

        #copy acceleration
        acc = raw_datas[['Ax','Ay','Az']].to_numpy()

        #mean gyro values mesured in static position
        gyro = raw_datas[['Gx','Gy','Gz']].to_numpy() - [0.8304288182559345,0.4775566651665757,0.44051712127550385]

        #magnetic sensor calibration using this method : https://www.appelsiini.net/2018/calibrate-magnetometer/
        mag = raw_datas[['Mx','My','Mz']].to_numpy()
        print('mag average', np.nanmean(mag,axis=0))
        #offset_mag = [ 31.63874988, -23.84223919, 21.44385468]
        #offset_mag = [ 15.826226, -11.149342,  34.053825]
        offset_mag = [ 50.24, -30.38, 9.92]
        scale_mag = offset_mag / np.average(offset_mag) 
        mag = (mag - offset_mag) / scale_mag

        #Correct axis orientation and position inside the board
        len = acc.shape[0]
        acc = np.concatenate((acc[:,0].reshape(len,1),acc[:,2].reshape(len,1),acc[:,1].reshape(len,1)),axis=1)*[1,1,1]
        gyro = np.concatenate((gyro[:,0].reshape(len,1),gyro[:,2].reshape(len,1),gyro[:,1].reshape(len,1)),axis=1)*[1,1,1]
        mag = np.concatenate((mag[:,0].reshape(len,1),mag[:,2].reshape(len,1),mag[:,1].reshape(len,1)),axis=1)*[1,-1,1]
        
        #get GPS datas
        location = raw_datas[['lon','lat']].to_numpy()
        speed = raw_datas['speed'].to_numpy()
        bearing = raw_datas['angle'].to_numpy()

        #get time 

        #interpolate data
        print("- interpolate datas")
        method = "linear"
        fF = interploateNan(timestamps_us,forces,kind=method,axis=0,fill_value="extrapolate")
        facc = interploateNan(timestamps_us,acc,kind=method,axis=0,fill_value="extrapolate")
        fgyro = interploateNan(timestamps_us,gyro,kind=method,axis=0,fill_value="extrapolate")
        fmag = interploateNan(timestamps_us,mag,kind=method,axis=0,fill_value="extrapolate")
        flocation = interploateNan(timestamps_us,location,kind=method,axis=0,fill_value="extrapolate")
        fspeed = interploateNan(timestamps_us,speed,kind=method,axis=0,fill_value="extrapolate")
        fbearing = interploateNan(timestamps_us,bearing,kind=method,axis=0,fill_value="extrapolate")

        #Resample data at defined frequency
        #new time vector
        resampler_us = np.arange(0,timestamps_us.max(),1e6/self.sampling_hz)
        self.datas['date_time'] = pd.DataFrame(pd.to_datetime(resampler_us,unit='us')+pd.DateOffset(hours=2,microseconds=self.offset_to_epoch_us))
        #convert date_time to matplotlib date
        self.datas['mdate_time'] =  mdates.date2num(self.datas.date_time)
        #add timestamps in ms
        #self.datas['time_ms'] = np.round(resampler_us/1e3,0)

        #resample data
        forces = fF(resampler_us)
        acc = facc(resampler_us)
        gyro = fgyro(resampler_us)
        mag = fmag(resampler_us)
        location = flocation(resampler_us)
        speed = fspeed(resampler_us)
        bearing = fbearing(resampler_us)

        #add raw forces to DataFrame
        self.datas[['F1','F2','F3','F4','F5','F6']] = forces

        #add raw IMU to DataFrame
        self.datas[['acc_x','acc_y','acc_z']] = acc
        self.datas[['gyro_x','gyro_y','gyro_z']] = gyro
        self.datas[['mag_x','mag_y','mag_z']] = mag

        #add GPS to DataFrame
        self.datas[['lon','lat']] = location
        self.datas['speed'] = speed
        self.datas['cap'] = bearing

        #add Wind to DataFrame
        if self.wind_file :
            self.datas[['wind_dir','wind_mean','wind_min','wind_max']] = self.raw_wind_datas[['Angle','Vent_moy','Vent_min','Vent_max']]

        print(self.datas.head())
        print(self.datas.tail())

        if self.wind_file :
            print("VMG computation")
            self._compute_vmg(speed,bearing)    
        print("START forces computation")
        self._compute_forces(forces)
        print("START attitude estimation")
        self._estimate_attitude(acc,gyro,mag)
        print("END of processing")
        

        #forces_array = np.concatenate((self.F_front,self.F_back,self.F_tot),axis=1)
        #len = resampler_us.size
        #len_contact = len
        #forces_array = np.concatenate((self.F_front.reshape((len,1)),self.F_back.reshape((len,1)),self.F_tot.reshape((len,1))),axis=1)
        #moments_array = np.concatenate((self.M_front_X.reshape((len,1)),self.M_front_Y.reshape((len,1)),self.M_back_X.reshape((len,1)),self.M_back_Y.reshape((len,1)),self.M_tot_X.reshape((len,1)),self.M_tot_Y.reshape((len,1))),axis=1)
        #contact_array = np.concatenate((self.P_front_X.reshape((len,1))+d/2,self.P_front_Y.reshape((len,1)),self.P_back_X.reshape((len,1))-d/2,self.P_back_Y.reshape((len,1)),self.P_tot_X.reshape((len,1)),self.P_tot_Y.reshape((len,1))),axis=1)
        #gps_array = np.concatenate((self.gps_location,self.gps_speed.reshape((len,1)),self.gps_bearing.reshape((len,1))),axis=1)
        #self.pd_raw_forces = pd.DataFrame(data=self.board_forces,columns=["F1","F2","F3","F4","F5","F6"])
        #self.pd_forces = pd.DataFrame(data=forces_array,columns=["front",'back','tot'])
        #self.pd_moment = pd.DataFrame(data=moments_array,columns=["front_X","front_Y",'back_X','back_Y','tot_X','tot_Y'])
        #self.pd_contact = pd.DataFrame(data=contact_array,columns=["front_X","front_Y",'back_X','back_Y','tot_X','tot_Y'])
        #self.pd_gps = pd.DataFrame(data=gps_array,columns=['lon','lat','speed','bearing'])
        #self.pd_angles_imu = pd.DataFrame(data=self.rot_imu,columns=['X_imu','Y_imu','Z_imu'])
        #self.pd_angles_marg = pd.DataFrame(data=self.rot_marg,columns=['X_marg','Y_marg','Z_marg'])


    def _compute_forces(self, F) :
        #compute force & moment
        Fz_front = np.sum(F[:,0:3],axis=1)
        Fz_back = np.sum(F[:,3:6],axis=1)
        Fz = np.sum(F[:,0:6],axis=1)
        Mx = (g1*F[:,1]+g2*F[:,5]-g1*F[:,2]-g2*F[:,4]) / 1000
        My = (e1*np.sum(F[:,1:3],axis=1)+e2*F[:,0]-e3*np.sum(F[:,4:6],axis=1)-e4*F[:,3]) / 1000
        Mx_front = (np.sin(alpha)* j1 * F[:,1] - np.sin(alpha) * j1 * F[:,2]) / 1000
        My_front = (j1*F[:,0] - np.cos(alpha)*j1*np.sum(F[:,1:3],axis=1)) / 1000
        Mx_back = (np.sin(alpha)*h2 * F[:,5] - np.sin(alpha)*h2 * F[:,4]) / 1000
        My_back = (np.cos(alpha)*h2*np.sum(F[:,4:6],axis=1) - h1*F[:,3]) / 1000

        #Compute force application point
        Px_front = My_front * 1000 / Fz_front
        Py_front = -Mx_front * 1000 / Fz_front
        Px_back = My_back * 1000 / Fz_back
        Py_back = -Mx_back * 1000 / Fz_back
        Px = My * 1000 / Fz
        Py = -Mx * 1000 / Fz

        #define a mask to compute reaction point only when force is applied on board
        mask = np.abs(Fz > threshold)
        Px_front[~mask] = np.nan
        Py_front[~mask] = np.nan
        Px_back[~mask] = np.nan
        Py_back[~mask] = np.nan
        Px[~mask] = np.nan
        Py[~mask] = np.nan

        #Copy data to pandas
        self.datas['Fz_front'] = Fz_front
        self.datas['Fz_back'] = Fz_back
        self.datas['Fz'] = Fz
        
        self.datas['Mx'] = Mx
        self.datas['My'] = My
        self.datas['Mx_front'] = Mx_front
        self.datas['My_front'] = My_front
        self.datas['Mx_back'] = Mx_back
        self.datas['My_back'] = My_back

        self.datas['Px_front'] = Px_front + d/2
        self.datas['Py_front'] = Py_front
        self.datas['Px_back'] = Px_back -d/2
        self.datas['Py_back'] = Py_back
        self.datas['Px'] = Px
        self.datas['Py'] = Py

        #save sensor position
        self.P_front_X_sensor = np.array([j1 , -j1*np.cos(alpha), -j1*np.cos(alpha)]) + d/2
        self.P_front_Y_sensor = [0 , -j1*np.sin(alpha), j1*np.sin(alpha)] 

        self.P_back_X_sensor = np.array([-h1 , h2*np.cos(alpha), h2*np.cos(alpha)]) - d/2
        self.P_back_Y_sensor = [0 , h2*np.sin(alpha), -h2*np.sin(alpha)]

    def _estimate_attitude(self,acc,gyro,mag) :
        #filter signal
        sos = signal.butter(4, 5, 'low',fs=40,output='sos')
        acc_filtered = signal.sosfiltfilt(sos,acc,axis=0)
        gyro_filtered = signal.sosfiltfilt(sos,gyro,axis=0)
        mag_filtered = mag


        from ahrs.filters import Madgwick
        q0 = R.from_rotvec([0,0,0],degrees=True)
        q_zero = np.roll(q0.as_quat(),1)
        q_zero /= np.linalg.norm(q_zero)
        IMU = Madgwick(acc=acc_filtered*9.81,gyr=gyro_filtered*np.pi/180, frequency = self.sampling_hz,q0=q_zero)#,mag=self.mag_filtered*1e3)
        MARG = Madgwick(acc=acc_filtered*9.81,gyr=gyro_filtered*np.pi/180, frequency = self.sampling_hz,q0=q_zero,mag=mag_filtered*1e3)

        #convert quaternion to Euler angle, roll array to fit quaternion convention of scipy and ahrs
        r_imu = R.from_quat(np.roll(IMU.Q,-1))
        r_marg = R.from_quat(np.roll(MARG.Q,-1))

        rot_imu = r_imu.as_euler('xyz',degrees=True)
        self.datas[['Euler_X','Euler_Y','Euler_Z']] = rot_imu
        rot_marg = r_marg.as_euler("xyz",degrees=True)
        rot_marg[:,2] = (rot_marg[:,2]+180) % 360 
        #r_mag.as_rotvec(degrees=True)

    def _compute_vmg(self,speed,bearing):
        bearing = np.radians(bearing)
        wind_dir = np.radians(self.raw_wind_datas['Angle'].to_numpy()+np.pi) #np.pi is added to get the angle between the two vector at the same orgin
        cap = np.array([np.cos(bearing), np.sin(bearing)])
        wind = np.array([np.cos(wind_dir), np.sin(wind_dir)])
        dot = []
        for i in range(len(bearing)) : 
            dot.append(np.dot(cap[:,i],wind[:,i]))
        TWA = np.arccos(dot)
        self.datas['TWA'] = np.degrees(TWA)
        VMG = speed*np.cos(TWA)
        self.datas['VMG'] = VMG

if __name__ == '__main__':
    kite = Kite()
    kite.set_folder(r"C:\Users\jparent1\Documents\GitHub\ISSUL-Kite-Analyse\data\test-angles\22_5_4-13_39_7")
    kite.read_bin_files_to_pandas()
    kite.analyze()
    kite.range = [0,9200]
    range = slice(kite.range[0],kite.range[1]+1)
    time = kite.time_us
    import matplotlib.pyplot as plt

    fig_imu = plt.figure("IMU")
    plt.subplot(1,3,1)
    plt.title("acc")
    #plt.plot(time[range],kite.board_acc[range])
    plt.plot(time[range],np.linalg.norm(kite.acc_filtered[range,:],axis=1),'r')
    plt.plot(time[range],kite.acc_filtered[range])
    plt.legend(('norm','X','Y','Z'))
    plt.subplot(1,3,2)
    plt.title("gyro")
    #plt.plot(time[range],kite.board_gyro[range])
    plt.plot(time[range],kite.gyro_filtered[range])
    plt.legend(('X','Y','Z'))
    plt.subplot(1,3,3)
    plt.title("mag")
    #plt.plot(time[range],kite.board_mag[range])
#    plt.plot(time[range],kite.mag_filtered[range])
    plt.legend(('X','Y','Z'))
    #plt.show()

    mag = kite.rot_mag
    ekf = kite.rot_ekf
    mahony = kite.rot_mahony
    #fourati = kite.rot_fourati
    plt.figure("angles")
    plt.subplot(4,1,1)
    plt.title("Magdwick")
    plt.plot(time[range],mag[range,:])
    plt.legend(('X','Y','Z'))
    plt.subplot(4,1,2)
    plt.title("Fourati")
    #plt.plot(time[range],fourati[range,:])
    plt.legend(('X','Y','Z'))
    plt.subplot(4,1,3)
    plt.title("Mahony")
    plt.plot(time[range],mahony[range,:])
    plt.legend(('X','Y','Z'))
    plt.subplot(4,1,4)
    plt.title("EKF")
    plt.plot(time[range],ekf[range,:])
    plt.legend(('X','Y','Z'))
    plt.show()