
import struct
from pathlib import Path

from datetime import datetime
#from tkinter import Button

import numpy as np
import pandas as pd
from pandas import DataFrame, Timedelta

#import xlsxwriter as xlsx

from scipy import interpolate
from scipy import signal

import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Button

#sampling rate
sampling_hz = 80 

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
threshold = 100     #threshold in N to find pressure point

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
    files = []
    folder = None
    range = None
    
    #set experiement folder and find bin files
    def set_folder(self,path) :
        folder = Path(path)
        count = 0
        for file in folder.iterdir() :
            if file.suffix == ".bin" :
                print("Data file : "+str(file))
                count += 1
        self.files = []
        for i in range(count) :
            self.files.append(folder/"log_{:04d}.bin".format(i))
        self.folder = folder

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
                dataframes.append(DataFrame(np.fromfile(f, dt)))

        #concatenate all dataframes into one
        self.raw_data = pd.concat(dataframes,ignore_index=True)
        #remove offset in time
        self.raw_data['t'] -= self.raw_data['t'].min()
        #convert first gps date to epoch
        offset_to_epoch_ns = pd.Timestamp(year=self.raw_data['year'][0]+2000,month=self.raw_data['month'][0],day=self.raw_data['day'][0],hour=self.raw_data['hour'][0],minute=self.raw_data['min'][0],second=self.raw_data['sec'][0],microsecond=self.raw_data['msec'][0]*1000)
        #add a timestamp series
        self.raw_data['Timestamp'] = pd.to_datetime(self.raw_data['t']+offset_to_epoch_ns.value/1000,unit='us')
        self.raw_data.set_index('Timestamp',inplace=True)
        #remove max value used as NAN on byte informations
        self.raw_data = self.raw_data.replace(to_replace={'msec':65535,'year':255,'month':255,'day':255,'hour':255,'min':255,'sec':255,'satellites':255,'fix_quality':255},value=np.nan)
        #self.filled_data = self.raw_data.asfreq('12500US',method='pad')
        self.raw_data.drop(columns=['msec','year','month','day','hour','min','sec','satellites','fix_quality','lat_NS','lon_EW','mag_EW','empty','mag_var','precision','altitude'],inplace=True)
        print(self.raw_data.head())
    
    def write_raw_data(self) :
        self.raw_data.to_excel(self.folder/"raw_data.xlsx",freeze_panes=(1,0))

    def write_filter_data(self) :
        range = slice(self.range[0],self.range[1])
        self.filter_datas.loc[range].to_excel(self.folder/"segmented_data.xlsx",freeze_panes=(1,0))
    
    def filter_data(self) :
        filter_datas = self.raw_data.copy()
        filter_datas['t'] -= filter_datas['t'].min()
        filter_datas['t'] /= 1e3
        filter_datas = filter_datas.interpolate(limit=16,method='cubic')
        self.filter_datas = filter_datas
        #print(filter_data['t'])

    def analyze(self) :
        datas = self.raw_data
        #Get numpy array from panda
        timestamps_us = datas['t']
        forces = datas[['F1','F2','F3','F4','F5','F6']].to_numpy()
        acc = datas[['Ax','Az','Ay']].to_numpy()*[-1,1,-1]
        gyro = datas[['Gx','Gz','Gy']].to_numpy()*[-1,-1,-1]
        mag = datas[['Mx','Mz','My']].to_numpy()*[1,1,-1]
        #remove outliners with a median filter of size 3 and smooth data
        forces = signal.medfilt(forces,(3,1))
        #get numpy from pandas
        location = datas[['lon','lat']].to_numpy()
        speed = datas['speed'].to_numpy()
        bearing = datas['angle'].to_numpy()

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
        resampler_us = np.arange(0,timestamps_us.max(),1e6/sampling_hz)
        #resample data
        self.board_forces = fF(resampler_us)
        self.board_acc = facc(resampler_us)
        self.board_gyro = fgyro(resampler_us)
        self.board_mag = fmag(resampler_us)
        self.gps_location = flocation(resampler_us)
        self.gps_speed = fspeed(resampler_us)
        self.gps_bearing = fbearing(resampler_us)
        self.time_us = resampler_us

        print("START forces computation")
        self._compute_forces()
        print("START attitude estimation")
        self._estimate_attitude()
        print("END of processing")

        #forces_array = np.concatenate((self.F_front,self.F_back,self.F_tot),axis=1)
        len = self.F_front.size
        len_contact = len
        forces_array = np.concatenate((self.F_front.reshape((len,1)),self.F_back.reshape((len,1)),self.F_tot.reshape((len,1))),axis=1)
        moments_array = np.concatenate((self.M_front_X.reshape((len,1)),self.M_front_Y.reshape((len,1)),self.M_back_X.reshape((len,1)),self.M_back_Y.reshape((len,1)),self.M_tot_X.reshape((len,1)),self.M_tot_Y.reshape((len,1))),axis=1)
        contact_array = np.concatenate((self.P_front_X.reshape((len,1))+e1,self.P_front_Y.reshape((len,1)),self.P_back_X.reshape((len,1))-e3,self.P_back_Y.reshape((len,1)),self.P_tot_X.reshape((len,1)),self.P_tot_Y.reshape((len,1))),axis=1)
        gps_array = np.concatenate((self.gps_location,self.gps_speed.reshape((len,1)),self.gps_bearing.reshape((len,1))),axis=1)
        self.pd_forces = pd.DataFrame(data=forces_array,columns=["front",'back','tot'])
        self.pd_moment = pd.DataFrame(data=moments_array,columns=["front_X","front_Y",'back_X','back_Y','tot_X','tot_Y'])
        self.pd_contact = pd.DataFrame(data=contact_array,columns=["front_X","front_Y",'back_X','back_Y','tot_X','tot_Y'])
        self.pd_gps = pd.DataFrame(data=gps_array,columns=['lon','lat','speed','bearing'])
        self.pd_angles = pd.DataFrame(data=self.rot_mag,columns=['X','Y','Z'])
        print(self.pd_forces.head())


    def _compute_forces(self) :
        F = self.board_forces
        self.F_front = np.sum(F[:,0:3],axis=1)
        self.F_back = np.sum(F[:,3:6],axis=1)
        self.F_tot = np.sum(F[:,0:6],axis=1)
        self.M_tot_X = (g1*F[:,1]+g2*F[:,5]-g1*F[:,2]-g2*F[:,4]) / 1000
        self.M_tot_Y = (e1*np.sum(F[:,1:3],axis=1)+e2*F[:,0]-e3*np.sum(F[:,4:6],axis=1)-e4*F[:,3]) / 1000

        self.M_front_X = (np.sin(alpha)* j1 * F[:,1] - np.sin(alpha) * j1 * F[:,2]) / 1000
        self.M_front_Y = (j1*F[:,0] - np.cos(alpha)*j1*np.sum(F[:,1:3],axis=1)) / 1000

        self.M_back_X = (np.sin(alpha)*h2 * F[:,5] - np.sin(alpha)*h2 * F[:,4]) / 1000
        self.M_back_Y = (np.cos(alpha)*h2*np.sum(F[:,4:6],axis=1) - h1*F[:,3]) / 1000

        #define a mask to compute reaction point only when force is applied on board
        self.P_front_X = self.M_front_Y * 1000 / self.F_front
        self.P_front_Y = -self.M_front_X * 1000 / self.F_front

        self.P_back_X = self.M_back_Y * 1000 / self.F_back
        self.P_back_Y = -self.M_back_X * 1000 / self.F_back

        self.P_tot_X = self.M_tot_Y * 1000 / self.F_tot
        self.P_tot_Y = -self.M_tot_X * 1000 / self.F_tot

        #define a mask to compute reaction point only when force is applied on board
        mask = np.abs(self.F_tot > threshold)
        self.P_front_X[~mask] = np.nan
        self.P_front_Y[~mask] = np.nan
        self.P_back_X[~mask] = np.nan
        self.P_back_Y[~mask] = np.nan
        self.P_tot_X[~mask] = np.nan
        self.P_tot_Y[~mask] = np.nan

        self.P_front_X_sensor = [0, j1 , -j1*np.cos(alpha), -j1*np.cos(alpha)] 
        self.P_front_Y_sensor = [0, 0 , -j1*np.sin(alpha), j1*np.sin(alpha)] 

        self.P_back_X_sensor = [0, -h1 , h2*np.cos(alpha), h2*np.cos(alpha)] 
        self.P_back_Y_sensor = [0, 0 , h2*np.sin(alpha), -h2*np.sin(alpha)] 

    def _estimate_attitude(self) :
        #filter signal
        sos = signal.butter(4, 5, 'low',fs=80,output='sos')
        self.acc_filtered = signal.sosfiltfilt(sos,self.board_acc,axis=0)
        self.gyro_filtered = signal.sosfiltfilt(sos,self.board_gyro,axis=0)+[0.9,0.5,0.5]
        #self.mag_filtered = signal.sosfiltfilt(sos,self.board_mag,axis=0)
        self.mag_filtered = None

        #from ahrs.filters import AQUA, EKF
        #aqua = AQUA(acc=self.acc_filtered*9.81,gyr=self.gyro_filtered,frequency = sampling_hz)#,mag=self.mag_filtered*1e-3

        from ahrs.filters import Madgwick
        madgwick = Madgwick(acc=self.acc_filtered*9.81,gyr=self.gyro_filtered*np.pi/180, frequency = sampling_hz,q0=[1,0,0,0])#,mag=self.mag_filtered*1e3

        #from ahrs.filters import Mahony
        #mahony = Mahony(acc=self.acc_filtered*9.81,gyr=self.gyro_filtered, frequency = sampling_hz,q0=[1,0,0,0])#,mag=self.mag_filtered*1e3

        #from ahrs.filters import Fourati
        #fourati = Fourati(acc=self.acc_filtered*9.81,gyr=self.gyro_filtered, frequency = sampling_hz,q0=[1,0,0,0])#,mag=self.mag_filtered*1e3

        #from ahrs.filters import EKF
        #ekf = EKF(acc=self.board_acc*9.81,gyr=self.board_gyro, frequency = sampling_hz, noises = [0.5**2, 0.8**2, 0.3**2],q0=[1,0,0,0])#,mag=self.mag_filtered*1e-3

        #convert quaternion to Euler angle more specifically Tait-Bryan angles
        from scipy.spatial.transform import Rotation as R
        #r_aqua = R.from_quat(np.roll(aqua.Q,-1))
        r_mag = R.from_quat(np.roll(madgwick.Q,-1))
        #r_ekf = R.from_quat(np.roll(ekf.Q,-1))
        #r_mahony = R.from_quat(np.roll(mahony.Q,-1))
        #r_fourati = R.from_quat(np.roll(fourati.Q,-1))

        #Tait-Bryan angles
        #self.angles_aqua = r_aqua.as_euler('zxy', degrees=False)*180/np.pi
        #self.angles_aqua = r_aqua.as_euler('zxy', degrees=False)*180/np.pi
        #self.angles_ekf = r_ekf.as_euler('zxy', degrees=False)*180/np.pi

        #self.rot_aqua = r_aqua.as_rotvec()*180/np.pi
        self.rot_mag = r_mag.as_rotvec()*180/np.pi
        #self.rot_ekf = r_ekf.as_rotvec()*180/np.pi
        #self.rot_mahony = r_mahony.as_rotvec()*180/np.pi
        #self.rot_fourati = r_fourati.as_rotvec()*180/np.pi


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