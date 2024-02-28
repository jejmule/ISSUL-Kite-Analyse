#from prompt_toolkit import PromptSession
from Kite import Kite
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import widgets
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
#from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd

#Define Tk root window
root = Tk()
root.geometry("500x150+50+50")
root.title("ISSUL lab kite analyzer")
#chart = Toplevel(root)

#Define kite class
kite = Kite()

def save_multi_image(folder,filename):
   #pp = PdfPages(filename)
   fig_nums = plt.get_fignums()
   figs = [plt.figure(n) for n in fig_nums]
   i=1
   for fig in figs:
      #fig.savefig(pp, format='pdf')
      file = filename+"_fig_"+str(i)+".png"
      #plt.figure(i)
      plt.get_current_fig_manager().window.state('zoomed')
      #splt.show()
      #import time
      #time.sleep(1)
      fig.savefig(folder/file, format='png',dpi=128)
      if i > 1 :
        plt.close(fig)
      i+=1
   #pp.close()

def plot_data() : 
    range = slice(kite.range[0],kite.range[1]+1)
    print(range)
    #time = kite.datas.loc[range,'date_time']
    #index = np.arange(len(time))

    #PLOT RAW FORCES
    kite.datas.loc[range,['mdate_time','F1','F2','F3','F4','F5','F6']].plot(x='mdate_time',subplots=True,sharex=True,title="Raw Forces")
    plt.get_current_fig_manager().window.state('zoomed')

    #PLOT FORCES MOMENT
    fig1, axs = plt.subplots(nrows=3,ncols=1,label='FORCES',sharex=True)
    kite.datas.loc[range,['mdate_time','Fz_front','Fz_back','Fz']].plot(x='mdate_time',ax=axs[0],legend=True,title="Forces Z",ylabel='[N]')
    kite.datas.loc[range,['mdate_time','Mx_front','Mx_back','Mx']].plot(x='mdate_time',ax=axs[1],legend=True,title="Moments X",ylabel='[Nm]')
    kite.datas.loc[range,['mdate_time','My_front','My_back','My']].plot(x='mdate_time',ax=axs[2],legend=True,title="Moments Y",ylabel='[Nm]')
    axs[2].set_xlabel("HH-MM-SS")
    axs[2].xaxis.set_major_locator(mdates.SecondLocator(interval=2))
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H-%M-%S'))
    
    fig1.canvas.draw_idle()
    plt.get_current_fig_manager().window.state('zoomed')

    #PLOT CONTACT
    fig2, axs = plt.subplots(nrows=1,ncols=1,label='Contact Point')
    kite.datas.loc[range].plot(x="Px_front",y="Py_front",ax=axs,legend=True,title="Contact point",style='.')
    kite.datas.loc[range].plot(x='Px_back',y='Py_back',ax=axs,legend=True,title="Contact point",style='.')
    kite.datas.loc[range].plot(x='Px',y='Py',ax=axs,legend=True,title="Contact point",style='.')
    axs.plot(kite.P_front_X_sensor,kite.P_front_Y_sensor,'*')
    axs.plot(kite.P_back_X_sensor,kite.P_back_Y_sensor,'*')
    axs.plot([-728/2,728/2],[0,0],'D')
    axs.legend(('F application front','F application back','F application total','front sensor','back sensor','sensor center'),loc='center left', bbox_to_anchor=(1, 0.5))
    axs.set_ylabel('X [mm]')
    axs.set_xlabel('Y [mm]')
    axs.set_xlim(-500,600)
    axs.set_ylim(-200,200)

    fig2.canvas.draw_idle()
    plt.get_current_fig_manager().window.state('zoomed')

    #PLOT RAW IMU datas
    fig3, axs = plt.subplots(nrows=3,ncols=1,label='IMU')
    kite.datas.loc[range,['mdate_time','acc_x','acc_y','acc_z']].plot(ax=axs[0],x='mdate_time',legend=True,title="Accelerations")
    kite.datas.loc[range,['mdate_time','gyro_x','gyro_y','gyro_z']].plot(ax=axs[1],x='mdate_time',legend=True,title="Gyroscopes")
    kite.datas.loc[range,['mdate_time','mag_x','mag_y','mag_z']].plot(ax=axs[2],x='mdate_time',legend=True,title="Magnetic fields")
    fig3.canvas.draw_idle()
    plt.get_current_fig_manager().window.state('zoomed')

    #PLOT ANGLES datas
    fig4, axs = plt.subplots(nrows=3,ncols=1,label='Angles',sharex=True)
    kite.datas.loc[range,['mdate_time','Euler_X']].plot(ax=axs[0],x='mdate_time',legend=True,title="Euler X",ylabel="[°]",)
    kite.datas.loc[range,['mdate_time','Euler_Y']].plot(ax=axs[1],x='mdate_time',legend=True,title="Euler Y",ylabel="[°]")
    if kite.wind_file : 
        kite.datas.loc[range,['mdate_time','cap','TWA']].plot(ax=axs[2],x='mdate_time',legend=True,title="Angle Z",ylabel="[°]")
        kite.datas.loc[range,['mdate_time','wind_dir']].plot(ax=axs[2],x='mdate_time',legend=True,title="Wind moy.",ylabel="[°]")
    else :  
        kite.datas.loc[range,['mdate_time','cap']].plot(ax=axs[2],x='mdate_time',legend=True,title="Angle Z",ylabel="[°]")
    axs[2].set_xlabel("HH-MM-SS")
    axs[2].xaxis.set_major_locator(mdates.SecondLocator(interval=2))
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H-%M-%S'))

    plt.get_current_fig_manager().window.state('zoomed')

    

    #fig4,axs = plt.subplots(nrows=1,ncols=1,label='Mag cut')
    #axs.plot(kite.board_mag[range,0],kite.board_mag[range,1],'.')
    #axs.plot(kite.board_mag[range,1],kite.board_mag[range,2],'.')
    #axs.plot(kite.board_mag[range,2],kite.board_mag[range,0],'.')
    #axs.legend(('XY','YZ','ZX'))

    #fig5,axs = plt.subplots(nrows=3,ncols=3,label='histograms')
    bins_forces = np.arange(0,2010,10)
    bins_moment = np.arange(-200,202,2)
    bins_angles = np.arange(-70,71,1)
    ax = kite.datas.loc[range,['Fz_front','Fz_back','Fz']].plot.hist(bins=bins_forces,alpha=0.5,title="Z Forces histogram")
    ax.set_xlabel("[N]")
    ax.set_ylabel("Occurrences ")
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(100))
    plt.get_current_fig_manager().window.state('zoomed')
    ax=kite.datas.loc[range,['Mx_front','Mx_back','Mx']].plot.hist(bins=bins_moment,alpha=0.5,title="X Moment histogram")
    ax.set_xlabel("[Nm]")
    ax.set_ylabel("Occurrences ")
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(2))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    plt.get_current_fig_manager().window.state('zoomed')
    ax=kite.datas.loc[range,['My_front','My_back','My']].plot.hist(bins=bins_moment,alpha=0.5,title="Y Moment histogram")
    ax.set_xlabel("[Nm]")
    ax.set_ylabel("Occurrences ")
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(2))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    plt.get_current_fig_manager().window.state('zoomed')
    ax=kite.datas.loc[range,['Euler_X','Euler_Y']].plot.hist(bins=bins_angles,title='Angles histogram')
    ax.set_xlabel("[°]")
    ax.set_ylabel("Occurrences ")
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    plt.get_current_fig_manager().window.state('zoomed')


    X = pd.concat([kite.datas.loc[range]['Px_front'],kite.datas.loc[range]['Px_back'],kite.datas.loc[range]['Px']],ignore_index=True)
    Y = pd.concat([kite.datas.loc[range]['Py_front'],kite.datas.loc[range]['Py_back'],kite.datas.loc[range]['Py']],ignore_index=True)
    h,xedges,yedges = np.histogram2d(X,Y,range= [[-728/2-200,728/2+200], [-200,200]],bins=300)
    plt.figure()
    ax = plt.subplot()
    ax.plot(kite.P_front_X_sensor,kite.P_front_Y_sensor,'*')
    ax.plot(kite.P_back_X_sensor,kite.P_back_Y_sensor,'*')
    ax.plot([-728/2,728/2,0],[0,0,0],'D')
    ax.legend(('front sensor','back sensor','center'), loc="lower left")
    ax.set_title('Contact point histogram')
    ax.set_ylabel('X [mm]')
    ax.set_xlabel('Y [mm]')
    im = ax.imshow(h.T,interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],cmap='hot')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im,cax=cax)
    plt.get_current_fig_manager().window.state('zoomed')
    #plt.get_current_fig_manager().window.state('zoomed')
    plt.show()

def trace_gps():
    #get and interpolate data to be traced
    lon = kite.datas['lon']
    lat = kite.datas['lat']
    speed = kite.datas['speed']
    if kite.wind_file : 
        VMG = kite.datas['VMG']
        wind = kite.datas['wind_mean']
    time = kite.datas['mdate_time']
    #create figure
    fig, axs = plt.subplots(nrows=1,ncols=2,num=kite.folder.name)
    plt.subplots_adjust(bottom=0.25)
    #plot trace
    p0 = axs[0].plot(lon,lat)
    axs[0].set_title('GPS trace')
    axs[0].axis('equal')
    axs[0].set_ylabel('latitude [°]')
    axs[0].set_xlabel('longitude [°]')

    #plot speed
    p1 = axs[1].plot(time,speed)
    if kite.wind_file :
        p2 = axs[1].plot(time,VMG)
        p3 = axs[1].plot(time,wind)
    axs[1].set_title('GPS speed')
    axs[1].set_ylabel('speed [knots]')
    axs[1].set_xlabel('HH-MM')
    axs[1].xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H-%M'))
    axs[1].legend(('GPS_speed','VMG','Wind_speed'), loc="lower left")#bbox_to_anchor=(0, 0),bbox_transform=axs[1].transAxes)
    time_delta = axs[1].text(0.9, 0.95, 'duration', horizontalalignment='center',verticalalignment='center', transform=axs[1].transAxes)
    # create axes
    slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
    button_ax = plt.axes([0.05, 0.025, 0.9, 0.05])
    # create slider
    slider = widgets.RangeSlider(slider_ax, "set range", 0, len(time),valstep=1,closedmax=False)
    val = [int(x) for x in slider.val]
    kite.range = val
    # plot start and end point controlled by slider
    start, = axs[0].plot(lon[val[0]],lat[val[0]], marker='o',markersize=10,markerfacecolor='green')
    stop, = axs[0].plot(lon[val[1]],lat[val[1]], marker='o',markersize=10,markerfacecolor='red')
    # Create the Vertical lines on the histogram
    lower_limit_line = axs[1].axvline(time[val[0]], color='k')
    upper_limit_line = axs[1].axvline(time[val[1]], color='k')
    # callback of slider 
    def update(val) :
        val = [int(x) for x in slider.val]
        start.set_xdata(lon[val[0]])
        start.set_ydata(lat[val[0]])
        stop.set_xdata(lon[val[1]])
        stop.set_ydata(lat[val[1]])
        lower_limit_line.set_xdata([time[val[0]], time[val[0]]])
        upper_limit_line.set_xdata([time[val[1]], time[val[1]]])
        fig.canvas.draw_idle()
        kite.range = val
        delta = kite.datas.loc[val[1],'date_time']-kite.datas.loc[val[0],'date_time']
        time_delta.set_text(str("{:.2f} s".format(delta.total_seconds())))
    update(val)
    slider.on_changed(update)
    # add a confirmation button
    button = widgets.Button(button_ax,"apply")
    def apply(event) :
        plot_data()
    button.on_clicked(apply)

    #canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    #canvas.draw()
    #canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    #toolbar = NavigationToolbar2Tk(canvas, root)
    #toolbar.update()
    #canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    #mgr = plt.get_current_fig_manager()
    #mgr.window.showMaximized()
    #show plot
    #fig.canvas.draw()
    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()

def select_folder() :
    root.folder = filedialog.askdirectory()
    kite.set_folder(root.folder,sampling_combo.get())
    kite.read_bin_files_to_pandas()
    segment_combo.config(values=kite.segments_list)
    segment_combo.current(0)
    kite.analyze()
    trace_gps()

def show_plots() :
    kite.plot_data()

def save_raw() :
    kite.write_raw_data()

def save_segmented() : 
    print("Saving segment to excel ...")
    kite.write_datas(segment_combo.get())
    filename = segment_combo.get()
    folder = kite.folder/"plots"
    print("Saving plot...")
    if not folder.exists() :
        folder.mkdir()
    save_multi_image(folder,filename)
    print("Done")

load_btn = Button(root, text="load data", command=select_folder)
load_btn.pack(pady = 5,side=LEFT,expand=True)
sampling_combo = ttk.Combobox(root, values=["10 Hz","20 Hz","50Hz","80 Hz"])
sampling_combo.current(0)
sampling_combo.pack(pady = 5,side=LEFT,expand=True)
segment_combo = ttk.Combobox(root, values=["load data first"])
segment_combo.current(0)
segment_combo.pack(pady = 5,side=LEFT,expand=True)
#save_raw = Button(root, text="save all to xls", command=save_raw).pack()
save_seg = Button(root, text="Save segment", command=save_segmented)
save_seg.pack(pady=5,side=LEFT,expand=True)

root.mainloop()