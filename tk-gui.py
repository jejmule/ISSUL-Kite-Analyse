#from prompt_toolkit import PromptSession
from Kite import Kite
from tkinter import *
from tkinter import filedialog

import matplotlib.pyplot as plt
from matplotlib import widgets

import numpy as np

#Define Tk root window
root = Tk()
root.title("ISSUL lab kite analyzer")

#Define kite class
kite = Kite()

def plot_data() : 
    range = slice(kite.range[0],kite.range[1]+1)
    print(range)
    time = kite.time_us
    angles = kite.rot_mag

    fig1, axs = plt.subplots(nrows=3,ncols=1,label='FORCES')
    kite.pd_forces.loc[range].plot(ax=axs[0],legend=True,title="Forces")
    kite.pd_moment.loc[range].plot(ax=axs[1],legend=True,title="Moment")
    kite.pd_contact.loc[range].plot(x="front_X",y="front_Y",ax=axs[2],legend=True,title="Contact",style='.')
    kite.pd_contact.loc[range].plot(x='back_X',y='back_Y',ax=axs[2],legend=True,title="Contact",style='.')
    kite.pd_contact.loc[range].plot(x='tot_X',y='tot_Y',ax=axs[2],legend=True,title="Contact",style='.')
    fig1.canvas.draw_idle()

    fig2, axs = plt.subplots(nrows=3,ncols=1,label='IMU')
    axs[0].set_title('acc')
    axs[0].plot(time[range],kite.board_acc[range])
    #plt.plot(time[range],np.linalg.norm(kite.board_acc[range,:],axis=1),'r')
    axs[0].plot(time[range],kite.acc_filtered[range])
    axs[0].legend(('X','Y','Z'))
    axs[1].set_title("gyro")
    axs[1].plot(time[range],kite.board_gyro[range])
    axs[1].plot(time[range],kite.gyro_filtered[range])
    axs[1].legend(('X','Y','Z'))
    axs[2].set_title("mag")
    axs[2].plot(time[range],kite.board_mag[range])
    axs[2].legend(('X','Y','Z'))
    fig2.canvas.draw_idle()

    fig3, axs = plt.subplots(nrows=3,ncols=1,label='Angles')
    kite.pd_angles.loc[range,'X'].plot(ax=axs[0],legend=False,title="rotation X")
    kite.pd_angles.loc[range,'Y'].plot(ax=axs[1],legend=False,title="rotation Y")
    kite.pd_angles.loc[range,'Z'].plot(ax=axs[2],legend=False,title="rotation Z")

    plt.show()

def trace_gps():
    #get and interpolate data to be traced
    lon = kite.pd_gps['lon']
    lat = kite.pd_gps['lat']
    speed = kite.pd_gps['speed']
    bearing = kite.pd_gps['bearing']
    #create figure
    fig, axs = plt.subplots(nrows=1,ncols=2)
    plt.subplots_adjust(bottom=0.25)
    #plot trace
    p0 = axs[0].plot(lon,lat)
    axs[0].set_title('GPS trace')
    axs[0].axis('equal')

    #plot speed
    p1 = axs[1].plot(speed)
    axs[1].set_title('GPS speed')
    axs[1].set_ylabel('speed [knots]')
    axs[1].set_xlabel('indexes')
    # create axes
    slider_ax = plt.axes([0.20, 0.1, 0.60, 0.03])
    button_ax = plt.axes([0.05, 0.025, 0.9, 0.05])
    # create slider
    slider = widgets.RangeSlider(slider_ax, "set range", 0, len(kite.pd_gps),valstep=1,closedmax=False)
    val = [int(x) for x in slider.val]
    kite.range = val
    # plot start and end point controlled by slider
    start, = axs[0].plot(lon[val[0]],lat[val[0]], marker='o',markersize=10,markerfacecolor='green')
    stop, = axs[0].plot(lon[val[1]],lat[val[1]], marker='o',markersize=10,markerfacecolor='red')
    # Create the Vertical lines on the histogram
    lower_limit_line = axs[1].axvline(val[0], color='k')
    upper_limit_line = axs[1].axvline(val[1], color='k')
    # callback of slider 
    def update(val) :
        val = [int(x) for x in slider.val]
        start.set_xdata(lon[val[0]])
        start.set_ydata(lat[val[0]])
        stop.set_xdata(lon[val[1]])
        stop.set_ydata(lat[val[1]])
        lower_limit_line.set_xdata([val[0], val[0]])
        upper_limit_line.set_xdata([val[1], val[1]])
        fig.canvas.draw_idle()
        kite.range = val
    slider.on_changed(update)
    # add a confirmation button
    button = widgets.Button(button_ax,"apply")
    def apply(event) :
        kite.filter_data()
        plot_data()
    button.on_clicked(apply)
    mgr = plt.get_current_fig_manager()
    mgr.window.showMaximized()
    #show plot
    #fig.canvas.draw()
    plt.show()

def select_folder() :
    root.folder = filedialog.askdirectory()
    kite.set_folder(root.folder)
    kite.read_bin_files_to_pandas()
    kite.analyze()
    trace_gps()

def show_plots() :
    kite.plot_data()

def save_raw() :
    kite.write_raw_data()

def save_segmented() : 
    kite.write_filter_data()

load_btn = Button(root, text="load data", command=select_folder).pack()
save_raw = Button(root, text="save all to xls", command=save_raw).pack()
save_seg = Button(root, text="save segmented to xls", command=save_segmented).pack()

root.mainloop()