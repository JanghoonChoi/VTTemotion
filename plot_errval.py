import os, time
import numpy as np
import matplotlib as mp
mp.use('Agg')

import matplotlib.pyplot as plt
import pylab

def eval_errval():
    timeNow = time.localtime()
    timeStr = str('%02d'%timeNow.tm_mon)+'-'+str('%02d'%timeNow.tm_mday)+','+str('%02d'%timeNow.tm_hour)+':'+str('%02d'%timeNow.tm_min)+':'+str('%02d'%timeNow.tm_sec)
    print timeStr
    
    all_files = np.array(os.listdir("err_log/"))
    all_files.sort()

    for selfile in all_files:
        if selfile[-3:] != 'txt':
            all_files = np.delete(all_files, np.where(all_files==selfile))
    
    sel_newest = pylab.genfromtxt("err_log/"+all_files[-1],delimiter=',')
    #sel_newest = sel_newest[25:,:]
    nums = sel_newest.shape[0]
    errs = sel_newest[:,0]
    vals = sel_newest[:,1]
    accs = sel_newest[:,2]

    print nums
    fig, ax = plt.subplots(1,2)
    fig.set_figheight(6)
    fig.set_figwidth(16)
    fig.suptitle('updated: ' + timeStr,fontsize=10)
    ax[0].plot(range(nums),errs,range(nums),vals)
    ax[1].plot(range(nums),accs)
    plt.savefig('plot.png')
    plt.close(fig)
    time.sleep(60)

while True:
    eval_errval()
