#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhaseNet wrapper
    1) convert miniseed files to npz files for PhaseNet input
    2) Run PhaseNet to detect phases
    3) Phrase the output to REAL input format

Created on Wed Jul  8 17:49:17 2020
@author: jw
"""
from obspy import read
import os
import numpy as np
import pandas as pd
from datetime import date, timedelta
import pickle

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

MAX_INT32 = 2147483647
BATCH_SIZE = 3000
STRIDE = 1500

def gen_day_mseed(mseed_dir, mseed_day_dir):
    if not os.path.exists(mseed_day_dir):
        os.makedirs(mseed_day_dir)

    st = read(mseed_dir + '/*.mseed')
    startdate = min([tr.stats.starttime for tr in st]).date
    endtdate = max([tr.stats.endtime for tr in st]).date

    delta = timedelta(days=1)

    while start_date <= end_date:
        print(start_date.strftime("%Y-%m-%d"))
        year = start_date.year
        doy = start_date.strftime("%j")
        
        _dir = os.path.join(mseed_day_dir, yr, doy)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        ms_start = UTCDateTime("%s:%sT00:00:00" % (yr, doy))
        _st = st.cut(starttime=ms_start, endtime=ms_start+86400)
        _st = _st._groupby('{network}.{station}')

        for key in st.keys():
            f_mseed = key + '.mseed'
            print(f_mseed)
            _st.write(os.path.join(_dir, f_mseed))

        start_date+=delta
            

    

def gen_mseed_fname(mseed_dir, channel, savedir):
    
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    with open(os.path.join(savedir, "fname.csv"), "w+") as fp,\
         open(os.path.join(savedir, "f_sampling_rate.csv"), "w+") as fs, \
         open(os.path.join(savedir, "f_starttime.csv"), "w+") as f_starttime:
        fp.write("fname,E,N,Z\n")

        for chan in channel:
            st = read(os.path.join(mseed_dir, ('*' + chan)))
            st = st.select(channel=chan)
            st = st._groupby('{network}.{station}')

            for key in st.keys():
                if len(st[key]) == 3:
                    print(key)
                    mseed_file = key+".mseed"
                    st[key].write(os.path.join(savedir, mseed_file))
                    fs.write('{}.{},{}\n'.format(st[key][0].stats.network,\
                                               st[key][0].stats.station,\
                                               st[key][0].stats.sampling_rate))
                    if chan[0] == "H":
                        fp.write(mseed_file+",HHE,HHN,HHZ\n")
                    else:
                        fp.write(mseed_file+",BHE,BHN,BHZ\n")

    print("Output: "+savedir)


def gen_npz_intput(mseed_dir, channel, savedir):
    
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    starttime = []
    with open(os.path.join(savedir, "fname.csv"), "w+") as fp,\
         open(os.path.join(savedir, "f_sampling_rate.csv"), "w+") as fs, \
         open(os.path.join(savedir, "f_starttime.csv"), "w+") as f_starttime:
        fp.write("fname\n")

        for chan in channel:
            st = read(os.path.join(mseed_dir, ('*' + chan)))
            st = st.select(channel=chan)
            st = st._groupby('{network}.{station}')

            for key in st.keys():
                if len(st[key]) == 3:
                    print(key)
                    fs.write('{}.{},{}\n'.format(st[key][0].stats.network,\
                                               st[key][0].stats.station,\
                                               st[key][0].stats.sampling_rate))
                    _starttime = _create_npz_input(st[key],savedir,fp)               
                    starttime.append(_starttime)
                elif len(st[key]) == 1:
                    print(key, '1 component only')
                    fs.write('{}.{},{}\n'.format(st[key][0].stats.network,\
                                               st[key][0].stats.station,\
                                               st[key][0].stats.sampling_rate))
                    _starttime = _create_npz_input(st[key],savedir,fp, single_component=True)
                    starttime.append(_starttime)

        for _list in starttime:
            f_starttime.write("{}, {}, {}\n".format(*_list))

    print("Output: "+savedir)

    return starttime


def _create_npz_input(st, savedir, fcsv, single_component=False, \
                      trim_missingdata=True):
    """
    create npz file per station
        - mseed file is processed twice with 50% overlap to avoid phases being 
        cut in the middle
        - single-component station would duplicate to 3 channels
        - Phasenet would demean and normalize before feeding in the data to NN

    savedir: directory to save phasenet inputs
    """

    starttime = st[0].stats.starttime
    net = st[0].stats.network
    sta = st[0].stats.station

    if single_component:
        st_e = st
        st_n = st
        st_z = st

    else:
        st_e = st.select(channel="??E")
        st_n = st.select(channel="??N")
        st_z = st.select(channel="??Z")

    if trim_missingdata:
        st_e.merge(fill_value=MAX_INT32)
        st_n.merge(fill_value=MAX_INT32)
        st_z.merge(fill_value=MAX_INT32)

    else:
        st_e.merge(fill_value=0)
        st_n.merge(fill_value=0)
        st_z.merge(fill_value=0)

    #st = st.trim(min([tr.stats.starttime for tr in st]),
    #             max([tr.stats.endtime for tr in st]),
    #             pad=True, fill_value=0)

    data_e = st_e[0].data
    data_n = st_n[0].data
    data_z = st_z[0].data


    startidx = 0
    lastidx = min(len(data_e), len(data_n), len(data_z))
    cnt = 0

    batch = np.empty([BATCH_SIZE, 3])

    while startidx < lastidx - BATCH_SIZE:

      batch[:, 0] = np.array(data_e[startidx:startidx+BATCH_SIZE], dtype=np.float)
      batch[:, 1] = np.array(data_n[startidx:startidx+BATCH_SIZE], dtype=np.float)
      batch[:, 2] = np.array(data_z[startidx:startidx+BATCH_SIZE], dtype=np.float)

      if MAX_INT32 in batch:
        print("batch contains a time gap - skipping")
        startidx = startidx + STRIDE
        continue

      np.savez("{}/{}.{}.{}_{}".format(savedir, net,sta, 'npz',startidx), data=batch)
      fcsv.write("{}.{}.{}_{}.npz\n".format(net, sta, 'npz',startidx))

      startidx = startidx + STRIDE
      cnt = cnt + 1

    print("number of batches written:", cnt)
    return [net, sta, starttime]


def run_phasenet(inputdir, outputdir):
    """
    calls run.py and runs PhaseNet
        - Batch size: For large dataset and GPU usage, larger batch size can
          accelerate the prediction
        - Currently, I have found a way to use GPU backend in tensorflow in Mac
          Please contact me if there is a way 
          (Keras can modified to use GPU backend, but not tf)
        - Plot results, save results slow the processes
    """
    import subprocess

    p = subprocess.Popen([
        "python",
        "run.py",
        "--mode=pred",
        "--batch_size=1",
        "--model_dir=model/190703-214543",
        "--data_dir={}".format(inputdir),
        "--data_list={}/fname.csv".format(inputdir),
        "--output_dir={}".format(outputdir)
        #"--plot_figure",
        #"--save_result"
        ], cwd=".")
    p.wait()

    return


def phase_out_2_real(pick_csv_dir, real_dir, npz_dir):
    # Miniseed directory is needed to obtain the sampling rate
    # Read sampling rate

    # Clean npz files
    os.system("find ./" + npz_dir + ' -maxdepth 1 -name "*.npz" -print0| xargs -0 rm')

    print('Cleaned %s .npz files' % npz_dir)
    sampling_rate = np.loadtxt(os.path.join(npz_dir, "f_sampling_rate.csv"),\
    delimiter=',', dtype=str)

    if not os.path.exists(real_dir):
        os.makedirs(real_dir)

    # Read picks
    pd_picks = pd.read_csv(os.path.join(pick_csv_dir, 'picks.csv'))
    pd_p_picks = pd_picks[pd_picks.itp != '[]']
    pd_s_picks = pd_picks[pd_picks.its != '[]']
    p_picks = []
    s_picks = []

    for index, row in pd_p_picks.iterrows():
        fname = row['fname'].split('.')
        net = fname[0]
        sta = fname[1]
        starttime = float(fname[2].split('_')[1])
        itp = row['itp'][1:-1].split()
        tp_prob = row['tp_prob'][1:-1].split()
        for i in range(len(itp)):
            p_picks.append([net, sta, starttime+float(itp[i]), float(tp_prob[i])])
    p_picks = np.array(p_picks)
    
    for index, row in pd_s_picks.iterrows():
        fname = row['fname'].split('.')
        net = fname[0]
        sta = fname[1]
        starttime = float(fname[2].split('_')[1])
        its = row['its'][1:-1].split()
        ts_prob = row['ts_prob'][1:-1].split()
        for i in range(len(its)):
            s_picks.append([net, sta, starttime+float(its[i]), float(ts_prob[i])])
    s_picks = np.array(s_picks)

    # GENERATE REAL INPUT PHASE FILE
    for net in np.unique(p_picks[:,0]):
        _p_picks = p_picks[p_picks[:,0]==net, :]
        _s_picks = s_picks[s_picks[:,0]==net, :]
        for chan in np.unique(_p_picks[:,1]):
            __p_picks = _p_picks[_p_picks[:,1]==chan, :]
            __s_picks = _s_picks[_s_picks[:,1]==chan, :]
            tr_sr = float(sampling_rate[sampling_rate[:,0]==str(net+'.'+chan), 1][-1])
            f_name_P = '{}.{}.P.txt'.format(net, chan)
            f_name_S = '{}.{}.S.txt'.format(net, chan)
            print('No. of P and S picks: %d, %d on %s.%s' % (__p_picks.shape[0],\
                                                             __s_picks.shape[0],net, chan))
            with open(os.path.join(real_dir, f_name_P), 'w+') as P_phase_file, \
                 open(os.path.join(real_dir, f_name_S), 'w+') as S_phase_file:
                for i in range(len(__p_picks[:,1])):
                    P_phase_file.write("%.2f %.2f 0.0 \n" % (float(__p_picks[i,2])/tr_sr, float(__p_picks[i,3])))
                for i in range(len(__s_picks[:,1])):
                    S_phase_file.write("%.2f %.2f 0.0 \n" % (float(__s_picks[i,2])/tr_sr, float(__s_picks[i,3])))

    print(real_dir)
    

if __name__ == "__main__":
    
    for i in range(29,38):
        doy = str(i).zfill(3)
        print('Processing day: ' + doy)

        # Channel to be processed
        channel = ['HH*','BH*']
        
        # directory configuration
        mseed_dir = '/Volumes/JW harddisk/Seismology/Data/miniseed/hkss1_rtlocal/2020/' + doy
        # Directory for temporary npz files
        savedir_npz = "dataset/HK_mseed/2020/" +doy + "/npz"
        # Output dir for phase net
        output_dir = 'output/hk2/' + doy
        # Input dir for REAL earthquake association
        real_dir = 'output/real/' + doy

        # Start Running full process
        #gen_npz_intput(mseed_dir, channel, savedir_npz)
        run_phasenet(savedir_npz, output_dir) 
        phase_out_2_real(output_dir, real_dir, savedir_npz)
