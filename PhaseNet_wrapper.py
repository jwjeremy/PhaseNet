import sys
import os
import numpy as np
from obspy import read, UTCDateTime

SAMPLING_RATE = 200
MAX_INT32 = 2147483647
BATCH_SIZE = 3000
STRIDE = 1500

def create_input(mseeddir, savedir):
  """
  mseeddir: directory for the miniSEED file
  savedir: directory to save phasenet inputs
  """

  if not os.path.exists(savedir):
    os.makedirs(savedir)
  
  st = read(mseeddir)
  starttime = st[0].stats.starttime

  st_e = st.select(channel="HHE")
  st_n = st.select(channel="HHN")
  st_z = st.select(channel="HHZ")

  st_e.merge(fill_value=MAX_INT32)
  st_n.merge(fill_value=MAX_INT32)
  st_z.merge(fill_value=MAX_INT32)
  
  data_e = st_e[0].data
  data_n = st_n[0].data
  data_z = st_z[0].data

  with open(savedir + "/fname.csv", "w") as fname:
    fname.write("fname\n")

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

      np.savez("{}/{}".format(savedir, startidx), data=batch)
      fname.write("{}.npz\n".format(startidx))

      startidx = startidx + STRIDE
      cnt = cnt + 1
  
  print("number of batches written:", cnt)
  return starttime

def run_phasenet(inputdir, outputdir):
  """
  calls run.py and runs PhaseNet
  """
  import subprocess

  p = subprocess.Popen([
    "python",
    "run.py",
    "--mode=pred",
    #"--ckdir=PhaseNet/model/190227-104428",
    "--model_dir=model/190703-214543",
    "--data_dir={}".format(inputdir),
    "--data_list={}/fname.csv".format(inputdir),
    "--output_dir={}".format(outputdir)
    #"--plot_figure",
    #"--save_result"
  ], cwd=".")
  p.wait()

  return

def _parse_phasenet_output(fpicks_npz, phaseidx):
  """
  reads in .npz file and returns lists of start indices with picks, indices of picks, and probabilies of the picks
  """
  fname = fpicks_npz['fname']
  picks = fpicks_npz['picks']
  startidxs = [int((fname[i].decode().split('.')[0]).split('/')[-1]) for i in range(len(fname))]
  startidxs, picks = map(list, zip(*sorted(zip(startidxs, picks), reverse = False)))  # sort by index
  startidxs_with_picks = []
  idxs = []
  probs = []
  absidx_old = -1000
  for i in range(len(startidxs)):
    startidx = startidxs[i]
    if len(picks[i][phaseidx][0]) > 0:
      for j in range(len(picks[i][phaseidx][0])):
        idx = int(picks[i][phaseidx][0][j])
        absidx = startidx + idx
        prob = float(picks[i][phaseidx][1][j])
        if absidx - absidx_old > SAMPLING_RATE:
          startidxs_with_picks.append(startidx)
          idxs.append(idx)
          probs.append(prob)
          absidx_old = absidx
  return startidxs_with_picks, idxs, probs

def parse_output(picksdir, starttime):
  with np.load("{}/picks.npz".format(picksdir), allow_pickle=True) as fpicks_npz:
    startidxs_p, pickidxs_p, probs_p = _parse_phasenet_output(fpicks_npz, 0)
    startidxs_s, pickidxs_s, probs_s = _parse_phasenet_output(fpicks_npz, 1)
  with open("picks_p.txt", "w") as fpicks:
    for i in range(len(startidxs_p)):
      fpicks.write("{} {}\n".format(starttime + (startidxs_p[i] + pickidxs_p[i]) / SAMPLING_RATE, probs_p[i]))
  with open("picks_s.txt", "w") as fpicks:
    for i in range(len(startidxs_s)):
      fpicks.write("{} {}\n".format(starttime + (startidxs_s[i] + pickidxs_s[i]) / SAMPLING_RATE, probs_s[i]))


# --- main ---- #
if __name__ == "__main__":
  
  #mseeddir = "fdsnws-dataselect_2019-07-10T19_01_56Z.mseed"
  #mseeddir = "fdsnws-dataselect_2019-07-23T22_37_29Z_B921.mseed"
  #mseeddir = "AXCC1_20150101.mseed" 
  mseeddir = "dataset/HK_mseed/2020_001/HK.HKCC2.mseed" 

  # directory to save phasenet input files
  savedir = "dataset/HK_mseed/2020_001/npz"

  # directory to save phasenet output files
  outputdir = "output/hk3"

  starttime = create_input(mseeddir, savedir)

  # runs phasenet through the subprocess module
 # run_phasenet(savedir, outputdir)

  # parses phasenet output and saves into text files (P and S separately)
  parse_output(outputdir, starttime)

