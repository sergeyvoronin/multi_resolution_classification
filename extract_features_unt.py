#!/usr/bin/python3

# Here we use features extracted via librosa
# and some additional features with aubio and tsfresh library

import sys, getopt, os
import numpy as np
import librosa
import librosa.display
from sklearn import preprocessing as prep
import numpy.linalg as la
import pywt
import tsfresh

def main(argv):
    inputfile = sys.argv[1]
    cnum = sys.argv[2]
    outputfile = sys.argv[3]
    print 'Input file is: ', inputfile
    print 'Class num is: ', cnum
    print 'Output file is: ', outputfile

    
    # load signal
    y, sr = librosa.load(inputfile) 

    fp = open(outputfile,"a") 

    # print header
    #fp.write("label,  m1m,  m1s,  m2m,  m2s,  m3m,  m3s,  m4m,  m4s,  m5m,  m5s,  m6m,  m6s,  m7m,  m7s,  centm,  cents,  rollm,  rolls,  tonnm,  tonns,  zcrm,  zcrs,  rmsem,  rmses\n"); 
#label,  m1m,  m1s,  m2m,  m2s,  m3m,  m3s,  m4m,  m4s,  m5m,  m5s,  m6m,  m6s,  m7m,  m7s,  m8m,  m8s,  m9m,  m9s,  m10m,  m10s,  m11m,  m11s,  m12m,  m12s,  centm,  cents,  rollm,  rolls,  tonnm,  tonns,  zcrm,  zcrs,  rmsem,  rmses,  tempom,  onsetm,  pitchm,  wavea,  waved
    # print class name 
    fp.write("class_%s,  " % (cnum));

    # mfcc
    print("mfcc..\n");
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = 14)
    for num in range(2,14):
        print(num)
        mfcci = mfcc[num];
        mfcci = mfcci.reshape(-1,1);
        #mfcci = prep.normalize(mfcci);
        mfcci_mean = np.mean(mfcci);
        mfcci_std = np.std(mfcci);
        fp.write("%5.4e,  %5.4e,  " % (mfcci_mean, mfcci_std)) 


    # spectral centroid
    print("spectral centroid..\n");
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    #cent = prep.normalize(cent);
    cent_mean = np.mean(cent);
    cent_std = np.std(cent);
    fp.write("%5.4e,  %5.4e,  " % (cent_mean, cent_std)) 


    # spectral rolloff
    print("spectral rolloff..\n");
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    #rolloff = prep.normalize(rolloff);
    rolloff_mean = np.mean(rolloff);
    rolloff_std = np.std(rolloff);
    fp.write("%5.4e,  %5.4e,  " % (rolloff_mean, rolloff_std)) 

    
    # tonnetz
    print("tonnetz..\n");
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    #tonnetz = prep.normalize(tonnetz);
    tonnetz_mean = np.mean(tonnetz);
    tonnetz_std = np.std(tonnetz);
    fp.write("%5.4e,  %5.4e,  " % (tonnetz_mean, tonnetz_std)) 

    # zero crossing rate
    print("zcr..\n");
    zcr = librosa.feature.zero_crossing_rate(y)
    #zcr = prep.normalize(zcr);
    zcr_mean = np.mean(zcr);
    zcr_std = np.std(zcr);
    fp.write("%5.4e,  %5.4e,  " % (zcr_mean, zcr_std)) 

    # rmse
    print("rmse..\n");
    rmse = librosa.feature.rmse(y=y)
    #rmse = prep.normalize(rmse);
    rmse_mean = np.mean(rmse);
    rmse_std = np.std(rmse);
    fp.write("%5.4e,  %5.4e,  " % (rmse_mean, rmse_std)) 

    # tempo
    print("tempo..\n");
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    fp.write("%5.4e,  " % (tempo)) 

	# onset
    print("aubio onset..\n");
    cmd = "./run_aubio_onset.pl" + " " + inputfile;
    os.system(cmd);
    with open("ONSET") as faub:
		onsets = faub.readlines();	
    onsets = [x.strip() for x in onsets];
    onsets = [float(x) for x in onsets];
    onsets = np.asarray(onsets).reshape(-1,1);
    onsets = prep.normalize(onsets);
    onset_mean = (np.max(onsets) - np.min(onsets))/len(onsets);
    fp.write("%5.4e,  " % (onset_mean)) 

    # pitch
    print("aubio pitch..\n");
    cmd = "./run_aubio_pitch.pl" + " " + inputfile;
    os.system(cmd);
    with open("PITCH") as faub:
        pitches = faub.readlines();	
    pitches = [x.strip() for x in pitches];
    streak = 0;
    max_streak = 0;
    for s in pitches:
		s2 = s.split(" ")
		pitch = float(s2[1]);
		if pitch > 4500:
			streak = streak + 1;
			if streak > max_streak:
				max_streak = streak;
		else:
			streak = 0;
		
    max_streak = float(max_streak);
    fp.write("%5.4e,  " % (max_streak)) 

    # wavelet
    print("wavelet stats (smooth, sharp)..\n");
    (a1,d1) = pywt.dwt(y,'db8');
    fp.write("%5.4e,  " % (la.norm(a1))) 
    fp.write("%5.4e,  " % (la.norm(d1))) 

    (a2,d2) = pywt.dwt(y,'haar');
    fp.write("%5.4e,  " % (la.norm(a2))) 
    fp.write("%5.4e,  " % (la.norm(d2))) 


    print("wavelet reconstruction stats (approx, detail)..\n");
    transform_name = 'db4';
    coeffs = pywt.wavedec(y, transform_name, level=4);
    coeffs[1]=np.zeros(len(coeffs[1]))
    coeffs[2]=np.zeros(len(coeffs[2]))
    coeffs[3]=np.zeros(len(coeffs[3]))
    coeffs[4]=np.zeros(len(coeffs[4]))
    yr = pywt.waverec(coeffs, transform_name);
    nval = 0;
    for l in range(1,len(y)): 
        nval = nval + np.abs(y[l] - yr[l])*np.abs(y[l] - yr[l]);
    nval = np.sqrt(nval);
    p1 = nval/la.norm(y)
    fp.write("%5.4e,  " % p1) 

    coeffs = pywt.wavedec(y, transform_name, level=4);
    coeffs[0]=np.zeros(len(coeffs[0]))
    yr = pywt.waverec(coeffs, transform_name);
    nval = 0;
    for l in range(1,len(y)): 
        nval = nval + np.abs(y[l] - yr[l])*np.abs(y[l] - yr[l]);
    nval = np.sqrt(nval);
    p1 = nval/la.norm(y)
    fp.write("%5.4e,  " % p1) 


    transform_name = 'haar';
    coeffs = pywt.wavedec(y, transform_name, level=4);
    coeffs[1]=np.zeros(len(coeffs[1]))
    coeffs[2]=np.zeros(len(coeffs[2]))
    coeffs[3]=np.zeros(len(coeffs[3]))
    coeffs[4]=np.zeros(len(coeffs[4]))
    yr = pywt.waverec(coeffs, transform_name);
    nval = 0;
    for l in range(1,len(y)): 
        nval = nval + np.abs(y[l] - yr[l])*np.abs(y[l] - yr[l]);
    nval = np.sqrt(nval);
    p1 = nval/la.norm(y)
    fp.write("%5.4e,  " % p1) 

    coeffs = pywt.wavedec(y, transform_name, level=4);
    coeffs[0]=np.zeros(len(coeffs[0]))
    yr = pywt.waverec(coeffs, transform_name);
    nval = 0;
    for l in range(1,len(y)): 
        nval = nval + np.abs(y[l] - yr[l])*np.abs(y[l] - yr[l]);
    nval = np.sqrt(nval);
    p1 = nval/la.norm(y)
    fp.write("%5.4e,  " % p1) 

 

    # time series
    print("various time series stats..\n");

    msd = tsfresh.feature_extraction.feature_calculators.mean_second_derivative_central(y)
    fp.write("%5.4e,  " % (msd)) 

    mc = tsfresh.feature_extraction.feature_calculators.mean_change(y);
    fp.write("%5.4e,  " % (mc)) 

    sam = tsfresh.feature_extraction.feature_calculators.longest_strike_above_mean(y)
    fp.write("%5.4e,  " % (sam)) 

    sbm = tsfresh.feature_extraction.feature_calculators.longest_strike_below_mean(y)
    fp.write("%5.4e,  " % (sbm)) 


    kurtosis = tsfresh.feature_extraction.feature_calculators.kurtosis(y)
    fp.write("%5.4e,  " % (kurtosis)) 

    #meauto =  tsfresh.feature_extraction.feature_calculators.mean_autocorrelation(y)
    #fp.write("%5.4e,  " % (meauto)) 
    myparam = [{"f_agg": "mean"}];
    meauto = tsfresh.feature_extraction.feature_calculators.agg_autocorrelation(y,myparam);
    fp.write("%5.4e,  " % (meauto[0][1]))

    skewness = tsfresh.feature_extraction.feature_calculators.skewness(y)
    fp.write("%5.4e,  " % (skewness)) 

    timerev = tsfresh.feature_extraction.feature_calculators.time_reversal_asymmetry_statistic(y,10)
    fp.write("%5.4e\n" % (timerev)) 

    fp.close();


if __name__ == "__main__":
    main(sys.argv[1:])


