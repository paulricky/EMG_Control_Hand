import numpy as np


def emg_features(window,threshold=.01):
    x=np.asarray(window,float); d=np.diff(x,axis=-1)
    mav=np.mean(np.abs(x),axis=-1); rms=np.sqrt(np.mean(x*x,axis=-1)); var=np.var(x,axis=-1); wl=np.sum(np.abs(d),axis=-1)
    zc=np.sum((x[:,:-1]*x[:,1:]<0)&(np.abs(d)>threshold),axis=-1)
    ssc=np.sum((d[:,:-1]*d[:,1:]<0)&(np.abs(np.diff(d,axis=-1))>threshold),axis=-1)
    return np.concatenate((mav,rms,wl,var,zc,ssc))
