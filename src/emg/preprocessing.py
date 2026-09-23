from __future__ import annotations
import numpy as np


class CausalPreprocessor:
    """Stateful sample-by-sample EMG conditioning; never uses future samples."""
    def __init__(self,channels,baseline=None,mean=None,std=None,sample_rate_hz=2000,
                 notch_hz=None,notch_q=30.0,bandpass_hz=None):
        self.channels=int(channels); self.baseline=np.zeros(channels) if baseline is None else np.asarray(baseline,float)
        self.mean=np.zeros(channels) if mean is None else np.asarray(mean,float)
        self.std=np.ones(channels) if std is None else np.maximum(np.asarray(std,float),1e-6)
        self.sample_rate_hz=sample_rate_hz; self.notch_hz=notch_hz; self.notch_q=notch_q; self.bandpass_hz=bandpass_hz
        self.sos=[]; self.states=[]
        try:
            from scipy.signal import butter, iirnotch, tf2sos
            if notch_hz:
                b,a=iirnotch(float(notch_hz),float(notch_q),fs=float(sample_rate_hz)); self.sos.append(tf2sos(b,a))
            if bandpass_hz:
                self.sos.append(butter(4,bandpass_hz,btype="bandpass",fs=float(sample_rate_hz),output="sos"))
        except ImportError:
            if notch_hz or bandpass_hz: raise RuntimeError("scipy is required for EMG filtering")
        for sos in self.sos: self.states.append(np.zeros((sos.shape[0],2,self.channels),float))
    def process_sample(self,sample):
        from scipy.signal import sosfilt
        x=np.asarray(sample,float).reshape(self.channels)-self.baseline
        for i,sos in enumerate(self.sos):
            y,z=sosfilt(sos,x[None,:],axis=0,zi=self.states[i]); x=y[0]; self.states[i]=z
        return (x-self.mean)/self.std
    def transform(self,window):
        clone=CausalPreprocessor(self.channels,self.baseline,self.mean,self.std,self.sample_rate_hz,self.notch_hz,self.notch_q,self.bandpass_hz)
        return np.stack([clone.process_sample(s) for s in np.asarray(window).T],axis=1)
    def validate(self,window,adc_max=4095):
        x=np.asarray(window,float)
        return bool(np.isfinite(x).all() and np.all(np.var(x,axis=1)>1e-8) and np.mean((x<=0)|(x>=adc_max))<.05)
