from collections import deque
import numpy as np


class WindowBuffer:
    def __init__(self,channels=4,sample_rate_hz=2000,window_s=.2,stride_s=.05):
        self.channels=channels; self.window=int(round(sample_rate_hz*window_s)); self.stride=int(round(sample_rate_hz*stride_s))
        self.data=deque(maxlen=self.window); self.since=0
    def append(self,sample):
        self.data.append(np.asarray(sample,float).reshape(self.channels)); self.since+=1
        if len(self.data)==self.window and self.since>=self.stride:
            self.since=0; return np.stack(self.data,axis=1)
        return None
