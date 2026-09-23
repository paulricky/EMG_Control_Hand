from collections import deque
import numpy as np


class ClockAligner:
    def __init__(self,max_points=1000): self.points=deque(maxlen=max_points)
    def update(self,mcu_us,host_s): self.points.append((mcu_us*1e-6,host_s))
    def to_host(self,mcu_us):
        if not self.points: raise RuntimeError("clock aligner has no observations")
        x=np.array([p[0] for p in self.points]); y=np.array([p[1] for p in self.points])
        if len(x)<2 or np.ptp(x)<1e-9: return mcu_us*1e-6+float(np.mean(y-x))
        slope,offset=np.polyfit(x,y,1); return float(slope*mcu_us*1e-6+offset)


def label_timestamp(start_s,end_s,reference="center"):
    if reference=="center": return (start_s+end_s)/2
    if reference=="end": return end_s
    raise ValueError("reference must be center or end")
