import numpy as np


def rotation_vector(r):
    r=np.asarray(r,float); angle=np.arccos(np.clip((np.trace(r)-1)/2,-1,1))
    if angle<1e-8: return np.zeros(3)
    return angle*np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])/(2*np.sin(angle))


class WristMapper:
    def __init__(self,neutral_rotation,axis_order=(0,1,2),signs=(1,1,1),offsets=(0,0,0),limits=((-3.14,3.14),)*3):
        self.neutral=np.asarray(neutral_rotation,float); self.order=axis_order; self.signs=np.asarray(signs); self.offsets=np.asarray(offsets); self.limits=np.asarray(limits)
    def map(self,current):
        v=rotation_vector(self.neutral.T@np.asarray(current,float)); q=v[list(self.order)]*self.signs+self.offsets
        return np.clip(q,self.limits[:,0],self.limits[:,1])
