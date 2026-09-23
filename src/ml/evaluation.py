import numpy as np


def regression_metrics(y_true,y_pred):
    y=np.asarray(y_true,float); p=np.asarray(y_pred,float); e=p-y
    ss=np.sum((y-y.mean(axis=0))**2,axis=0)
    return {"mae":np.mean(np.abs(e),axis=0),"rmse":np.sqrt(np.mean(e*e,axis=0)),
            "r2":1-np.sum(e*e,axis=0)/np.maximum(ss,1e-12),
            "correlation":np.array([np.corrcoef(y[:,i],p[:,i])[0,1] for i in range(y.shape[1])])}


def estimate_delay_s(reference,response,sample_rate_hz):
    a=np.asarray(reference)-np.mean(reference); b=np.asarray(response)-np.mean(response)
    lag=np.argmax(np.correlate(b,a,mode="full"))-(len(a)-1); return lag/sample_rate_hz
