from __future__ import annotations
import time
import numpy as np


def _torch():
    try: import torch; return torch
    except ImportError as exc: raise RuntimeError("PyTorch is required for the temporal EMG model") from exc


class TemporalCNN:
    """Small causal-capable 1-D CNN wrapper with seven sigmoid outputs."""
    def __new__(cls,channels=4,outputs=7):
        torch=_torch(); nn=torch.nn
        class Network(nn.Module):
            def __init__(self):
                super().__init__(); self.net=nn.Sequential(nn.Conv1d(channels,32,9,padding=4),nn.ReLU(),nn.MaxPool1d(2),
                    nn.Conv1d(32,64,7,padding=3),nn.ReLU(),nn.MaxPool1d(2),nn.Conv1d(64,64,5,padding=2),nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1)); self.head=nn.Linear(64,outputs)
            def forward(self,x): return torch.sigmoid(self.head(self.net(x).squeeze(-1)))
        return Network()


class TemporalEmgModel:
    def __init__(self,model): self.model=model
    def predict(self,windows):
        torch=_torch(); self.model.eval()
        with torch.no_grad(): return self.model(torch.as_tensor(windows,dtype=torch.float32)).cpu().numpy()
    @classmethod
    def load(cls,path):
        torch=_torch(); checkpoint=torch.load(path,map_location="cpu",weights_only=True); model=TemporalCNN(checkpoint["channels"])
        model.load_state_dict(checkpoint["state_dict"]); return cls(model)


def windows_from_session(path,window=400,stride=100):
    d=np.load(path,allow_pickle=False); x=np.asarray(d["processed_emg"],np.float32); y=np.asarray(d["compact01"],np.float32)
    if x.ndim!=2: raise ValueError(f"{path}: processed_emg must be samples x channels")
    return np.stack([x[i-window+1:i+1].T for i in range(window-1,len(x),stride)]),y[window-1::stride]


def train_temporal(train_sessions,val_sessions,epochs=20,batch_size=64,learning_rate=1e-3,lambda_vel=0.0,window=400,stride=100):
    torch=_torch(); train=[windows_from_session(p,window,stride) for p in train_sessions]; val=[windows_from_session(p,window,stride) for p in val_sessions]
    x=np.concatenate([a for a,_ in train]); y=np.concatenate([b for _,b in train]); model=TemporalCNN(x.shape[1]); opt=torch.optim.Adam(model.parameters(),lr=learning_rate)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=batch_size,shuffle=not bool(lambda_vel))
    for _ in range(epochs):
        model.train()
        for xb,yb in loader:
            pred=model(xb); loss=torch.nn.functional.smooth_l1_loss(pred,yb)
            if lambda_vel and len(pred)>1: loss=loss+lambda_vel*torch.mean((pred[1:]-pred[:-1])**2)
            opt.zero_grad(); loss.backward(); opt.step()
    return model,val


def evaluate_temporal(model,sessions,window=400,stride=100):
    from .evaluation import regression_metrics
    torch=_torch(); pairs=[windows_from_session(p,window,stride) for p in sessions]; x=np.concatenate([a for a,_ in pairs]); y=np.concatenate([b for _,b in pairs])
    model.eval(); start=time.perf_counter()
    with torch.no_grad(): pred=model(torch.from_numpy(x)).numpy()
    elapsed=(time.perf_counter()-start)/max(len(x),1); metrics=regression_metrics(y,pred); metrics["inference_latency_s"]=elapsed
    metrics["overall_mae"]=float(np.mean(metrics["mae"])); metrics["overall_rmse"]=float(np.mean(metrics["rmse"])); return metrics
