from __future__ import annotations
from pathlib import Path
import json, uuid
import numpy as np


class SessionLogger:
    def __init__(self,path,metadata=None): self.path=Path(path); self.records=[]; self.metadata={"session_id":str(uuid.uuid4()),**(metadata or {})}
    @staticmethod
    def _jsonable(value):
        if isinstance(value,np.ndarray): return value.tolist()
        if isinstance(value,np.generic): return value.item()
        if isinstance(value,dict): return {k:SessionLogger._jsonable(v) for k,v in value.items()}
        return value
    def append(self,**record):
        self.records.append({k:(json.dumps(self._jsonable(v)) if isinstance(v,dict) else v) for k,v in record.items()})
    def close(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        keys=set().union(*(r.keys() for r in self.records)) if self.records else set()
        arrays={k:np.asarray([r.get(k) for r in self.records]) for k in keys}
        arrays["metadata_json"]=np.array(json.dumps(self.metadata))
        np.savez_compressed(self.path,**arrays)
