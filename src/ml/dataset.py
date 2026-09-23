from pathlib import Path
import numpy as np


def load_sessions(paths):
    return [np.load(Path(p),allow_pickle=False) for p in paths]


def split_by_session(paths,seed=0,ratios=(.7,.15,.15)):
    paths=list(map(str,paths)); rng=np.random.default_rng(seed); rng.shuffle(paths)
    n=len(paths); a=int(n*ratios[0]); b=int(n*(ratios[0]+ratios[1]))
    return paths[:a],paths[a:b],paths[b:]
