import numpy as np
import pytest
torch=pytest.importorskip("torch")
from src.ml.temporal import TemporalCNN
from src.ml.dataset import split_by_session


def test_temporal_model_shape_and_bounds():
    model=TemporalCNN(4); y=model(torch.randn(3,4,400)); assert tuple(y.shape)==(3,7)
    assert torch.all(y>=0) and torch.all(y<=1)


def test_session_split_has_no_leakage():
    train,val,test=split_by_session([f"session-{i}" for i in range(10)],seed=2)
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test) and set(val).isdisjoint(test)
    assert set(train+val+test)=={f"session-{i}" for i in range(10)}
