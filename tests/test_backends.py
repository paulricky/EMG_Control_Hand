import numpy as np
from src.backends.real_aero import RealAeroBackend
from src.backends.dual_aero import DualAeroBackend
from src.models.aero_command import AeroCommand


class FakeHand:
    joint_names=range(16)
    def __init__(self,**kwargs): self.sent=[]
    def get_actuations(self): return [0]*7
    def get_actuator_currents(self): return [0]*7
    def get_actuator_speeds(self): return [0]*7
    def get_actuator_temperatures(self): return [20]*7
    def set_joint_positions(self,x): self.sent.append(x)
    def close(self): pass
    def set_torque(self,index,value): self.torque=(index,value)


def command(): return AeroCommand(1,np.arange(16),np.zeros(7),True,"test")


def test_real_backend_dry_run_never_constructs_sdk():
    calls=[]; b=RealAeroBackend(enable_hardware=False,sdk_factory=lambda **k:calls.append(k)); b.connect(); b.send(command())
    assert calls==[] and b.read_state()["dry_run"]


def test_real_backend_uses_full_joint_sdk_call():
    b=RealAeroBackend(enable_hardware=True,sdk_factory=FakeHand).connect(); b.send(command())
    assert len(b.hand.sent[0])==16


def test_real_backend_executes_configured_fault_action():
    b=RealAeroBackend(enable_hardware=True,sdk_factory=FakeHand).connect(); c=command(); b.send(c); b.handle_fault("hold",c)
    assert len(b.hand.sent)==2
    b.handle_fault("controlled_open",c); assert b.hand.sent[-1]==[0.0]*16
    b.handle_fault("disable",c); assert b.hand.torque==(6,0)


class Capture:
    def __init__(self): self.sent=[]
    def connect(self): return self
    def send(self,c): self.sent.append(c)
    def read_state(self): return {}
    def shutdown(self): pass
    def step(self): pass
    def render(self): pass
    def reset(self): pass


def test_dual_receives_identical_canonical_object():
    a,b=Capture(),Capture(); d=DualAeroBackend(a,b); c=command(); d.send(c)
    assert a.sent[0] is c and b.sent[0] is c
