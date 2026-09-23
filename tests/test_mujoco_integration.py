import numpy as np
import pytest

mujoco=pytest.importorskip("mujoco"); mm=pytest.importorskip("mujoco_menagerie")
from src.backends.aero_coupling import MUJOCO_ACTUATOR_NAMES
from src.backends.mujoco_aero import MujocoAeroBackend
from src.models.aero_command import AeroCommand


def _run(backend,joints,seconds=4):
    backend.reset(); backend.send(AeroCommand(0,joints,np.zeros(7),True,"test"))
    for _ in range(round(seconds/backend.model.opt.timestep)): backend.step()
    state=backend.read_state(); assert np.isfinite(state["joint_rad"]).all() and np.isfinite(state["tendon_length_m"]).all(); return state


def test_official_right_hand_executes_pose_sequence():
    path=mm.get("tetheria_aero_hand_open").path()/"scene_right.xml"; b=MujocoAeroBackend(path).connect()
    try:
        names=tuple(mujoco.mj_id2name(b.model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(b.model.nu))
        assert names==MUJOCO_ACTUATOR_NAMES; assert b.model.opt.timestep==pytest.approx(.01); assert b.model.ntendon==20
        np.testing.assert_allclose(b.model.actuator_ctrlrange,np.array([[.058520,.110387]]*4+[[-.1,1.75],[.026152,.038389],[.081568,.112138]]))
        opened=_run(b,np.zeros(16)); graded=[]
        for level in (.25,.5,.75,1): graded.append(_run(b,np.array([100*level,55*level]+[90*level]*14))["joint_rad"])
        assert graded[-1][0]>opened["joint_rad"][0]+1 and graded[-1][3]>opened["joint_rad"][3]+1
        for start,index in zip((4,7,10,13),(0,3,6,9)):
            joints=np.zeros(16); joints[start:start+3]=90; assert _run(b,joints)["joint_rad"][index]>1
        for joints,index in ((np.array([100]+[0]*15),12),(np.array([0,55]+[0]*14),13),(np.array([0,0,90,90]+[0]*12),14)):
            assert _run(b,joints)["joint_rad"][index]>.25
    finally: b.shutdown()
