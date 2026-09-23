import struct
import numpy as np
import pytest

from src.perception.hand_geometry import internal_angle, flexion_angle, palm_frame
from src.perception.hand_calibration import HandCalibration
from src.control.command_mapper import compact_to_full_deg, full_to_compact01
from src.control.filters import ExponentialFilter, SlewRateLimiter
from src.control.safety import SafetySupervisor, SafetyError
from src.models.aero_command import AeroCommand
from src.backends.aero_coupling import joints_to_mujoco_ctrl, MUJOCO_CTRL_LOW, MUJOCO_CTRL_HIGH
from src.emg.serial_reader import HEADER, EmgPacketParser, crc16_ccitt
from src.emg.window_buffer import WindowBuffer
from src.logging.synchronization import ClockAligner, label_timestamp
from src.app.sources import landmark_arrays
from src.control.filters import CommandConditioner
from src.emg.preprocessing import CausalPreprocessor


def test_angles():
    assert internal_angle([1,0,0],[0,0,0],[-1,0,0]) == pytest.approx(np.pi)
    assert flexion_angle([1,0,0],[0,0,0],[-1,0,0]) == pytest.approx(0)


def landmarks():
    p=np.zeros((21,3)); p[0]=[0,0,0]; p[5]=[1,1,0]; p[9]=[0,2,0]; p[17]=[-1,1,0]; return p


@pytest.mark.parametrize("rotation",[
    np.eye(3),np.diag([-1,1,-1]),
    np.array([[1,0,0],[0,0,-1],[0,1,0]]),np.array([[1,0,0],[0,0,1],[0,-1,0]]),
    np.array([[0,0,1],[0,1,0],[-1,0,0]]),np.array([[0,0,-1],[0,1,0],[1,0,0]])])
def test_right_hand_palm_frame_all_reference_orientations(rotation):
    r=palm_frame(landmarks()@rotation.T)
    np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12); assert np.linalg.det(r)==pytest.approx(1)


def test_reversed_calibration_and_clamp():
    c=HandCalibration(np.ones(16),np.zeros(16)); np.testing.assert_allclose(c.normalize([.5]*16),.5)
    assert np.all(c.normalize([2]*16)==0)


def test_five_pose_calibration_uses_thumb_specific_poses():
    base=np.zeros(16); fist=np.ones(16)
    poses={"open_palm":[base],"closed_fist":[fist],"thumb_abduction":[np.r_[2,np.zeros(15)]],
           "thumb_opposition":[np.r_[0,3,4,5,np.zeros(12)]],"pinch":[np.r_[0,0,6,7,np.zeros(12)]]}
    c=HandCalibration.from_pose_samples(poses,"id"); assert c.closed_rad[0]==2 and c.closed_rad[1]==3
    assert c.closed_rad[2]==6 and c.closed_rad[3]==7 and c.hand=="Right" and c.version==2


def test_compact_round_trip():
    q=np.linspace(0,1,7); np.testing.assert_allclose(full_to_compact01(compact_to_full_deg(q)),q)


def test_time_filter_and_slew():
    f=ExponentialFilter(1); f.update([0],0); assert f.update([1],1)[0]==pytest.approx(1-np.exp(-1))
    s=SlewRateLimiter([.5]); s.update([0],0); assert s.update([1],1)[0]==pytest.approx(.5)


def test_safety_clamps_and_rejects_nan():
    s=SafetySupervisor(timeout_s=1); c=AeroCommand(1,np.ones(16)*999,np.ones(7)*2,True,"test")
    valid=s.validate(c,1.1); assert valid.joint_angles_deg[0]==100 and valid.compact01.max()==1
    with pytest.raises(ValueError): AeroCommand(1,[np.nan]*16,np.zeros(7),True,"test")


def test_official_mujoco_controls_are_bounded():
    for joints in (np.zeros(16),np.array([100,55]+[90]*14)):
        c=joints_to_mujoco_ctrl(joints); assert np.all(c>=MUJOCO_CTRL_LOW) and np.all(c<=MUJOCO_CTRL_HIGH)


def test_emg_packet_parser_and_crc_recovery():
    samples=np.array([1,2,3,4],dtype="<u2"); body=HEADER.pack(b"\xA5\x5A",1,4,7,123)+samples.tobytes(); frame=body+struct.pack("<H",crc16_ccitt(body))
    p=EmgPacketParser(4); assert p.feed(b"bad"+frame[:5])==[]; out=p.feed(frame[5:],4.0)
    assert len(out)==1 and out[0].sequence==7; np.testing.assert_array_equal(out[0].samples,samples)


def test_window_stride():
    b=WindowBuffer(2,10,.4,.2); outputs=[b.append([i,i]) for i in range(8)]
    assert sum(x is not None for x in outputs)==3 and outputs[-1].shape==(2,4)


def test_clock_alignment():
    c=ClockAligner(); c.update(0,10); c.update(1_000_000,11.001)
    assert c.to_host(500_000)==pytest.approx(10.5005); assert label_timestamp(2,4)==3 and label_timestamp(2,4,"end")==4


class _Point:
    def __init__(self,x,y,z): self.x=x; self.y=y; self.z=z
class _Landmarks:
    def __init__(self,value): self.landmark=[_Point(value,i,value+1) for i in range(21)]
class _Result: pass


def test_world_landmarks_are_geometry_source_and_image_is_preserved():
    r=_Result(); r.multi_hand_landmarks=[_Landmarks(1)]; r.multi_hand_world_landmarks=[_Landmarks(7)]
    image,geometry,used=landmark_arrays(r); assert used and image[0,0]==1 and geometry[0,0]==7


def test_landmark_fallback_is_documented_image_copy():
    r=_Result(); r.multi_hand_landmarks=[_Landmarks(2)]; r.multi_hand_world_landmarks=[]
    image,geometry,used=landmark_arrays(r); assert not used; np.testing.assert_array_equal(image,geometry); assert image is not geometry


def test_causal_preprocessor_has_no_future_response():
    a=CausalPreprocessor(1,sample_rate_hz=2000,notch_hz=60,bandpass_hz=(20,450)); b=CausalPreprocessor(1,sample_rate_hz=2000,notch_hz=60,bandpass_hz=(20,450))
    prefix=np.linspace(0,1,20); out_a=[a.process_sample([x])[0] for x in prefix]; out_b=[b.process_sample([x])[0] for x in prefix]
    b.process_sample([999])
    np.testing.assert_allclose(out_a,out_b)


def test_command_conditioner_operates_before_backend():
    c=AeroCommand(0,np.zeros(16),np.zeros(7),True,"test"); f=CommandConditioner(0,[.5]*7); f.update(c)
    out=f.update(AeroCommand(1,np.array([100,55]+[90]*14),np.ones(7),True,"test"))
    assert np.all(out.compact01<=.5+1e-12)
