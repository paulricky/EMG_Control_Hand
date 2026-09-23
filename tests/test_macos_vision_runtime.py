import os
import queue
from types import SimpleNamespace

import numpy as np

from src.app import macos_launcher
from src.app.camera_preview import CameraPreview
from src.app.sources import VisionSource
from src.backends.aero_coupling import joints_to_mujoco_ctrl
from src.backends.mujoco_aero import MujocoAeroBackend
from src.control.command_mapper import CommandMapper
from src.models.hand_state import HumanHandState


def test_macos_launcher_decision_and_marker():
    assert macos_launcher.should_relaunch(True, "darwin", {})
    assert not macos_launcher.should_relaunch(True, "darwin", {macos_launcher.MJPYTHON_MARKER: "1"})
    assert not macos_launcher.should_relaunch(False, "darwin", {})


def test_macos_launcher_preserves_arguments(monkeypatch, tmp_path):
    mjpython=tmp_path/"mjpython"; mjpython.touch(); mjpython.chmod(0o755)
    captured={}
    monkeypatch.setattr(macos_launcher,"should_relaunch",lambda *_: True)
    monkeypatch.setattr(macos_launcher,"find_mjpython",lambda: str(mjpython))
    monkeypatch.setattr(os,"execve",lambda exe,args,env: captured.update(exe=exe,args=args,env=env))
    macos_launcher.ensure_mjpython(True,["--source","vision","--show-mujoco"],module="src.app.runtime")
    assert captured["args"]==[str(mjpython),"-m","src.app.runtime","--source","vision","--show-mujoco"]
    assert captured["env"][macos_launcher.MJPYTHON_MARKER]=="1"


class _FakeProcess:
    def is_alive(self): return True


def test_camera_preview_queue_is_latest_only():
    preview=CameraPreview.__new__(CameraPreview); preview.queue=queue.Queue(maxsize=1); preview.process=_FakeProcess()
    for value in range(20): assert preview.publish(value)
    assert preview.queue.qsize()==1 and preview.queue.get_nowait()==19


class _Calibration:
    def normalize(self,angles): return np.linspace(0,1,16)


class _Cap:
    def read(self): return True,np.zeros((20,20,3),dtype=np.uint8)


class _Hands:
    def process(self,frame):
        point=lambda i: SimpleNamespace(x=(i%5)/5,y=(i//5)/5,z=.01*i)
        landmarks=SimpleNamespace(landmark=[point(i) for i in range(21)])
        classification=SimpleNamespace(label="Right",score=.9)
        return SimpleNamespace(multi_hand_landmarks=[landmarks],multi_hand_world_landmarks=[landmarks],multi_handedness=[SimpleNamespace(classification=[classification])])


class _CV:
    COLOR_BGR2RGB=1
    @staticmethod
    def flip(frame,axis): return frame
    @staticmethod
    def cvtColor(frame,kind): return frame


class _DeadPreview:
    shutdown_requested=False
    def publish(self,frame): return False
    def close(self): self.closed=True


def test_preview_failure_does_not_prevent_vision_state():
    source=VisionSource.__new__(VisionSource); source.cv2=_CV(); source.cap=_Cap(); source.hands=_Hands(); source.calibration=_Calibration()
    source.frame_number=0; source.last_observation=None; source.preview=_DeadPreview(); source.show=True; source._preview_warned=False
    source._annotate=lambda *args: np.zeros((2,2,3),dtype=np.uint8)
    state=source.update()
    assert isinstance(state,HumanHandState) and source.preview is None


class _CaptureBackend:
    def __init__(self): self.commands=[]; self.data=SimpleNamespace(ctrl=np.zeros(7))
    def send(self,command):
        self.commands.append(command); self.data.ctrl[:]=joints_to_mujoco_ctrl(command.joint_angles_deg)


def test_valid_vision_state_reaches_backend_and_changes_ctrl():
    mapper=CommandMapper([1/3]*3,[.5,.5]); backend=_CaptureBackend()
    make=lambda q: HumanHandState(1,np.zeros(16),np.asarray(q,float),np.eye(3),np.array([0,0,0,1]),1,"Right")
    backend.send(mapper.map(make(np.zeros(7)),"vision")); before=backend.data.ctrl.copy()
    backend.send(mapper.map(make(np.ones(7)),"vision"))
    assert len(backend.commands)==2 and not np.allclose(before,backend.data.ctrl)


def test_closed_mujoco_viewer_requests_clean_exit():
    backend=MujocoAeroBackend("scene_right.xml",show=True)
    backend.viewer=SimpleNamespace(is_running=lambda: False,sync=lambda: (_ for _ in ()).throw(AssertionError()))
    assert backend.render() is False
