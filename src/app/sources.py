from __future__ import annotations
from abc import ABC, abstractmethod
import time
import numpy as np
from src.models.hand_state import HumanHandState
from src.perception.hand_geometry import anatomical_angles, matrix_to_quaternion_xyzw, palm_frame
from src.perception.hand_calibration import HandCalibration

HAND_CONNECTIONS=((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20))


def landmark_arrays(result, index=0):
    """Return (image, geometry, used_world). World coordinates drive geometry."""
    image=np.array([[v.x,v.y,v.z] for v in result.multi_hand_landmarks[index].landmark],dtype=float)
    worlds=getattr(result,"multi_hand_world_landmarks",None)
    if worlds and len(worlds)>index:
        return image,np.array([[v.x,v.y,v.z] for v in worlds[index].landmark],dtype=float),True
    # Documented compatibility fallback for MediaPipe builds lacking world output.
    return image,image.copy(),False


class ControlSource(ABC):
    @abstractmethod
    def update(self): ...
    def close(self): pass


class VisionSource(ControlSource):
    def __init__(self,camera_id=0,min_confidence=.7,show=False,calibration=None):
        try: import cv2; import mediapipe as mp
        except ImportError as exc: raise RuntimeError("install opencv-python and mediapipe for vision input") from exc
        self.cv2=cv2; self.show=show; self.cap=cv2.VideoCapture(camera_id); self.frame_number=0; self.preview=None; self._preview_warned=False
        self.calibration=HandCalibration.load(calibration) if isinstance(calibration,(str,bytes)) else calibration
        if self.calibration is None: raise ValueError("VisionSource requires a right-hand calibration file")
        self.last_observation=None
        self.hands=mp.solutions.hands.Hands(max_num_hands=1,min_detection_confidence=min_confidence,min_tracking_confidence=min_confidence)
        if show:
            from .camera_preview import CameraPreview
            self.preview=CameraPreview()
    @property
    def shutdown_requested(self): return bool(self.preview and self.preview.shutdown_requested)
    def update(self):
        ok,frame=self.cap.read(); timestamp=time.monotonic(); self.frame_number+=1
        if not ok: return None
        frame=self.cv2.flip(frame,1)
        result=self.hands.process(self.cv2.cvtColor(frame,self.cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks or not result.multi_handedness:
            self._publish(frame,None,None,None,None,0.0,timestamp)
            return None
        cls=result.multi_handedness[0].classification[0]
        if cls.label != "Right":
            image_p=np.array([[v.x,v.y,v.z] for v in result.multi_hand_landmarks[0].landmark],dtype=float)
            self._publish(frame,image_p,None,None,None,float(cls.score),timestamp)
            return None
        image_p,p,used_world=landmark_arrays(result); r=palm_frame(p); angles=anatomical_angles(p); q16=self.calibration.normalize(angles)
        q=np.array([q16[0],q16[1],np.mean(q16[2:4]),*(np.mean(q16[s:s+3]) for s in (4,7,10,13))])
        self.last_observation={"camera_timestamp_s":timestamp,"camera_frame":self.frame_number,"image_landmarks":image_p,
            "world_landmarks":p,"used_world_landmarks":used_world,"raw_frame":frame}
        state=HumanHandState(timestamp,angles,np.clip(q,0,1),r,matrix_to_quaternion_xyzw(r),float(cls.score),"Right",q16)
        self._publish(frame,image_p,r,state.joint_normalized01,q16,float(cls.score),timestamp)
        return state
    def _publish(self,frame,image_p,r,q,q16,confidence,timestamp):
        if self.preview is None: return
        try:
            annotated=self._annotate(frame.copy(),image_p,r,q,q16,confidence,timestamp)
            if not self.preview.publish(annotated): raise RuntimeError("preview process exited")
        except Exception as exc:
            if not self._preview_warned: print(f"WARNING: camera preview disabled: {exc}"); self._preview_warned=True
            self.preview.close(); self.preview=None; self.show=False
    def _annotate(self,frame,image_p,r,q,q16,confidence,timestamp):
        h,w=frame.shape[:2]
        if not hasattr(self,"_last_preview_time"): self._last_preview_time=timestamp; self._preview_fps=0.0
        dt=timestamp-self._last_preview_time
        if dt>0: self._preview_fps=.9*self._preview_fps+.1/dt
        self._last_preview_time=timestamp
        if image_p is not None:
            points=[(int(p[0]*w),int(p[1]*h)) for p in image_p]
            for a,b in HAND_CONNECTIONS: self.cv2.line(frame,points[a],points[b],(0,180,0),1)
            for point in points: self.cv2.circle(frame,point,3,(0,255,0),-1)
            if r is not None:
                origin=points[0]
                for axis,color in zip(r.T,((0,0,255),(0,255,0),(255,0,0))):
                    self.cv2.line(frame,origin,(origin[0]+int(axis[0]*80),origin[1]+int(axis[1]*80)),color,2)
        tracked=q is not None
        lines=["RIGHT HAND" if tracked else "TRACKING LOST",f"confidence {confidence:.2f}",f"FPS {self._preview_fps:.1f}"]
        if tracked: lines += ["q7 "+" ".join(f"{x:.2f}" for x in q),"q16 "+" ".join(f"{x:.2f}" for x in q16)]
        for i,line in enumerate(lines): self.cv2.putText(frame,line,(10,25+22*i),self.cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        return frame
    def close(self):
        if self.preview is not None: self.preview.close(); self.preview=None
        self.cap.release(); self.hands.close()


class PlaybackSource(ControlSource):
    def __init__(self,path): self.data=np.load(path,allow_pickle=False); self.i=0
    def update(self):
        if self.i>=len(self.data["compact01"]): raise StopIteration
        i=self.i; self.i+=1
        return HumanHandState(time.monotonic(),self.data["joint_angles_rad"][i],self.data["compact01"][i],
            self.data["palm_rotation"][i],self.data["palm_quaternion"][i],1.0,"Right")


class EMGSource(ControlSource):
    def __init__(self,reader,preprocessor,window,model,visualizer=None):
        self.reader=reader; self.pre=preprocessor; self.window=window; self.model=model; self.visualizer=visualizer; self.last_packets=[]; self.last_prediction=None; self.last_confidence=0.0
    def update(self):
        self.last_packets=self.reader.read()
        for packet in self.last_packets:
            processed=self.pre.process_sample(packet.samples); sample_window=self.window.append(processed)
            self.last_confidence=float(np.exp(-max(0,np.max(np.abs(processed))-5))) if np.all((packet.samples>0)&(packet.samples<4095)) else 0.0
            if self.visualizer is not None: self.visualizer.update(packet.samples,processed,self.last_prediction,self.last_confidence)
            if sample_window is not None:
                q=np.clip(np.asarray(self.model.predict(sample_window[None]))[0],0,1); self.last_prediction=q
                from src.control.command_mapper import compact_to_full_deg
                angles=np.deg2rad(compact_to_full_deg(q))
                return HumanHandState(time.monotonic(),angles,q,np.eye(3),np.array([0,0,0,1]),1.0,"Right")
        return None
    def close(self): self.reader.close()
