from __future__ import annotations
import argparse, json, time
from collections import deque
from pathlib import Path
import sys
import numpy as np
from src.control.command_mapper import CommandMapper
from src.control.safety import SafetySupervisor, SafetyError
from src.control.filters import CommandConditioner
from src.backends.real_aero import RealAeroBackend
from src.backends.mujoco_aero import MujocoAeroBackend
from src.backends.dual_aero import DualAeroBackend
from .sources import VisionSource, PlaybackSource, EMGSource
from src.emg.serial_reader import SerialEmgReader
from src.emg.preprocessing import CausalPreprocessor
from src.emg.window_buffer import WindowBuffer
from src.ml.train import RidgeEmgModel
from src.logging.session_logger import SessionLogger
from src.logging.synchronization import ClockAligner
from src.arm.wrist_mapper import WristMapper
from src.arm.existing_arm_adapter import ExistingArmAdapter
from .macos_launcher import ensure_mjpython

PROJECT_ROOT=Path(__file__).resolve().parents[2]


def load_config(path):
    config={}
    candidates=[PROJECT_ROOT/"configs"/name for name in ("hand_right.yaml","vision.yaml","emg.yaml","safety.yaml","mujoco.yaml","hardware.yaml")]
    if path: candidates.append(Path(path) if Path(path).is_absolute() else PROJECT_ROOT/Path(path))
    for candidate in candidates:
        if not candidate.is_file(): continue
        text=candidate.read_text()
        try: import yaml; values=yaml.safe_load(text) or {}
        except ImportError: values=json.loads(text)
        config.update(values)
    for key in ("calibration_file","emg_calibration_file","model_checkpoint","mujoco_model_path","playback_path","record_path"):
        if config.get(key) and not Path(config[key]).is_absolute(): config[key]=str(PROJECT_ROOT/config[key])
    return config


def parser():
    p=argparse.ArgumentParser(description="Right-hand Aero teleoperation")
    p.add_argument("--source",choices=("vision","emg","playback"),required=True); p.add_argument("--backend",choices=("sim","real","both"),required=True)
    p.add_argument("--config"); p.add_argument("--playback"); p.add_argument("--right-hand-only",action="store_true",default=True)
    p.add_argument("--show-camera",action="store_true"); p.add_argument("--show-mujoco",action="store_true"); p.add_argument("--record",action="store_true")
    p.add_argument("--record-emg",action="store_true"); p.add_argument("--dry-run",action="store_true"); p.add_argument("--enable-hardware",action="store_true")
    p.add_argument("--wrist-follow",action="store_true")
    p.add_argument("--show-emg",action="store_true")
    p.add_argument("--diagnostics",action="store_true",help="rate-limited vision/command/MuJoCo motion diagnostics")
    return p


def build(args,cfg):
    if args.source=="vision": source=VisionSource(cfg.get("camera_id",0),cfg.get("right_hand_confidence",.7),args.show_camera,cfg.get("calibration_file"))
    elif args.source=="playback": source=PlaybackSource(args.playback or cfg["playback_path"])
    else:
        emg_cal={}
        if cfg.get("emg_calibration_file") and Path(cfg["emg_calibration_file"]).is_file(): emg_cal=json.loads(Path(cfg["emg_calibration_file"]).read_text())
        reader=SerialEmgReader(cfg["emg_serial_port"],cfg.get("emg_baudrate",921600),cfg.get("emg_channels",4)).connect()
        pre=CausalPreprocessor(cfg.get("emg_channels",4),emg_cal.get("baseline",cfg.get("emg_baseline")),cfg.get("emg_mean"),emg_cal.get("scale",cfg.get("emg_std")),
            cfg.get("emg_sample_rate_hz",2000),cfg.get("notch_hz"),cfg.get("notch_q",30),cfg.get("bandpass_hz"))
        window=WindowBuffer(cfg.get("emg_channels",4),cfg.get("emg_sample_rate_hz",2000),cfg.get("window_s",.2),cfg.get("stride_s",.05))
        visualizer=None
        if args.show_emg:
            from src.app.emg_visualization import EmgVisualizer
            visualizer=EmgVisualizer(cfg.get("emg_channels",4))
        checkpoint=cfg["model_checkpoint"]
        if str(checkpoint).endswith((".pt",".pth")):
            from src.ml.temporal import TemporalEmgModel
            model=TemporalEmgModel.load(checkpoint)
        else: model=RidgeEmgModel.load(checkpoint)
        source=EMGSource(reader,pre,window,model,visualizer)
    real=RealAeroBackend(cfg.get("aero_serial_port"),cfg.get("aero_baudrate",921600),args.enable_hardware and not args.dry_run,
        speed=cfg.get("aero_speed",5000),torque=cfg.get("aero_torque",300))
    if args.backend=="real": backend=real
    else:
        model=cfg.get("mujoco_model_path","mujoco_menagerie/tetheria_aero_hand_open/scene_right.xml")
        sim=MujocoAeroBackend(model,args.show_mujoco); backend=sim if args.backend=="sim" else DualAeroBackend(sim,real)
    return source,backend


def main(argv=None):
    args=parser().parse_args(argv)
    ensure_mjpython(args.show_mujoco, argv=(sys.argv[1:] if argv is None else argv), module="src.app.runtime")
    cfg=load_config(args.config); source,backend=build(args,cfg)
    mapper=CommandMapper(cfg.get("finger_weights",[1/3]*3),cfg.get("thumb_weights",[.5,.5]))
    conditioner=CommandConditioner(cfg.get("command_filter_tau_s",.08),cfg.get("slew_rate_01_per_s",[1]*7))
    safety=SafetySupervisor(cfg.get("command_timeout_s",.5),cfg.get("fault_action","hold"),cfg.get("max_current_ma",1200),
        cfg.get("max_temperature_c",65),cfg.get("max_velocity_rpm",100),cfg.get("communication_timeout_s",.5))
    logger=SessionLogger(cfg.get("record_path","recordings/session.npz"),{"source":args.source,"backend":args.backend,
        "calibration_id":getattr(getattr(source,"calibration",None),"calibration_id",None),"label_reference":cfg.get("label_reference","center")}) if args.record or args.record_emg else None
    emg_reader=emg_pre=None; aligner=ClockAligner()
    if args.record_emg:
        if args.source != "vision": raise ValueError("--record-emg requires --source vision; vision remains the controller")
        emg_cal={}
        if cfg.get("emg_calibration_file") and Path(cfg["emg_calibration_file"]).is_file(): emg_cal=json.loads(Path(cfg["emg_calibration_file"]).read_text())
        emg_reader=SerialEmgReader(cfg["emg_serial_port"],cfg.get("emg_baudrate",921600),cfg.get("emg_channels",4)).connect()
        emg_pre=CausalPreprocessor(cfg.get("emg_channels",4),emg_cal.get("baseline",cfg.get("emg_baseline")),cfg.get("emg_mean"),emg_cal.get("scale",cfg.get("emg_std")),
            cfg.get("emg_sample_rate_hz",2000),cfg.get("notch_hz"),cfg.get("notch_q",30),cfg.get("bandpass_hz"))
    arm=None; wrist_mapper=None
    if args.wrist_follow:
        if not args.enable_hardware: raise RuntimeError("--wrist-follow requires --enable-hardware")
        from robot_controller import SOArmHardwareController
        neutral=cfg.get("arm_neutral_rad",{})
        arm=ExistingArmAdapter(SOArmHardwareController(),neutral,cfg.get("wrist_target_joints",["wrist_pitch","wrist_roll","wrist_yaw"])).connect()
    backend.connect()
    last_state=last_command=None; last_telemetry={}; last_poll=0.0; last_physics=time.monotonic(); visual_history=deque(maxlen=240); source_faulted=False
    last_diag=last_change=time.monotonic(); previous_q=previous_ctrl=previous_qpos=None
    try:
        while True:
            if getattr(source,"shutdown_requested",False): break
            state=source.update()
            if state is not None:
                if source_faulted: print(f"INFO: {args.source} tracking restored")
                source_faulted=False
                command=safety.validate(conditioner.update(mapper.map(state,args.source))); backend.send(command)
                last_state,last_command=state,command
                visual_history.append((state.timestamp_s,state,command,dict(getattr(source,"last_observation",{}) or {})))
                if args.wrist_follow:
                    if wrist_mapper is None: wrist_mapper=WristMapper(state.palm_rotation,cfg.get("wrist_axis_order",[0,1,2]),cfg.get("wrist_axis_signs",[1,1,1]),cfg.get("wrist_zero_offsets_rad",[0,0,0]),cfg.get("wrist_limits_rad",[[-3.14,3.14]]*3))
                    arm.send_wrist(wrist_mapper.map(state.palm_rotation))
            now=time.monotonic(); timestep=float(cfg.get("physics_timestep_s",.01)); steps=min(20,max(1,int((now-last_physics)/timestep)))
            timeout=cfg.get("emg_inference_timeout_s",.25) if args.source=="emg" else cfg.get("tracking_timeout_s",.5)
            if last_state is not None and now-last_state.timestamp_s>timeout and not source_faulted:
                backend.handle_fault(safety.fault_action,last_command); source_faulted=True
                print(f"WARNING: {args.source} source timeout; applied {safety.fault_action} and waiting for recovery")
            for _ in range(steps): backend.step()
            last_physics += steps*timestep
            if backend.render() is False: break
            if now-last_poll >= 1/max(float(cfg.get("telemetry_poll_hz",20)),1):
                try: last_telemetry=backend.read_state(); safety.check_telemetry(last_telemetry.get("hardware",last_telemetry),now)
                except Exception as exc:
                    backend.handle_fault(safety.fault_action,last_command)
                    raise SafetyError(f"telemetry/communication failure: {exc}") from exc
                last_poll=now
            if args.diagnostics and now-last_diag>=1:
                q=None if state is None else np.asarray(state.compact01); ctrl=np.asarray(last_telemetry.get("actuator_ctrl",[])); qpos=np.asarray(last_telemetry.get("joint_rad",[]))
                changed=lambda a,b: a is not None and b is not None and a.size and b.size and not np.allclose(a,b,atol=1e-5)
                moved_q,moved_ctrl,moved_qpos=changed(q,previous_q),changed(ctrl,previous_ctrl),changed(qpos,previous_qpos)
                if moved_q or moved_ctrl or moved_qpos: last_change=now
                print(f"DIAG tracking={state is not None} confidence={getattr(state,'tracking_confidence',0):.2f} q7={None if q is None else np.round(q,3)} command={None if last_command is None else np.round(last_command.compact01,3)} ctrl={np.round(ctrl,4)} qpos_changed={moved_qpos}")
                if state is not None and now-last_change>3: print("WARNING: valid tracking is present but q7/ctrl/qpos have remained unchanged for over 3 seconds")
                previous_q=None if q is None else q.copy(); previous_ctrl=ctrl.copy(); previous_qpos=qpos.copy(); last_diag=now
            if logger is not None and state is not None and not args.record_emg:
                obs=getattr(source,"last_observation",{}) or {}; logger.append(host_timestamp_s=state.timestamp_s,camera_timestamp_s=obs.get("camera_timestamp_s"),camera_frame=obs.get("camera_frame"),
                    image_landmarks=obs.get("image_landmarks"),world_landmarks=obs.get("world_landmarks"),joint_angles_rad=state.joint_angles_rad,
                    joint_normalized01=state.joint_normalized01,compact01=last_command.compact01,joint_angles_deg=last_command.joint_angles_deg,
                    tracking_confidence=state.tracking_confidence,backend_state=last_telemetry)
            if emg_reader is not None and last_state is not None:
                obs=getattr(source,"last_observation",{}) or {}
                for packet in emg_reader.read():
                    aligner.update(packet.mcu_timestamp_us,packet.host_timestamp_s); processed=emg_pre.process_sample(packet.samples)
                    aligned=aligner.to_host(packet.mcu_timestamp_us); label_time=aligned-(cfg.get("window_s",.2)/2 if cfg.get("label_reference","center")=="center" else 0)
                    _,label_state,label_command,label_obs=min(visual_history,key=lambda item:abs(item[0]-label_time))
                    logger.append(mcu_timestamp_us=packet.mcu_timestamp_us,emg_host_timestamp_s=packet.host_timestamp_s,
                        aligned_host_timestamp_s=aligned,label_timestamp_s=label_time,raw_emg=packet.samples,processed_emg=processed,
                        camera_timestamp_s=label_obs.get("camera_timestamp_s"),camera_frame=label_obs.get("camera_frame"),image_landmarks=label_obs.get("image_landmarks"),
                        world_landmarks=label_obs.get("world_landmarks"),joint_angles_rad=label_state.joint_angles_rad,joint_normalized01=label_state.joint_normalized01,
                        compact01=label_command.compact01,joint_angles_deg=label_command.joint_angles_deg,tracking_confidence=label_state.tracking_confidence,
                        calibration_id=source.calibration.calibration_id,backend_state=last_telemetry)
            if state is None: time.sleep(.001)
    except SafetyError as exc:
        print(f"SAFETY STOP ({safety.fault_action}): {exc}")
    except (KeyboardInterrupt,StopIteration): pass
    finally:
        source.close(); backend.shutdown()
        if emg_reader is not None: emg_reader.close()
        if arm is not None: arm.close()
        if logger is not None: logger.close()


if __name__=="__main__": main()
