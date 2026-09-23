# Right-hand Aero teleoperation

Install with `python -m pip install -r requirements.txt`. Obtain the official model with `git clone https://github.com/google-deepmind/mujoco_menagerie.git`, then run `python scripts/setup_mujoco.py /path/to/mujoco_menagerie`. Install the current hardware SDK with `python -m pip install aero-open-sdk`.

Run vision simulation with `python -m src.app.runtime --source vision --backend sim --config configs/mujoco.yaml --show-camera --show-mujoco`. Replace `sim` with `real` or `both`; physical movement additionally requires `--enable-hardware`. Without that switch, real mode is telemetry-free dry-run. Playback uses `--source playback --playback SESSION.npz`.

EMG configuration is in `configs/emg.yaml`; packet firmware must emit the binary format documented in `ARCHITECTURE.md`. Train the feature baseline with `python scripts/train_emg_model.py SESSION... --output models/emg_baseline.npz`. Compare a synchronized sim-real log using `python scripts/compare_sim_real.py LOG.npz`.

Calibrate vision first with `python scripts/calibrate_hand.py`. Calibrate the N-channel EMG array with `python scripts/calibrate_emg.py --port /dev/ttyACM0`. Record synchronized training data with `python -m src.app.runtime --source vision --backend sim --record-emg`; vision remains the controller while raw and causal-processed EMG are recorded.

Train the temporal baseline with `python scripts/train_emg_temporal.py recordings/*.npz --output models/emg_temporal.pt`. Splits are performed by entire recording session.

The project is right-hand only. Perform the five-pose vision calibration and the rest/open/fist/individual-finger/thumb/pinch EMG calibration for each operator. Hardware identity, homing, telemetry, safety thresholds, palm-axis signs, and wrist-axis mapping must be validated on the actual apparatus.
