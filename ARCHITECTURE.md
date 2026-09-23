# Architecture

```text
Right hand -> VisionSource ----\
                              -> HumanHandState -> CommandMapper -> AeroCommand -> Safety -> MuJoCo
4ch EMG -> EMGSource/model ---/                                            |-----> official Aero SDK
                                      synchronized SessionLogger <---------+ telemetry
```

`HumanHandState` and `AeroCommand` are immutable canonical boundaries. Vision and EMG therefore have no backend-specific mapping. The real backend sends all 16 desired joint angles to the official SDK. The simulation backend applies the same published Chestnut joint-to-actuation equations, then converts 9 mm pulley travel to the official Menagerie tendon-length controls. `DualAeroBackend` forwards the identical object to both.

The dispatch order is: world-landmark geometry → user calibration → canonical mapping → time-constant low-pass filter → per-channel slew limiter → joint-limit/safety validation → backend. Both children of the dual backend receive the same final immutable command object.

The legacy SO-arm XYZ/IK path remains intact but is not used by direct mimic. Optional wrist following uses relative palm rotation and an explicit configurable axis map.

The EMG wire format is little-endian: `A5 5A`, version byte, channel-count byte, uint32 sequence, uint32 MCU microseconds, N uint16 ADC samples, CRC16-CCITT. Window labels default to the center timestamp.

Vision recording retains both normalized image landmarks and MediaPipe world landmarks. MCU receive observations continuously update a linear host/MCU clock fit. Each sample is associated with the nearest retained camera label at the configured window center or end reference.
