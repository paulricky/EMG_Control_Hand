# Safety

The Aero hardware backend is dry-run unless `--enable-hardware` is explicitly supplied. Before enabling it, secure the arm, keep people clear of pinch points, verify right-hand firmware, homing, telemetry, current and temperature limits, and test a slow open-hand target. Use a reachable physical power disconnect.

Surface electrodes are body-connected instrumentation. Use a battery-powered, medically appropriate isolated acquisition path; never connect an unsafe mains-referenced circuit to electrodes. Inspect leads and skin, stop on discomfort, and do not use this research apparatus for diagnosis or treatment.

Communication, stale tracking, non-finite input, over-current, over-temperature, or excessive velocity are faults. Configure `hold`, `controlled_open`, or `disable` for the actual fixture; opening can itself be hazardous. Validate the selected response at low torque before normal operation.
