#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
import uuid

import cv2
import mediapipe as mp

from src.app.sources import landmark_arrays
from src.perception.hand_geometry import anatomical_angles
from src.perception.hand_calibration import HandCalibration


WINDOW_NAME = "Right-hand calibration"
POSES = (
    ("open_palm", "OPEN PALM"),
    ("closed_fist", "CLOSED FIST"),
    ("thumb_abduction", "MAXIMUM THUMB ABDUCTION"),
    ("thumb_opposition", "THUMB ACROSS PALM / OPPOSITION"),
    ("pinch", "THUMB-INDEX PINCH"),
)


def draw_lines(frame, lines):
    for row, (text, color) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (12, 30 + row * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )


def tracked_right_hand(result, confidence_threshold):
    if not result.multi_hand_landmarks or not result.multi_handedness:
        return False, 0.0
    classification = result.multi_handedness[0].classification[0]
    confidence = float(classification.score)
    return classification.label == "Right" and confidence >= confidence_threshold, confidence


def capture_pose(cap, hands, drawing, styles, label, seconds, minimum_frames, confidence_threshold):
    state = "waiting"
    samples = []
    deadline = None

    while True:
        ok, frame = cap.read()
        if not ok:
            key = cv2.waitKey(10) & 0xFF
            if key == 27:
                return None
            continue

        # MediaPipe Hands handedness is defined for mirrored/selfie input. The
        # same mirrored frame is displayed so landmarks and feedback coincide.
        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detected, confidence = tracked_right_hand(result, confidence_threshold)

        if result.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame,
                result.multi_hand_landmarks[0],
                mp.solutions.hands.HAND_CONNECTIONS,
                styles.get_default_hand_landmarks_style(),
                styles.get_default_hand_connections_style(),
            )

        now = time.monotonic()
        if state == "capturing":
            if detected:
                try:
                    # World landmarks are selected when present; landmark_arrays
                    # provides the documented image-landmark compatibility fallback.
                    _, geometry, _ = landmark_arrays(result)
                    samples.append(anatomical_angles(geometry))
                except (ValueError, IndexError):
                    pass
            remaining = max(0.0, deadline - now)
            lines = (
                (f"Capturing {label}...", (0, 255, 255)),
                (f"Right hand detected: {'YES' if detected else 'NO'}", (0, 255, 0) if detected else (0, 0, 255)),
                (f"Confidence: {confidence:.2f}", (255, 255, 255)),
                (f"Valid frames: {len(samples)}", (255, 255, 255)),
                (f"Time remaining: {remaining:.1f} s", (255, 255, 255)),
                ("ESC to cancel", (200, 200, 200)),
            )
            if now >= deadline:
                if len(samples) >= minimum_frames:
                    return samples
                state = "retry"
        else:
            instruction = "Insufficient valid frames — press SPACE to retry" if state == "retry" else "Press SPACE to begin capture"
            lines = (
                (label, (0, 255, 255)),
                (f"Right hand detected: {'YES' if detected else 'NO'}", (0, 255, 0) if detected else (0, 0, 255)),
                (f"Confidence: {confidence:.2f}", (255, 255, 255)),
                (f"Valid frames: {len(samples)}", (255, 255, 255)),
                (instruction, (255, 255, 255)),
                ("ESC to cancel", (200, 200, 200)),
            )

        draw_lines(frame, lines)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return None
        if key == 32 and state in {"waiting", "retry"}:
            samples = []
            deadline = time.monotonic() + seconds
            state = "capturing"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", default="calibration_data/aero_right_hand_calibration.json")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--minimum-frames", type=int, default=10)
    args = parser.parse_args(argv)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Unable to open camera {args.camera}", file=sys.stderr)
        return 1

    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=args.confidence,
        min_tracking_confidence=args.confidence,
    )
    poses = {}
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        for key, label in POSES:
            samples = capture_pose(
                cap,
                hands,
                mp.solutions.drawing_utils,
                mp.solutions.drawing_styles,
                label,
                args.seconds,
                args.minimum_frames,
                args.confidence,
            )
            if samples is None:
                print("Calibration cancelled; no calibration file was written.")
                return 0
            poses[key] = samples

        calibration = HandCalibration.from_pose_samples(poses, str(uuid.uuid4()))
        calibration.save(args.output)
        print(f"saved {args.output} ({calibration.calibration_id})")
        return 0
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
