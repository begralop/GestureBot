"""Visualiza en tiempo real los 21 landmarks de una mano con MediaPipe Tasks."""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def draw_hand(frame: np.ndarray, landmarks: list, label: str) -> None:
    h, w = frame.shape[:2]
    points = [
        (int(np.clip(lm.x, 0, 1) * w), int(np.clip(lm.y, 0, 1) * h))
        for lm in landmarks
    ]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (40, 210, 255), 2, cv2.LINE_AA)
    for idx, point in enumerate(points):
        cv2.circle(frame, point, 5, (255, 80, 80), -1, cv2.LINE_AA)
        if idx in (0, 4, 8, 12, 16, 20):
            cv2.putText(frame, str(idx), (point[0] + 5, point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    x, y = points[0]
    cv2.putText(frame, label, (max(10, x - 30), max(30, y + 35)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 120), 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model_path = args.model or (root / "models" / "hand_landmarker.task")
    if not model_path.exists():
        print(f"No existe {model_path}. Ejecuta: python scripts/download_hand_model.py", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara {args.camera}.", file=sys.stderr)
        return 1

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    start = time.monotonic()
    previous = start
    last_ts = -1
    print("MediaPipe iniciado. Pulsa q o ESC para salir.")
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    return 1
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
                last_ts = ts
                result = landmarker.detect_for_video(mp_image, ts)

                if result.hand_landmarks:
                    label = "Mano"
                    if result.handedness and result.handedness[0]:
                        c = result.handedness[0][0]
                        label = f"{c.category_name or 'Mano'} {float(c.score or 0):.2f}"
                    draw_hand(frame, result.hand_landmarks[0], label)
                    status, color = "21 landmarks detectados", (80, 255, 120)
                else:
                    status, color = "No se detecta ninguna mano", (80, 80, 255)

                now = time.monotonic()
                fps = 1.0 / max(now - previous, 1e-6)
                previous = now
                cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("GestureBot - Hand Landmarker", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
