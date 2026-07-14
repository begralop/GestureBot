"""Recoge un dataset propio de gestos usando MediaPipe Hand Landmarker.

Controles:
- 1: seleccionar FORWARD
- 2: seleccionar STOP
- 3: seleccionar BACKWARD
- 4: seleccionar GRIPPER
- 0: seleccionar OTHER
- R: iniciar/detener grabación
- Q o ESC: salir
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np

LABEL_KEYS = {
    ord("1"): "FORWARD",
    ord("2"): "STOP",
    ord("3"): "BACKWARD",
    ord("4"): "GRIPPER",
    ord("0"): "OTHER",
}

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crea un dataset CSV de landmarks de la mano."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--session", type=str, default="1")
    parser.add_argument("--sample-every", type=int, default=3)
    return parser.parse_args()


def corrected_handedness(category_name: str | None) -> str:
    if category_name == "Left":
        return "RIGHT"
    if category_name == "Right":
        return "LEFT"
    return "UNKNOWN"


def build_header() -> list[str]:
    header = [
        "timestamp_utc",
        "subject",
        "session",
        "label",
        "handedness",
        "handedness_score",
    ]
    for prefix in ("img", "world"):
        for index in range(21):
            header.extend(
                (
                    f"{prefix}_x{index}",
                    f"{prefix}_y{index}",
                    f"{prefix}_z{index}",
                )
            )
    return header


def flatten_landmarks(landmarks: Iterable) -> list[float]:
    values: list[float] = []
    for landmark in landmarks:
        values.extend(
            (
                float(landmark.x),
                float(landmark.y),
                float(landmark.z),
            )
        )
    return values


def append_sample(
    csv_path: Path,
    *,
    subject: str,
    session: str,
    label: str,
    handedness: str,
    handedness_score: float,
    image_landmarks: list,
    world_landmarks: list,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0

    row = [
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        subject,
        session,
        label,
        handedness,
        f"{handedness_score:.6f}",
        *flatten_landmarks(image_landmarks),
        *flatten_landmarks(world_landmarks),
    ]

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(build_header())
        writer.writerow(row)


def read_existing_counts(csv_path: Path) -> Counter:
    counts: Counter = Counter()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return counts

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                label = row.get("label")
                if label:
                    counts[label] += 1
    except (OSError, csv.Error):
        pass
    return counts


def draw_hand(frame: np.ndarray, landmarks: list) -> None:
    height, width = frame.shape[:2]
    points = [
        (
            int(np.clip(landmark.x, 0.0, 1.0) * width),
            int(np.clip(landmark.y, 0.0, 1.0) * height),
        )
        for landmark in landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (40, 210, 255),
            2,
            cv2.LINE_AA,
        )

    for point in points:
        cv2.circle(frame, point, 4, (255, 80, 80), -1, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    *,
    selected_label: str,
    recording: bool,
    counts: Counter,
    subject: str,
    session: str,
    hand_status: str,
) -> None:
    cv2.rectangle(frame, (0, 0), (590, 245), (15, 15, 15), -1)

    state_text = "GRABANDO" if recording else "PAUSADO"
    state_color = (50, 50, 255) if recording else (180, 180, 180)

    lines = [
        (f"Persona: {subject} | Sesion: {session}", (255, 255, 255)),
        (f"Clase seleccionada: {selected_label}", (80, 255, 120)),
        (f"Estado: {state_text}", state_color),
        (hand_status, (255, 220, 80)),
        ("1 FORWARD | 2 STOP | 3 BACKWARD | 4 GRIPPER | 0 OTHER", (255, 255, 255)),
        ("R grabar/pausar | Q/ESC salir", (255, 255, 255)),
        (
            "Muestras: "
            + " | ".join(
                f"{label}:{counts[label]}"
                for label in ("FORWARD", "STOP", "BACKWARD", "GRIPPER", "OTHER")
            ),
            (255, 255, 255),
        ),
    ]

    y = 28
    for text, color in lines:
        scale = 0.54 if len(text) > 55 else 0.65
        cv2.putText(
            frame,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 31


def main() -> int:
    args = parse_args()

    if args.sample_every < 1:
        print("--sample-every debe ser mayor o igual que 1.", file=sys.stderr)
        return 2

    project_root = Path(__file__).resolve().parents[1]
    model_path = args.model or (project_root / "models" / "hand_landmarker.task")
    output_path = args.output or (
        project_root / "data" / "raw" / "gesture_landmarks.csv"
    )

    if not model_path.exists():
        print(
            f"No existe el modelo: {model_path}\n"
            "Ejecuta primero: python scripts/download_hand_model.py",
            file=sys.stderr,
        )
        return 1

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
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

    selected_label = "FORWARD"
    recording = False
    frame_number = 0
    counts = read_existing_counts(output_path)

    start_time = time.monotonic()
    last_timestamp_ms = -1

    print(f"Dataset: {output_path}")
    print("Controles: 1/2/3/4/0 seleccionan clase, R graba, Q/ESC sale.")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    print("No se pudo leer un frame.", file=sys.stderr)
                    return 1

                frame_number += 1
                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int((time.monotonic() - start_time) * 1000)
                timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                hand_status = "Sin mano detectada"

                if result.hand_landmarks:
                    image_landmarks = result.hand_landmarks[0]

                    if not result.hand_world_landmarks:
                        hand_status = "Mano detectada, pero sin landmarks 3D"
                        draw_hand(frame, image_landmarks)
                        draw_overlay(
                            frame,
                            selected_label=selected_label,
                            recording=recording,
                            counts=counts,
                            subject=args.subject,
                            session=args.session,
                            hand_status=hand_status,
                        )
                        cv2.imshow("GestureBot - Dataset Collector", frame)
                        key = cv2.waitKey(1) & 0xFF

                        if key in LABEL_KEYS:
                            selected_label = LABEL_KEYS[key]
                            recording = False
                        elif key in (ord("r"), ord("R")):
                            recording = not recording
                        elif key in (ord("q"), ord("Q"), 27):
                            break
                        continue

                    world_landmarks = result.hand_world_landmarks[0]
                    draw_hand(frame, image_landmarks)

                    handedness = "UNKNOWN"
                    handedness_score = 0.0

                    if result.handedness and result.handedness[0]:
                        category = result.handedness[0][0]
                        handedness = corrected_handedness(category.category_name)
                        handedness_score = float(category.score or 0.0)

                    hand_status = (
                        f"Mano: {handedness} | confianza: {handedness_score:.2f}"
                    )

                    if recording and frame_number % args.sample_every == 0:
                        append_sample(
                            output_path,
                            subject=args.subject.strip(),
                            session=args.session.strip(),
                            label=selected_label,
                            handedness=handedness,
                            handedness_score=handedness_score,
                            image_landmarks=image_landmarks,
                            world_landmarks=world_landmarks,
                        )
                        counts[selected_label] += 1

                draw_overlay(
                    frame,
                    selected_label=selected_label,
                    recording=recording,
                    counts=counts,
                    subject=args.subject,
                    session=args.session,
                    hand_status=hand_status,
                )

                cv2.imshow("GestureBot - Dataset Collector", frame)
                key = cv2.waitKey(1) & 0xFF

                if key in LABEL_KEYS:
                    selected_label = LABEL_KEYS[key]
                    recording = False
                elif key in (ord("r"), ord("R")):
                    recording = not recording
                elif key in (ord("q"), ord("Q"), 27):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Captura terminada. Dataset guardado en: {output_path}")
    print("Conteo total:", dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
