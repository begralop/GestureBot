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
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def correct_handedness(category_name: str | None) -> str:
    """
    Corrige la lateralidad porque la imagen de la webcam se muestra
    invertida horizontalmente, como un espejo.
    """
    if category_name == "Left":
        return "Derecha"

    if category_name == "Right":
        return "Izquierda"

    return "Mano"


def draw_hand(
    frame: np.ndarray,
    landmarks: list,
    label: str,
) -> None:
    """Dibuja los landmarks y las conexiones de la mano."""

    height, width = frame.shape[:2]

    points = [
        (
            int(np.clip(landmark.x, 0.0, 1.0) * width),
            int(np.clip(landmark.y, 0.0, 1.0) * height),
        )
        for landmark in landmarks
    ]

    # Dibujar conexiones.
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (40, 210, 255),
            2,
            cv2.LINE_AA,
        )

    # Dibujar puntos.
    for index, point in enumerate(points):
        cv2.circle(
            frame,
            point,
            5,
            (255, 80, 80),
            -1,
            cv2.LINE_AA,
        )

        # Mostrar los índices de puntos relevantes.
        if index in (0, 4, 8, 12, 16, 20):
            cv2.putText(
                frame,
                str(index),
                (point[0] + 5, point[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Mostrar lateralidad junto a la muñeca.
    wrist_x, wrist_y = points[0]

    cv2.putText(
        frame,
        label,
        (max(10, wrist_x - 30), max(30, wrist_y + 35)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (80, 255, 120),
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualiza los landmarks de una mano con MediaPipe."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Índice de la webcam.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Ruta al modelo hand_landmarker.task.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    model_path = (
        args.model
        or project_root / "models" / "hand_landmarker.task"
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
        print(
            f"No se pudo abrir la cámara {args.camera}.",
            file=sys.stderr,
        )
        return 1

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    start_time = time.monotonic()
    previous_time = start_time
    last_timestamp_ms = -1

    print("MediaPipe iniciado. Pulsa q o ESC para salir.")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                ok, frame = capture.read()

                if not ok or frame is None:
                    print(
                        "No se pudo leer un frame de la webcam.",
                        file=sys.stderr,
                    )
                    return 1

                # Imagen en modo espejo.
                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(
                    (time.monotonic() - start_time) * 1000
                )

                timestamp_ms = max(
                    timestamp_ms,
                    last_timestamp_ms + 1,
                )

                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                if result.hand_landmarks:
                    label = "Mano"

                    if result.handedness and result.handedness[0]:
                        category = result.handedness[0][0]

                        corrected_name = correct_handedness(
                            category.category_name
                        )

                        confidence = float(
                            category.score or 0.0
                        )

                        label = (
                            f"{corrected_name} "
                            f"{confidence:.2f}"
                        )

                    draw_hand(
                        frame,
                        result.hand_landmarks[0],
                        label,
                    )

                    status = "21 landmarks detectados"
                    status_color = (80, 255, 120)

                else:
                    status = "No se detecta ninguna mano"
                    status_color = (80, 80, 255)

                current_time = time.monotonic()

                fps = 1.0 / max(
                    current_time - previous_time,
                    1e-6,
                )

                previous_time = current_time

                cv2.putText(
                    frame,
                    status,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "q/ESC para salir",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    "GestureBot - Hand Landmarker",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())