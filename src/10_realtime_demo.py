"""Demo en tiempo real de GestureBot sin ESP32.

Abre la webcam, detecta una mano con MediaPipe, genera las mismas 128
características usadas durante el entrenamiento y muestra en pantalla:

- gesto predicho;
- confianza;
- mano detectada;
- comando simulado;
- estabilidad temporal.

Modelo por defecto:
    models/gesture_model_ensemble.joblib

Uso:
    python src/10_realtime_demo.py

Controles:
    Q o ESC: cerrar
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import joblib
import mediapipe as mp
import numpy as np


CLASS_ORDER = ("FORWARD", "STOP", "BACKWARD", "GRIPPER", "OTHER")
PALM_MCP_INDICES = (5, 9, 13, 17)
EPSILON = 1e-8

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
        description="Demo visual de GestureBot sin ESP32."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=None,
        help="Modelo hand_landmarker.task de MediaPipe.",
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        default=None,
        help="Modelo joblib entrenado. Por defecto usa el ensemble.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Confianza mínima para aceptar un gesto.",
    )
    parser.add_argument(
        "--confirm-frames",
        type=int,
        default=4,
        help="Frames consecutivos necesarios para confirmar un comando.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Número de predicciones usadas para suavizar probabilidades.",
    )
    return parser.parse_args()


def corrected_handedness(category_name: str | None) -> str:
    """Corrige la lateralidad porque la imagen se muestra en modo espejo."""
    if category_name == "Left":
        return "RIGHT"
    if category_name == "Right":
        return "LEFT"
    return "UNKNOWN"


def landmarks_to_array(landmarks: list) -> np.ndarray:
    array = np.asarray(
        [[float(point.x), float(point.y), float(point.z)] for point in landmarks],
        dtype=np.float32,
    )
    if array.shape != (21, 3):
        raise ValueError(f"Forma inesperada de landmarks: {array.shape}")
    return array


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Muñeca al origen y escala por el tamaño medio de la palma."""
    centered = landmarks - landmarks[0]
    palm_vectors = centered[list(PALM_MCP_INDICES)]
    palm_distances = np.linalg.norm(palm_vectors, axis=1)
    scale = float(np.mean(palm_distances))

    if not math.isfinite(scale) or scale <= EPSILON:
        raise ValueError("No se puede calcular una escala válida.")

    return (centered / scale).astype(np.float32)


def handedness_features(handedness: str) -> np.ndarray:
    if handedness == "LEFT":
        return np.asarray([1.0, 0.0], dtype=np.float32)
    if handedness == "RIGHT":
        return np.asarray([0.0, 1.0], dtype=np.float32)
    raise ValueError(f"Lateralidad no válida: {handedness}")


def build_feature_vector(
    image_landmarks: list,
    world_landmarks: list,
    handedness: str,
) -> np.ndarray:
    image_array = landmarks_to_array(image_landmarks)
    world_array = landmarks_to_array(world_landmarks)

    image_normalized = normalize_landmarks(image_array)
    world_normalized = normalize_landmarks(world_array)

    vector = np.concatenate(
        (
            image_normalized.reshape(-1),
            world_normalized.reshape(-1),
            handedness_features(handedness),
        )
    ).astype(np.float32)

    if vector.shape != (128,):
        raise ValueError(f"Se esperaban 128 características y hay {vector.shape[0]}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("El vector contiene NaN o infinitos.")

    return vector


def softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    values -= np.max(values)
    exponentials = np.exp(values)
    return exponentials / np.sum(exponentials)


def aligned_model_probabilities(
    model: Any,
    sample: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    """Devuelve probabilidades ordenadas como labels."""
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(sample)[0], dtype=np.float64)
    elif hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(sample))
        if decision.ndim == 2:
            decision = decision[0]
        raw = softmax(decision)
    else:
        prediction = str(model.predict(sample)[0])
        raw = np.zeros(len(labels), dtype=np.float64)
        raw[labels.index(prediction)] = 1.0
        return raw

    model_classes = [str(value) for value in model.classes_]
    aligned = np.zeros(len(labels), dtype=np.float64)

    for target_index, label in enumerate(labels):
        if label not in model_classes:
            raise ValueError(f"El modelo no contiene la clase {label}.")
        aligned[target_index] = raw[model_classes.index(label)]

    return aligned


def predict_artifact(
    artifact: dict[str, Any],
    feature_vector: np.ndarray,
) -> tuple[str, float, np.ndarray, list[str]]:
    labels = [str(value) for value in artifact["class_names"]]
    sample = feature_vector.reshape(1, -1)

    if "models" in artifact:
        models = artifact["models"]
        weights = artifact.get("weights", {})
        combined = np.zeros(len(labels), dtype=np.float64)
        total_weight = 0.0

        for model_name, model in models.items():
            weight = float(weights.get(model_name, 1.0))
            if weight <= 0:
                continue
            combined += weight * aligned_model_probabilities(model, sample, labels)
            total_weight += weight

        if total_weight <= 0:
            raise ValueError("Los pesos del ensemble no son válidos.")
        probabilities = combined / total_weight

    elif "model" in artifact:
        probabilities = aligned_model_probabilities(
            artifact["model"], sample, labels
        )
    else:
        raise ValueError("El artefacto no contiene 'models' ni 'model'.")

    best_index = int(np.argmax(probabilities))
    return (
        labels[best_index],
        float(probabilities[best_index]),
        probabilities,
        labels,
    )


def draw_hand(frame: np.ndarray, landmarks: list) -> None:
    height, width = frame.shape[:2]
    points = [
        (
            int(np.clip(point.x, 0.0, 1.0) * width),
            int(np.clip(point.y, 0.0, 1.0) * height),
        )
        for point in landmarks
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


def put_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    scale: float = 0.65,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame: np.ndarray,
    *,
    handedness: str,
    raw_label: str,
    confidence: float,
    stable_command: str,
    progress: int,
    confirm_frames: int,
    hand_detected: bool,
) -> None:
    height, width = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (min(width, 480), 185), (18, 18, 18), -1)

    if hand_detected:
        prediction_color = (90, 255, 120) if confidence >= 0.65 else (0, 190, 255)
        lines = [
            ("GestureBot - demo sin ESP32", (255, 255, 255)),
            (f"Mano: {handedness}", (255, 220, 90)),
            (f"Prediccion: {raw_label}", prediction_color),
            (f"Confianza: {confidence * 100:.1f} %", prediction_color),
            (
                f"Estabilidad: {min(progress, confirm_frames)}/{confirm_frames}",
                (220, 220, 220),
            ),
        ]
    else:
        lines = [
            ("GestureBot - demo sin ESP32", (255, 255, 255)),
            ("Sin mano detectada", (0, 190, 255)),
            ("Prediccion: --", (220, 220, 220)),
            ("Confianza: --", (220, 220, 220)),
            ("Seguridad: comando STOP", (90, 255, 120)),
        ]

    y = 30
    for text, color in lines:
        put_text(frame, text, (15, y), 0.65, color)
        y += 32

    command_color = {
        "FORWARD": (70, 220, 70),
        "BACKWARD": (255, 170, 60),
        "GRIPPER": (230, 120, 255),
        "STOP": (60, 60, 255),
    }.get(stable_command, (180, 180, 180))

    box_height = 72
    cv2.rectangle(
        frame,
        (0, height - box_height),
        (width, height),
        (18, 18, 18),
        -1,
    )
    put_text(
        frame,
        f"COMANDO SIMULADO: {stable_command}",
        (20, height - 25),
        0.9,
        command_color,
        3,
    )
    put_text(
        frame,
        "Q / ESC para salir",
        (max(20, width - 205), height - 26),
        0.48,
        (210, 210, 210),
        1,
    )


def main() -> int:
    args = parse_args()

    if not 0.0 <= args.min_confidence <= 1.0:
        print("--min-confidence debe estar entre 0 y 1.", file=sys.stderr)
        return 2
    if args.confirm_frames < 1:
        print("--confirm-frames debe ser al menos 1.", file=sys.stderr)
        return 2
    if args.smoothing_window < 1:
        print("--smoothing-window debe ser al menos 1.", file=sys.stderr)
        return 2

    project_root = Path(__file__).resolve().parents[1]
    hand_model_path = args.hand_model or (
        project_root / "models" / "hand_landmarker.task"
    )
    classifier_path = args.classifier or (
        project_root / "models" / "gesture_model_ensemble.joblib"
    )

    if not hand_model_path.exists():
        print(
            f"No existe el modelo de MediaPipe: {hand_model_path}\n"
            "Ejecuta: python scripts/download_hand_model.py",
            file=sys.stderr,
        )
        return 1

    if not classifier_path.exists():
        print(
            f"No existe el clasificador: {classifier_path}\n"
            "Ejecuta primero el entrenamiento del ensemble.",
            file=sys.stderr,
        )
        return 1

    try:
        artifact = joblib.load(classifier_path)
    except Exception as error:
        print(f"No se pudo cargar el clasificador: {error}", file=sys.stderr)
        return 1

    expected_features = int(artifact.get("number_of_features", 128))
    if expected_features != 128:
        print(
            f"El modelo espera {expected_features} características, no 128.",
            file=sys.stderr,
        )
        return 1

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"No se pudo abrir la cámara {args.camera}.", file=sys.stderr)
        return 1

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(hand_model_path)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    probability_history: deque[np.ndarray] = deque(
        maxlen=args.smoothing_window
    )
    candidate_label = "STOP"
    candidate_count = 0
    stable_command = "STOP"

    start_time = time.monotonic()
    last_timestamp_ms = -1

    print("=" * 68)
    print("GESTUREBOT — DEMO EN TIEMPO REAL")
    print("=" * 68)
    print(f"Clasificador: {classifier_path.relative_to(project_root)}")
    print("Q o ESC para cerrar.")
    print()

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    print("No se pudo leer un frame.", file=sys.stderr)
                    return 1

                # Igual que durante la recogida del dataset: imagen en espejo.
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

                hand_detected = bool(
                    result.hand_landmarks and result.hand_world_landmarks
                )
                handedness = "--"
                raw_label = "--"
                confidence = 0.0

                if hand_detected:
                    image_landmarks = result.hand_landmarks[0]
                    world_landmarks = result.hand_world_landmarks[0]
                    draw_hand(frame, image_landmarks)

                    if result.handedness and result.handedness[0]:
                        category = result.handedness[0][0]
                        handedness = corrected_handedness(
                            category.category_name
                        )

                    try:
                        feature_vector = build_feature_vector(
                            image_landmarks,
                            world_landmarks,
                            handedness,
                        )
                        (
                            _instant_label,
                            _instant_confidence,
                            probabilities,
                            labels,
                        ) = predict_artifact(artifact, feature_vector)

                        probability_history.append(probabilities)
                        smoothed = np.mean(
                            np.stack(probability_history, axis=0),
                            axis=0,
                        )
                        best_index = int(np.argmax(smoothed))
                        raw_label = labels[best_index]
                        confidence = float(smoothed[best_index])

                        accepted_label = (
                            raw_label
                            if confidence >= args.min_confidence
                            and raw_label != "OTHER"
                            else "STOP"
                        )

                        # STOP siempre se aplica inmediatamente por seguridad.
                        if accepted_label == "STOP":
                            stable_command = "STOP"
                            candidate_label = "STOP"
                            candidate_count = args.confirm_frames
                        else:
                            if accepted_label == candidate_label:
                                candidate_count += 1
                            else:
                                candidate_label = accepted_label
                                candidate_count = 1

                            if candidate_count >= args.confirm_frames:
                                stable_command = candidate_label

                    except ValueError as error:
                        raw_label = "ERROR"
                        confidence = 0.0
                        stable_command = "STOP"
                        candidate_label = "STOP"
                        candidate_count = 0
                        probability_history.clear()
                        print(f"Frame descartado: {error}")
                else:
                    probability_history.clear()
                    candidate_label = "STOP"
                    candidate_count = 0
                    stable_command = "STOP"

                draw_overlay(
                    frame,
                    handedness=handedness,
                    raw_label=raw_label,
                    confidence=confidence,
                    stable_command=stable_command,
                    progress=candidate_count,
                    confirm_frames=args.confirm_frames,
                    hand_detected=hand_detected,
                )

                cv2.imshow("GestureBot - Demo", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break

    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
