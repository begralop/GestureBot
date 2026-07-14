"""Prepara y normaliza los datasets de landmarks de GestureBot.

Lee todos los CSV válidos de ``data/raw`` y genera:

- ``data/processed/gesture_dataset.csv``
- ``data/processed/gesture_dataset.npz``
- ``data/processed/dataset_summary.json``

La normalización se realiza por muestra:

1. La muñeca (landmark 0) pasa a ser el origen.
2. Se divide por el tamaño de la palma.
3. Se conserva la orientación de la mano y la profundidad Z.
4. Se añaden dos variables binarias para la lateralidad.

No realiza todavía la división train/test. Esa división se hará por persona
o sesión durante el entrenamiento para evitar fuga de datos.

Ejemplos:

    python src/05_prepare_dataset.py

    python src/05_prepare_dataset.py \
        --input-pattern "data/raw/*_s*.csv" \
        --output-dir data/processed
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


VALID_LABELS = ("FORWARD", "STOP", "BACKWARD", "GRIPPER", "OTHER")
METADATA_COLUMNS = (
    "timestamp_utc",
    "subject",
    "session",
    "label",
    "handedness",
    "handedness_score",
)

# Articulaciones metacarpofalángicas usadas para estimar el tamaño de la palma.
PALM_MCP_INDICES = (5, 9, 13, 17)
EPSILON = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normaliza y combina los CSV de GestureBot."
    )
    parser.add_argument(
        "--input-pattern",
        type=str,
        default="data/raw/*_s*.csv",
        help=(
            "Patrón glob de entrada. Por defecto evita archivos de prueba "
            "como belen_test1.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directorio de salida.",
    )
    parser.add_argument(
        "--min-handedness-score",
        type=float,
        default=0.5,
        help="Confianza mínima de lateralidad para aceptar una muestra.",
    )
    parser.add_argument(
        "--keep-low-confidence",
        action="store_true",
        help="No descartar muestras con baja confianza de lateralidad.",
    )
    return parser.parse_args()


def expected_coordinate_columns(prefix: str) -> list[str]:
    columns: list[str] = []
    for index in range(21):
        columns.extend(
            (
                f"{prefix}_x{index}",
                f"{prefix}_y{index}",
                f"{prefix}_z{index}",
            )
        )
    return columns


IMAGE_COLUMNS = expected_coordinate_columns("img")
WORLD_COLUMNS = expected_coordinate_columns("world")
EXPECTED_COLUMNS = set(METADATA_COLUMNS) | set(IMAGE_COLUMNS) | set(WORLD_COLUMNS)


def parse_landmarks(row: dict[str, str], columns: Iterable[str]) -> np.ndarray:
    values: list[float] = []

    for column in columns:
        raw_value = row.get(column, "")
        if raw_value == "":
            raise ValueError(f"Valor vacío en {column}")

        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"Valor no finito en {column}")

        values.append(value)

    array = np.asarray(values, dtype=np.float32).reshape(21, 3)
    return array


def normalize_landmarks(landmarks: np.ndarray) -> tuple[np.ndarray, float]:
    """Centra en la muñeca y escala usando el tamaño medio de la palma."""
    if landmarks.shape != (21, 3):
        raise ValueError(f"Forma inesperada: {landmarks.shape}")

    centered = landmarks - landmarks[0]

    palm_vectors = centered[list(PALM_MCP_INDICES)]
    palm_distances = np.linalg.norm(palm_vectors, axis=1)
    scale = float(np.mean(palm_distances))

    if not math.isfinite(scale) or scale <= EPSILON:
        raise ValueError("No se puede calcular una escala válida para la mano")

    normalized = centered / scale
    return normalized.astype(np.float32), scale


def handedness_features(handedness: str) -> tuple[float, float]:
    handedness = handedness.strip().upper()

    if handedness == "LEFT":
        return 1.0, 0.0
    if handedness == "RIGHT":
        return 0.0, 1.0

    raise ValueError(f"Lateralidad no válida: {handedness!r}")


def build_feature_names() -> list[str]:
    names: list[str] = []

    for prefix in ("img_norm", "world_norm"):
        for index in range(21):
            names.extend(
                (
                    f"{prefix}_x{index}",
                    f"{prefix}_y{index}",
                    f"{prefix}_z{index}",
                )
            )

    names.extend(("hand_LEFT", "hand_RIGHT"))
    return names


def discover_files(project_root: Path, pattern: str) -> list[Path]:
    pattern_path = Path(pattern)

    if pattern_path.is_absolute():
        base = pattern_path.anchor
        relative_pattern = str(pattern_path)[len(base):].lstrip("/\\")
        files = sorted(Path(base).glob(relative_pattern))
    else:
        files = sorted(project_root.glob(pattern))

    return [path for path in files if path.is_file()]


def main() -> int:
    args = parse_args()

    if not 0.0 <= args.min_handedness_score <= 1.0:
        print("--min-handedness-score debe estar entre 0 y 1.", file=sys.stderr)
        return 2

    project_root = Path(__file__).resolve().parents[1]
    input_files = discover_files(project_root, args.input_pattern)

    if not input_files:
        print(
            f"No se encontraron CSV con el patrón: {args.input_pattern}",
            file=sys.stderr,
        )
        return 1

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = build_feature_names()

    features: list[np.ndarray] = []
    labels: list[str] = []
    subjects: list[str] = []
    sessions: list[str] = []
    timestamps: list[str] = []
    handedness_values: list[str] = []
    source_files: list[str] = []

    rejected_reasons: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    session_counts: Counter[str] = Counter()
    hand_counts: Counter[str] = Counter()
    accepted_by_file: Counter[str] = Counter()

    total_rows = 0

    for csv_path in input_files:
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames:
                rejected_reasons[f"{csv_path.name}:sin_cabecera"] += 1
                continue

            missing = EXPECTED_COLUMNS - set(reader.fieldnames)
            if missing:
                print(
                    f"Se omite {csv_path.name}: faltan {len(missing)} columnas.",
                    file=sys.stderr,
                )
                rejected_reasons[f"{csv_path.name}:columnas_ausentes"] += 1
                continue

            for row in reader:
                total_rows += 1

                try:
                    label = (row.get("label") or "").strip().upper()
                    subject = (row.get("subject") or "").strip()
                    session = (row.get("session") or "").strip()
                    timestamp = (row.get("timestamp_utc") or "").strip()
                    handedness = (row.get("handedness") or "").strip().upper()
                    handedness_score = float(row.get("handedness_score") or "")

                    if label not in VALID_LABELS:
                        raise ValueError("etiqueta_no_valida")

                    if not subject:
                        raise ValueError("subject_vacio")

                    if not session:
                        raise ValueError("session_vacia")

                    if not math.isfinite(handedness_score):
                        raise ValueError("confianza_no_finita")

                    if (
                        not args.keep_low_confidence
                        and handedness_score < args.min_handedness_score
                    ):
                        raise ValueError("baja_confianza_lateralidad")

                    image_landmarks = parse_landmarks(row, IMAGE_COLUMNS)
                    world_landmarks = parse_landmarks(row, WORLD_COLUMNS)

                    image_normalized, _ = normalize_landmarks(image_landmarks)
                    world_normalized, _ = normalize_landmarks(world_landmarks)

                    hand_left, hand_right = handedness_features(handedness)

                    feature_vector = np.concatenate(
                        (
                            image_normalized.reshape(-1),
                            world_normalized.reshape(-1),
                            np.asarray((hand_left, hand_right), dtype=np.float32),
                        )
                    ).astype(np.float32)

                    if feature_vector.shape != (128,):
                        raise ValueError(
                            f"numero_caracteristicas_{feature_vector.shape[0]}"
                        )

                    if not np.all(np.isfinite(feature_vector)):
                        raise ValueError("caracteristicas_no_finitas")

                except (TypeError, ValueError) as error:
                    rejected_reasons[str(error)] += 1
                    continue

                features.append(feature_vector)
                labels.append(label)
                subjects.append(subject)
                sessions.append(session)
                timestamps.append(timestamp)
                handedness_values.append(handedness)
                source_files.append(csv_path.name)

                class_counts[label] += 1
                subject_counts[subject] += 1
                session_counts[f"{subject}:{session}"] += 1
                hand_counts[handedness] += 1
                accepted_by_file[csv_path.name] += 1

    if not features:
        print("No quedó ninguna muestra válida tras el procesamiento.", file=sys.stderr)
        return 1

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels, dtype=str)
    subject_array = np.asarray(subjects, dtype=str)
    session_array = np.asarray(sessions, dtype=str)
    timestamp_array = np.asarray(timestamps, dtype=str)
    handedness_array = np.asarray(handedness_values, dtype=str)
    source_file_array = np.asarray(source_files, dtype=str)
    feature_name_array = np.asarray(feature_names, dtype=str)

    npz_path = output_dir / "gesture_dataset.npz"
    csv_output_path = output_dir / "gesture_dataset.csv"
    summary_path = output_dir / "dataset_summary.json"

    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        subject=subject_array,
        session=session_array,
        timestamp_utc=timestamp_array,
        handedness=handedness_array,
        source_file=source_file_array,
        feature_names=feature_name_array,
    )

    output_header = [
        "timestamp_utc",
        "subject",
        "session",
        "label",
        "handedness",
        "source_file",
        *feature_names,
    ]

    with csv_output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(output_header)

        for index in range(X.shape[0]):
            writer.writerow(
                [
                    timestamps[index],
                    subjects[index],
                    sessions[index],
                    labels[index],
                    handedness_values[index],
                    source_files[index],
                    *X[index].tolist(),
                ]
            )

    summary = {
        "input_pattern": args.input_pattern,
        "input_files": [str(path.relative_to(project_root)) for path in input_files],
        "total_rows_read": total_rows,
        "accepted_rows": int(X.shape[0]),
        "rejected_rows": int(total_rows - X.shape[0]),
        "number_of_features": int(X.shape[1]),
        "normalization": {
            "translation": "landmark 0 (wrist) is moved to origin",
            "scale": "mean 3D distance from wrist to MCP landmarks 5, 9, 13 and 17",
            "orientation": "preserved",
            "handedness": "two one-hot features: hand_LEFT and hand_RIGHT",
        },
        "class_counts": dict(class_counts),
        "subject_counts": dict(subject_counts),
        "session_counts": dict(session_counts),
        "handedness_counts": dict(hand_counts),
        "accepted_by_file": dict(accepted_by_file),
        "rejected_reasons": dict(rejected_reasons),
        "outputs": {
            "npz": str(npz_path.relative_to(project_root)),
            "csv": str(csv_output_path.relative_to(project_root)),
        },
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("=" * 68)
    print("GESTUREBOT — PREPARACIÓN DEL DATASET")
    print("=" * 68)
    print("Archivos leídos:")
    for path in input_files:
        print(f"  - {path.relative_to(project_root)}")
    print()
    print(f"Filas leídas:     {total_rows}")
    print(f"Filas aceptadas:  {X.shape[0]}")
    print(f"Filas rechazadas: {total_rows - X.shape[0]}")
    print(f"Características:  {X.shape[1]}")
    print()
    print("Clases:", dict(class_counts))
    print("Personas:", dict(subject_counts))
    print("Sesiones:", dict(session_counts))
    print("Manos:", dict(hand_counts))
    print()
    print(f"NPZ:     {npz_path.relative_to(project_root)}")
    print(f"CSV:     {csv_output_path.relative_to(project_root)}")
    print(f"Resumen: {summary_path.relative_to(project_root)}")
    print("=" * 68)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
