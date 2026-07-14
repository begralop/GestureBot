"""Audita un CSV de landmarks de GestureBot sin modificarlo.

Ejemplo:
    python src/04_check_dataset.py data/raw/belen_s1.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path


VALID_LABELS = {"FORWARD", "STOP", "BACKWARD", "GRIPPER", "OTHER"}
METADATA_COLUMNS = {
    "timestamp_utc",
    "subject",
    "session",
    "label",
    "handedness",
    "handedness_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprueba estructura, equilibrio y valores de un dataset de GestureBot."
    )
    parser.add_argument("csv_path", type=Path, help="Ruta al CSV que se quiere comprobar.")
    return parser.parse_args()


def feature_columns(fieldnames: list[str]) -> list[str]:
    return [name for name in fieldnames if name not in METADATA_COLUMNS]


def main() -> int:
    args = parse_args()
    csv_path = args.csv_path

    if not csv_path.exists():
        print(f"No existe el archivo: {csv_path}", file=sys.stderr)
        return 1

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            print("El CSV no contiene cabecera.", file=sys.stderr)
            return 1

        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    missing_metadata = sorted(METADATA_COLUMNS - set(fieldnames))
    features = feature_columns(fieldnames)

    labels = Counter()
    subjects = Counter()
    sessions = Counter()
    hands = Counter()

    empty_cells = 0
    invalid_numeric_cells = 0
    nonfinite_numeric_cells = 0
    low_handedness_rows = 0
    invalid_label_rows = 0

    exact_feature_rows = Counter()

    coordinate_min: dict[str, float] = {}
    coordinate_max: dict[str, float] = {}

    for row in rows:
        label = (row.get("label") or "").strip()
        subject = (row.get("subject") or "").strip()
        session = (row.get("session") or "").strip()
        hand = (row.get("handedness") or "").strip()

        labels[label] += 1
        subjects[subject] += 1
        sessions[session] += 1
        hands[hand] += 1

        if label not in VALID_LABELS:
            invalid_label_rows += 1

        try:
            handedness_score = float(row.get("handedness_score") or "")
            if handedness_score < 0.5:
                low_handedness_rows += 1
        except ValueError:
            invalid_numeric_cells += 1

        feature_signature: list[str] = []

        for column in features:
            raw_value = row.get(column)

            if raw_value is None or raw_value == "":
                empty_cells += 1
                feature_signature.append("")
                continue

            feature_signature.append(raw_value)

            try:
                value = float(raw_value)
            except ValueError:
                invalid_numeric_cells += 1
                continue

            if not math.isfinite(value):
                nonfinite_numeric_cells += 1
                continue

            coordinate_min[column] = min(coordinate_min.get(column, value), value)
            coordinate_max[column] = max(coordinate_max.get(column, value), value)

        exact_feature_rows[tuple(feature_signature)] += 1

    duplicate_rows = sum(count - 1 for count in exact_feature_rows.values() if count > 1)

    print("=" * 64)
    print("GESTUREBOT — AUDITORÍA DEL DATASET")
    print("=" * 64)
    print(f"Archivo: {csv_path.resolve()}")
    print(f"Filas: {len(rows)}")
    print(f"Columnas: {len(fieldnames)}")
    print(f"Características numéricas esperadas/obtenidas: 126/{len(features)}")
    print()

    print("Clases:")
    for label in ("FORWARD", "STOP", "BACKWARD", "GRIPPER", "OTHER"):
        print(f"  {label:10s}: {labels[label]}")

    print()
    print("Personas:", dict(subjects))
    print("Sesiones:", dict(sessions))
    print("Manos:", dict(hands))
    print()

    print("Calidad:")
    print(f"  Metadatos ausentes: {missing_metadata or 'ninguno'}")
    print(f"  Celdas vacías en características: {empty_cells}")
    print(f"  Valores numéricos inválidos: {invalid_numeric_cells}")
    print(f"  Valores no finitos (NaN/inf): {nonfinite_numeric_cells}")
    print(f"  Filas con etiqueta no válida: {invalid_label_rows}")
    print(f"  Filas con confianza de lateralidad < 0.5: {low_handedness_rows}")
    print(f"  Filas exactamente duplicadas en características: {duplicate_rows}")
    print()

    warnings: list[str] = []

    if len(fieldnames) != 132:
        warnings.append(f"Se esperaban 132 columnas y hay {len(fieldnames)}.")

    if len(features) != 126:
        warnings.append(f"Se esperaban 126 características y hay {len(features)}.")

    if missing_metadata:
        warnings.append("Faltan columnas de metadatos.")

    if empty_cells or invalid_numeric_cells or nonfinite_numeric_cells:
        warnings.append("Hay valores vacíos o numéricos no válidos.")

    if invalid_label_rows:
        warnings.append("Hay etiquetas fuera del conjunto permitido.")

    class_counts = [labels[label] for label in VALID_LABELS if labels[label] > 0]
    if len(class_counts) < len(VALID_LABELS):
        warnings.append("Falta al menos una de las cinco clases.")
    elif max(class_counts) / min(class_counts) > 1.5:
        warnings.append("El dataset está algo desequilibrado entre clases.")

    if labels["OTHER"] < max(labels[label] for label in VALID_LABELS - {"OTHER"}):
        warnings.append(
            "OTHER debería tener al menos tantas muestras como la clase positiva más grande."
        )

    if duplicate_rows > len(rows) * 0.1:
        warnings.append("Hay más de un 10 % de filas exactamente duplicadas.")

    print("Resultado:")
    if warnings:
        for warning in warnings:
            print(f"  AVISO: {warning}")
    else:
        print("  OK: la estructura y los valores básicos parecen correctos.")

    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
