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
    sample_every: int,
) -> None:
    height, width = frame.shape[:2]

    panel_x = 20
    panel_y = 20
    panel_w = min(460, max(370, width // 3 + 35))
    footer_h = 84
    footer_y = height - footer_h - 20
    panel_h = min(540, footer_y - panel_y - 14)

    total_samples = sum(
        int(counts[label])
        for label in ("FORWARD", "STOP", "BACKWARD", "GRIPPER", "OTHER")
    )
    max_count = max(
        1,
        max(
            (
                counts[label]
                for label in ("FORWARD", "STOP", "BACKWARD", "GRIPPER", "OTHER")
            ),
            default=1,
        ),
    )

    # Palette (BGR)
    bg = (16, 23, 34)
    border = (63, 78, 104)
    accent = (210, 96, 38)
    text_main = (245, 248, 252)
    text_muted = (176, 188, 208)
    card_bg = (27, 36, 50)
    track = (42, 49, 62)
    selected_border = (96, 165, 250)
    state_color = (37, 99, 235) if recording else (59, 130, 246)
    pill_text = "RECORDING" if recording else "PAUSED"

    palette = {
        "FORWARD": (246, 130, 59),
        "STOP": (99, 102, 241),
        "BACKWARD": (196, 113, 248),
        "GRIPPER": (45, 212, 191),
        "OTHER": (8, 179, 234),
    }

    def fit_text(value: str, limit: int = 24) -> str:
        value = str(value)
        return value if len(value) <= limit else value[: limit - 3] + "..."

    def put_centered(
        text_value: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        scale: float,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        (tw, th), _ = cv2.getTextSize(
            text_value,
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            thickness,
        )
        tx = x1 + max(0, (x2 - x1 - tw) // 2)
        ty = y1 + max(th + 4, (y2 - y1 + th) // 2)
        cv2.putText(
            frame,
            text_value,
            (tx, ty),
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    # Background cards.
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        bg,
        -1,
    )
    cv2.rectangle(
        overlay,
        (20, footer_y),
        (width - 20, footer_y + footer_h),
        bg,
        -1,
    )
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        border,
        1,
    )
    cv2.rectangle(
        frame,
        (20, footer_y),
        (width - 20, footer_y + footer_h),
        border,
        1,
    )

    # Header.
    header_h = 48
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + header_h),
        accent,
        -1,
    )
    put_centered(
        "GESTUREBOT",
        panel_x,
        panel_y,
        panel_x + panel_w,
        panel_y + header_h,
        0.90,
        (255, 255, 255),
        2,
    )

    # Recording state.
    pill_w = 182
    pill_h = 42
    pill_x = panel_x + panel_w - pill_w - 18
    pill_y = panel_y + header_h + 18
    cv2.rectangle(
        frame,
        (pill_x, pill_y),
        (pill_x + pill_w, pill_y + pill_h),
        state_color,
        -1,
    )
    put_centered(
        pill_text,
        pill_x,
        pill_y,
        pill_x + pill_w,
        pill_y + pill_h,
        0.82,
        (255, 255, 255),
        2,
    )

    # Identification data. The explicit coordinates keep LABEL clear of the card below.
    info_x = panel_x + 18
    info_rows = [
        ("SUBJECT", fit_text(subject, 18), panel_y + 82),
        ("SESSION", fit_text(session, 18), panel_y + 154),
        ("LABEL", fit_text(selected_label, 18), panel_y + 226),
    ]
    for title, value, y in info_rows:
        cv2.putText(
            frame,
            title,
            (info_x, y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.45,
            text_muted,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            value,
            (info_x, y + 34),
            cv2.FONT_HERSHEY_DUPLEX,
            0.78,
            text_main,
            1,
            cv2.LINE_AA,
        )

    # Samples and hand status. The status already arrives as RIGHT | C: 99%.
    stats_y = panel_y + 282
    cv2.rectangle(
        frame,
        (panel_x + 14, stats_y),
        (panel_x + panel_w - 14, stats_y + 66),
        card_bg,
        -1,
    )
    cv2.putText(
        frame,
        "TOTAL SAMPLES",
        (panel_x + 24, stats_y + 19),
        cv2.FONT_HERSHEY_DUPLEX,
        0.43,
        text_muted,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        str(total_samples),
        (panel_x + 24, stats_y + 50),
        cv2.FONT_HERSHEY_DUPLEX,
        0.92,
        text_main,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"HAND: {fit_text(hand_status, 20)}",
        (panel_x + 126, stats_y + 45),
        cv2.FONT_HERSHEY_DUPLEX,
        0.46,
        text_main,
        1,
        cv2.LINE_AA,
    )

    # Counts.
    section_y = stats_y + 92
    cv2.putText(
        frame,
        "CLASS COUNTS",
        (panel_x + 14, section_y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.46,
        text_muted,
        1,
        cv2.LINE_AA,
    )
    bar_y = section_y + 18
    bar_x = panel_x + 150
    bar_w = panel_w - 192
    bar_h = 11

    for label in ("FORWARD", "STOP", "BACKWARD", "GRIPPER", "OTHER"):
        count = int(counts[label])
        cv2.putText(
            frame,
            label,
            (panel_x + 18, bar_y + 11),
            cv2.FONT_HERSHEY_DUPLEX,
            0.50,
            text_main,
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            track,
            -1,
        )
        filled = int(bar_w * (count / max_count))
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + filled, bar_y + bar_h),
            palette[label],
            -1,
        )
        cv2.putText(
            frame,
            str(count),
            (panel_x + panel_w - 34, bar_y + 11),
            cv2.FONT_HERSHEY_DUPLEX,
            0.46,
            text_main,
            1,
            cv2.LINE_AA,
        )
        if label == selected_label:
            cv2.rectangle(
                frame,
                (panel_x + 10, bar_y - 7),
                (panel_x + panel_w - 10, bar_y + 20),
                selected_border,
                1,
            )
        bar_y += 24

    # Footer shortcuts.
    cv2.putText(
        frame,
        "SHORTCUTS",
        (34, footer_y + 22),
        cv2.FONT_HERSHEY_DUPLEX,
        0.46,
        text_muted,
        1,
        cv2.LINE_AA,
    )
    footer_lines = [
        "1 Forward   2 Stop   3 Backward   4 Gripper   0 Other",
        "R Record / Pause   Q or ESC Exit",
        f"Save 1 sample every {max(1, sample_every)} frames",
    ]
    for i, line in enumerate(footer_lines):
        cv2.putText(
            frame,
            line,
            (34, footer_y + 45 + i * 16),
            cv2.FONT_HERSHEY_DUPLEX,
            0.39,
            text_main,
            1,
            cv2.LINE_AA,
        )



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
    cv2.namedWindow("GestureBot", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GestureBot", 1280, 720)

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
                            sample_every=args.sample_every,
                        )
                        cv2.imshow("GestureBot — Dataset Studio", frame)
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
                        f"{handedness} | C: {handedness_score * 100:.0f}%"
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
                    sample_every=args.sample_every,
                )

                cv2.imshow("GestureBot — Dataset Studio", frame)
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
