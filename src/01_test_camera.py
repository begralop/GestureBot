"""Prueba mínima de la webcam. Salir con q o ESC."""
from __future__ import annotations
import argparse
import sys
import cv2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara {args.camera}.", file=sys.stderr)
        return 1
    print("Webcam abierta. Pulsa q o ESC para salir.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("No se pudo leer un frame.", file=sys.stderr)
                return 1
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "GestureBot - webcam | q/ESC para salir", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("GestureBot - Webcam", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
