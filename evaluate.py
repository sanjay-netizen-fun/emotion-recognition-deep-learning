#!/usr/bin/env python3
"""Evaluate the trained emotion model on a labelled image folder.

Examples
--------
Evaluate the bundled model on the FER-style test split::

    python evaluate.py --data images --split test

Evaluate a custom checkpoint on a flat folder of class sub-folders, running
Haar-cascade face detection first (like the desktop app)::

    python evaluate.py --data my_faces --model checkpoints/best.keras --detect

Write machine-readable output and fail if accuracy drops below a threshold::

    python evaluate.py --data images --split test --json runs/eval.json \
        --csv runs/predictions.csv --min-accuracy 0.85
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from emotion_recognition.dataset import scan_dataset  # noqa: E402
from emotion_recognition.detection import FaceDetector  # noqa: E402
from emotion_recognition.labels import EMOTIONS  # noqa: E402
from emotion_recognition.metrics import classification_metrics, top_k_accuracy  # noqa: E402
from emotion_recognition.model import EmotionRecognizer, default_model_path  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the emotion recognition model on a labelled dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("images"),
        help="dataset root (either images/<split>/<class> or <class>/... folders)",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=["train", "validation", "test"],
        help="which split to evaluate when the dataset root contains train/, validation/ or test/",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=f"path to a .keras model (default: {default_model_path()})",
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="run Haar face detection on each image before classification",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="with --detect, skip images where no face is found instead of using the whole frame",
    )
    parser.add_argument("--limit", type=int, default=None, help="evaluate at most N images")
    parser.add_argument("--shuffle", action="store_true", help="shuffle samples before evaluating")
    parser.add_argument("--seed", type=int, default=0, help="seed for shuffling/limiting")
    parser.add_argument("--batch-size", type=int, default=64, help="inference batch size")
    parser.add_argument("--json", type=Path, default=None, help="write a JSON report here")
    parser.add_argument("--csv", type=Path, default=None, help="write per-image predictions here")
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="write a Markdown report (tables + confusion matrix) here",
    )
    parser.add_argument(
        "--errors",
        type=int,
        default=10,
        help="how many misclassified examples to print (0 disables)",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="exit with status 1 if accuracy is below this value",
    )
    return parser.parse_args(argv)


def load_gray_image(path: Path) -> np.ndarray | None:
    """Read an image as grayscale (matching the training pipeline)."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        # Fall back to colour + conversion for formats OpenCV cannot read as gray.
        colour = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if colour is None:
            return None
        return cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
    return image


def format_confusion(matrix: np.ndarray, labels: tuple[str, ...]) -> str:
    short = [label[:3] for label in labels]
    width = max(5, max(len(name) for name in short) + 1)
    header = "true\\pred".ljust(width + 2) + "".join(name.rjust(width) for name in short)
    lines = [header, "-" * len(header)]
    for row_index, name in enumerate(short):
        row = "".join(str(int(value)).rjust(width) for value in matrix[row_index])
        lines.append(name.ljust(width + 2) + row)
    return "\n".join(lines)


def format_confusion_markdown(matrix: np.ndarray, labels: tuple[str, ...]) -> str:
    header = "| true \\ predicted | " + " | ".join(labels) + " |"
    divider = "|" + "---|" * (len(labels) + 1)
    rows = [header, divider]
    for index, name in enumerate(labels):
        cells = " | ".join(str(int(value)) for value in matrix[index])
        rows.append(f"| **{name}** | {cells} |")
    return "\n".join(rows)


def most_confused_pairs(matrix: np.ndarray, labels: tuple[str, ...], top: int = 5) -> list[dict]:
    pairs = []
    for true_index in range(len(labels)):
        for pred_index in range(len(labels)):
            if true_index == pred_index:
                continue
            count = int(matrix[true_index, pred_index])
            if count:
                pairs.append(
                    {"true": labels[true_index], "predicted": labels[pred_index], "count": count}
                )
    pairs.sort(key=lambda item: item["count"], reverse=True)
    return pairs[:top]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("Emotion Recognition — evaluation")
    print("=" * 60)

    try:
        samples = scan_dataset(
            args.data,
            split=args.split,
            limit=args.limit,
            shuffle=args.shuffle,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"dataset      : {args.data}")
    print(f"samples      : {len(samples)}")

    class_counts = Counter(sample.label for sample in samples)
    print("classes      : " + ", ".join(f"{name}={class_counts.get(name, 0)}" for name in EMOTIONS))

    model_path = args.model or default_model_path()
    print(f"model        : {model_path}")
    print(f"detection    : {'haar cascade' if args.detect else 'whole image (no detection)'}")

    try:
        recognizer = EmotionRecognizer(model_path, warmup=args.detect is False)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    detector = None
    if args.detect:
        try:
            detector = FaceDetector()
        except (FileNotFoundError, ValueError) as error:
            print(f"error: face detection unavailable: {error}", file=sys.stderr)
            return 2
    fallback = not args.no_fallback

    y_true: list[int] = []
    y_pred: list[int] = []
    all_scores: list[np.ndarray] = []
    predictions: list[dict] = []
    skipped: list[dict] = []
    images_with_faces = 0
    used_fallback = 0

    started = time.perf_counter()
    batch_faces: list[np.ndarray] = []
    batch_meta: list[tuple] = []

    from emotion_recognition.detection import (
        crop_box,  # local import keeps startup lean
    )

    def flush() -> None:
        nonlocal batch_faces, batch_meta
        if not batch_faces:
            return
        scores = recognizer.predict_batch(batch_faces)
        for (sample, boxes, fallback_used), score_row in zip(batch_meta, scores, strict=True):
            index = int(np.argmax(score_row))
            y_true.append(sample.index)
            y_pred.append(index)
            all_scores.append(score_row)
            predictions.append(
                {
                    "path": str(sample.path),
                    "true": sample.label,
                    "predicted": EMOTIONS[index],
                    "correct": EMOTIONS[index] == sample.label,
                    "confidence": round(float(score_row[index]), 6),
                    "faces": len(boxes),
                    "fallback": fallback_used,
                }
            )
        batch_faces = []
        batch_meta = []

    for number, sample in enumerate(samples, start=1):
        image = load_gray_image(sample.path)
        if image is None:
            skipped.append({"path": str(sample.path), "reason": "unreadable"})
            continue

        boxes: np.ndarray
        fallback_used = False
        if detector is not None:
            boxes = detector.detect(image)
            if len(boxes) == 0:
                if not fallback:
                    skipped.append({"path": str(sample.path), "reason": "no face detected"})
                    continue
                boxes = np.empty((0, 4), dtype=np.int32)
                fallback_used = True
            else:
                images_with_faces += 1
        else:
            boxes = np.empty((0, 4), dtype=np.int32)

        if fallback_used or len(boxes) == 0:
            face = image
            if detector is not None:
                used_fallback += 1
        else:
            # Evaluate on the largest detected face.
            areas = boxes[:, 2] * boxes[:, 3]
            face = crop_box(image, boxes[int(np.argmax(areas))])

        batch_faces.append(face)
        batch_meta.append((sample, boxes, fallback_used))

        if len(batch_faces) >= max(1, args.batch_size):
            flush()

        if number % 200 == 0 or number == len(samples):
            print(f"  processed {number}/{len(samples)}", end="\r", flush=True)

    flush()
    elapsed = time.perf_counter() - started

    if not y_true:
        print("\nerror: no samples could be evaluated", file=sys.stderr)
        return 2

    metrics = classification_metrics(y_true, y_pred, EMOTIONS)
    scores_array = np.vstack(all_scores)
    top2 = top_k_accuracy(scores_array, y_true, k=2)

    print("\n")
    print(metrics.report())
    print(f"\ntop-2 accuracy: {top2:.4f}")
    print(f"throughput    : {len(y_true) / elapsed:.1f} images/s ({elapsed:.1f}s total)")

    if detector is not None:
        print(f"faces detected: {images_with_faces}/{len(y_true) + len(skipped)} image(s)")
        if used_fallback:
            print(f"whole-frame   : {used_fallback} image(s) had no detected face")

    print("\nConfusion matrix (rows = true, columns = predicted)")
    print(format_confusion(metrics.matrix, metrics.labels))

    confused = most_confused_pairs(metrics.matrix, metrics.labels)
    if confused:
        print("\nMost confused pairs")
        for pair in confused:
            print(f"  {pair['true']:<10} -> {pair['predicted']:<10} {pair['count']}")

    mistakes = [row for row in predictions if not row["correct"]]
    if args.errors and mistakes:
        print(f"\nMisclassified examples ({min(args.errors, len(mistakes))} of {len(mistakes)})")
        for row in mistakes[: args.errors]:
            print(
                f"  {Path(row['path']).name:<28} true={row['true']:<9} "
                f"pred={row['predicted']:<9} conf={row['confidence']:.2f}"
            )

    if skipped:
        print(f"\nSkipped {len(skipped)} image(s); first 5:")
        for item in skipped[:5]:
            print(f"  {item['path']} ({item['reason']})")

    if args.json:
        report = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "dataset": str(args.data),
            "split": args.split,
            "model": str(model_path),
            "detection": "haar" if args.detect else "none",
            "fallback_to_frame": bool(fallback),
            "top2_accuracy": round(top2, 6),
            "throughput_images_per_second": round(len(y_true) / elapsed, 2),
            "skipped": skipped,
            "metrics": metrics.to_dict(),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON report written to {args.json}")

    if args.markdown:
        total = metrics.support
        lines = [
            "# Evaluation report",
            "",
            f"- **Dataset:** `{args.data}`" + (f" (split: `{args.split}`)" if args.split else ""),
            f"- **Images evaluated:** {total}",
            f"- **Model:** `{model_path.name}`",
            f"- **Face detection:** {'Haar cascade' if args.detect else 'none (whole image)'}",
            f"- **Accuracy:** {metrics.accuracy:.2%}",
            f"- **Balanced accuracy:** {metrics.balanced_accuracy:.2%}",
            f"- **Macro F1:** {metrics.macro_f1:.3f}",
            f"- **Weighted F1:** {metrics.weighted_f1:.3f}",
            f"- **Top-2 accuracy:** {top2:.2%}",
            f"- **Throughput:** {len(y_true) / elapsed:.0f} images/s (CPU)",
            "",
            "## Per-class metrics",
            "",
            "| Emotion | Precision | Recall | F1 | Support |",
            "|---|---|---|---|---|",
        ]
        for row in metrics.per_class:
            lines.append(
                f"| {row['label']} | {row['precision']:.3f} | {row['recall']:.3f} "
                f"| {row['f1']:.3f} | {row['support']} |"
            )
        lines += [
            "",
            "## Confusion matrix",
            "",
            format_confusion_markdown(metrics.matrix, metrics.labels),
            "",
            "## Most confused pairs",
            "",
            "| True | Predicted | Count |",
            "|---|---|---|",
        ]
        for pair in confused:
            lines.append(f"| {pair['true']} | {pair['predicted']} | {pair['count']} |")
        lines.append("")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(lines), encoding="utf-8")
        print(f"Markdown report written to {args.markdown}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["path", "true", "predicted", "correct", "confidence", "faces", "fallback"],
            )
            writer.writeheader()
            writer.writerows(predictions)
        print(f"Predictions written to {args.csv}")

    if args.min_accuracy is not None and metrics.accuracy < args.min_accuracy:
        print(
            f"\nFAIL: accuracy {metrics.accuracy:.4f} is below threshold {args.min_accuracy:.4f}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
