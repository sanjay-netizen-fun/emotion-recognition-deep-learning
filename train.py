#!/usr/bin/env python3
"""Train the emotion CNN from a folder-per-class dataset.

The data layout matches ``trainmodel.ipynb``::

    images/
      train/<emotion>/*.jpg
      validation/<emotion>/*.jpg     (or test/)

Quick smoke test on a subset::

    python train.py --data images --train-dir images/train \
        --val-dir images/validation --limit 280 --epochs 2 --output /tmp/smoke.keras

Full training run::

    python train.py --data images --epochs 40 --output emotiondetector.keras
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from emotion_recognition.dataset import scan_dataset  # noqa: E402
from emotion_recognition.labels import EMOTIONS  # noqa: E402
from emotion_recognition.metrics import classification_metrics  # noqa: E402
from emotion_recognition.training import build_model, load_sample_images, one_hot  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the emotion recognition CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=Path("images"), help="dataset root")
    parser.add_argument("--train-dir", type=Path, default=None, help="explicit training split")
    parser.add_argument("--val-dir", type=Path, default=None, help="explicit validation split")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.0,
        help="hold out this fraction of the training data when no validation split exists",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5, help="early-stopping patience")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "emotiondetector.keras",
        help="where to save the trained model",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of training images (balanced across classes)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true", help="disable tqdm progress bars")
    return parser.parse_args(argv)


def resolve_train_val(args: argparse.Namespace):
    train_dir = args.train_dir or (
        args.data / "train" if (args.data / "train").is_dir() else args.data
    )
    val_dir = args.val_dir
    if val_dir is None:
        for candidate in ("validation", "test"):
            if (args.data / candidate).is_dir():
                val_dir = args.data / candidate
                break

    train_samples = scan_dataset(train_dir, limit=args.limit, shuffle=True, seed=args.seed)

    if val_dir is not None:
        val_samples = scan_dataset(val_dir, limit=args.limit, shuffle=True, seed=args.seed + 1)
        return train_samples, val_samples

    if not 0.0 < args.validation_split < 1.0:
        raise SystemExit(
            "error: no validation split found; pass --val-dir or --validation-split 0.1"
        )

    rng = random.Random(args.seed)
    rng.shuffle(train_samples)
    cut = max(1, int(len(train_samples) * args.validation_split))
    return train_samples[cut:], train_samples[:cut]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import keras
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    keras.utils.set_random_seed(args.seed)

    try:
        train_samples, val_samples = resolve_train_val(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print("Emotion Recognition — training")
    print("=" * 60)
    print(f"train images : {len(train_samples)}")
    print(f"val images   : {len(val_samples)}")
    print(f"output       : {args.output}")

    started = time.perf_counter()
    x_train, y_train, _ = load_sample_images(train_samples, progress=not args.no_progress)
    x_val, y_val, _ = load_sample_images(val_samples, progress=not args.no_progress)
    print(f"loaded       : {x_train.shape} / {x_val.shape} in {time.perf_counter() - started:.1f}s")

    model = build_model(num_classes=len(EMOTIONS), dropout=args.dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        ModelCheckpoint(
            filepath=str(args.output),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        one_hot(y_train),
        validation_data=(x_val, one_hot(y_val)),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # Make sure the final weights (best, thanks to restore_best_weights) are stored.
    model.save(args.output)
    architecture_path = args.output.with_suffix(".json")
    architecture_path.write_text(model.to_json(), encoding="utf-8")

    probabilities = model.predict(x_val, batch_size=args.batch_size, verbose=0)
    metrics = classification_metrics(y_val, probabilities.argmax(axis=1), EMOTIONS)
    print("\nValidation metrics")
    print(metrics.report())

    summary = {
        "epochs_requested": args.epochs,
        "epochs_run": len(history.history.get("loss", [])),
        "train_images": int(x_train.shape[0]),
        "val_images": int(x_val.shape[0]),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "seed": args.seed,
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
        "final_val_accuracy": metrics.accuracy,
        "model": str(args.output),
    }
    summary_path = args.output.with_suffix(".training.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nSaved model          : {args.output}")
    print(f"Saved architecture   : {architecture_path}")
    print(f"Saved training report: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
