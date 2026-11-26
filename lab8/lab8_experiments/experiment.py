"""Reproducible Lab 8 model experiments.

Run from the lab8 directory through the repository's Nix shell, for example:

    nix develop -c python lab8_experiments/experiment.py \
        --model mobilenet_unet --epochs 40 --output lab8_experiments/runs/mobilenet

The script keeps the official 80/20 split and seed from lab8_tasks.py, trains a
candidate segmentation model, and reports both global and per-breed IoU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

LAB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_DIR))

from lab8_tasks import DataLoader, IoUScore


def conv_block(
    inputs: tf.Tensor,
    filters: int,
    *,
    dropout: float = 0.0,
    batch_norm: bool = True,
) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same", use_bias=not batch_norm)(inputs)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=not batch_norm)(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    return x


def build_unet(filters: int = 32) -> keras.Model:
    inputs = layers.Input((128, 128, 3))

    c1 = conv_block(inputs, filters)
    p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, filters * 2)
    p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, filters * 4)
    p3 = layers.MaxPooling2D()(c3)
    bottleneck = conv_block(p3, filters * 8, dropout=0.2)

    u3 = layers.Conv2DTranspose(filters * 4, 2, strides=2, padding="same")(bottleneck)
    d3 = conv_block(layers.Concatenate()([u3, c3]), filters * 4)
    u2 = layers.Conv2DTranspose(filters * 2, 2, strides=2, padding="same")(d3)
    d2 = conv_block(layers.Concatenate()([u2, c2]), filters * 2)
    u1 = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(d2)
    d1 = conv_block(layers.Concatenate()([u1, c1]), filters)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d1)
    return keras.Model(inputs, outputs, name="unet")


def build_mobilenet_unet(
    trainable_backbone: bool = False,
    weights: str | None = "imagenet",
) -> keras.Model:
    backbone = keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights=weights,
    )
    backbone.trainable = trainable_backbone

    skip_names = [
        "block_1_expand_relu",  # 64 x 64
        "block_3_expand_relu",  # 32 x 32
        "block_6_expand_relu",  # 16 x 16
        "block_13_expand_relu",  # 8 x 8
    ]
    skips = [backbone.get_layer(name).output for name in skip_names]
    x = backbone.get_layer("block_16_project").output  # 4 x 4

    for filters, skip in zip((512, 256, 128, 64), reversed(skips)):
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, filters)

    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = conv_block(x, 32, dropout=0.1)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    model = keras.Model(backbone.input, outputs, name="mobilenet_unet")
    object.__setattr__(
        model,
        "_backbone_layer_names",
        tuple(layer.name for layer in backbone.layers),
    )
    return model


def build_efficientnet_unet(
    trainable_backbone: bool = False,
    weights: str | None = "imagenet",
) -> keras.Model:
    backbone = keras.applications.EfficientNetB0(
        input_shape=(128, 128, 3),
        include_top=False,
        weights=weights,
    )
    backbone.trainable = trainable_backbone

    # (output name, spatial size) pairs used as decoder skips.
    skip_names = [
        "block2a_expand_activation",  # 64 x 64
        "block3a_expand_activation",  # 32 x 32
        "block4a_expand_activation",  # 16 x 16
        "block6a_expand_activation",  # 8 x 8
    ]
    skips = [backbone.get_layer(name).output for name in skip_names]
    x = backbone.get_layer("top_activation").output  # 4 x 4

    for filters, skip in zip((256, 192, 128, 64), reversed(skips)):
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("swish")(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, filters, dropout=0.15)

    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = conv_block(x, 32)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    model = keras.Model(backbone.input, outputs, name="efficientnet_unet")
    object.__setattr__(
        model,
        "_backbone_layer_names",
        tuple(layer.name for layer in backbone.layers),
    )
    return model


def make_augmenter() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(
                0.07,
                0.07,
                fill_mode="constant",
                fill_value=0.0,
                interpolation="nearest",
            ),
            layers.RandomZoom(
                0.08,
                fill_mode="constant",
                fill_value=0.0,
                interpolation="nearest",
            ),
            layers.RandomRotation(
                0.025,
                fill_mode="constant",
                fill_value=0.0,
                interpolation="nearest",
            ),
        ],
        name="mask_aware_augmentation",
    )


def build_datasets(
    loader: DataLoader,
    batch_size: int,
    augment: bool,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train = tf.data.Dataset.from_tensor_slices((loader.X_train, loader.y_train))
    train = train.shuffle(len(loader.X_train), seed=42, reshuffle_each_iteration=True)
    train = train.batch(batch_size)

    val = tf.data.Dataset.from_tensor_slices((loader.X_val, loader.y_val)).batch(batch_size)

    if augment:
        augmenter = make_augmenter()

        def augment_batch(images: tf.Tensor, masks: tf.Tensor):
            combined = augmenter(tf.concat([images, masks], axis=-1), training=True)
            images = tf.clip_by_value(combined[..., :3], 0.0, 1.0)
            masks = tf.cast(combined[..., 3:] > 0.5, tf.float32)
            images = tf.image.random_brightness(images, max_delta=0.08)
            images = tf.image.random_contrast(images, 0.9, 1.1)
            images = tf.clip_by_value(images, 0.0, 1.0)
            return images, masks

        train = train.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)

    return train.prefetch(tf.data.AUTOTUNE), val.prefetch(tf.data.AUTOTUNE)


def build_model(name: str, imagenet_weights: bool = True) -> keras.Model:
    weights = "imagenet" if imagenet_weights else None
    if name == "unet":
        return build_unet()
    if name == "mobilenet_unet":
        return build_mobilenet_unet(weights=weights)
    if name == "efficientnet_unet":
        return build_efficientnet_unet(weights=weights)
    raise ValueError(f"Unknown model: {name}")


def backbone_layers(model: keras.Model) -> list[layers.Layer]:
    names = set(getattr(model, "_backbone_layer_names", ()))
    return [layer for layer in model.layers if layer.name in names]


def set_backbone_trainable(model: keras.Model, trainable: bool) -> None:
    for layer in backbone_layers(model):
        layer.trainable = trainable


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-4,
            clipnorm=1.0,
        ),
        loss=weighted_binary_crossentropy,
        metrics=[IoUScore()],
    )


def weighted_binary_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """BCE with extra weight on the smaller foreground region."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
    foreground = tf.reduce_mean(y_true, axis=(1, 2, 3))
    # Boundary/background dominates the image. Keep the weighting bounded so
    # unusual (nearly empty or nearly full) masks do not destabilize training.
    positive_weight = tf.clip_by_value((1.0 - foreground) / (foreground + 0.05), 1.0, 4.0)
    per_pixel = -(positive_weight[:, None, None, None] * y_true * tf.math.log(y_pred)
                  + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(per_pixel)


def evaluate_per_breed(
    model: keras.Model,
    loader: DataLoader,
    val_paths: list[tuple[Path, Path]],
    batch_size: int,
) -> dict:
    predictions = model.predict(loader.X_val, batch_size=batch_size, verbose=0)
    pred = predictions[..., 0] > 0.5
    true = loader.y_val[..., 0] > 0.5
    intersection = np.logical_and(pred, true).sum(axis=(1, 2), dtype=np.int64)
    union = np.logical_or(pred, true).sum(axis=(1, 2), dtype=np.int64)

    by_breed: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for (image_path, _), item_intersection, item_union in zip(val_paths, intersection, union):
        breed = image_path.stem.rsplit("_", 1)[0]
        by_breed[breed][0] += int(item_intersection)
        by_breed[breed][1] += int(item_union)
        by_breed[breed][2] += 1

    breeds = {}
    for breed, (item_intersection, item_union, count) in by_breed.items():
        breeds[breed] = {
            "iou": item_intersection / max(item_union, 1),
            "images": count,
            "intersection": item_intersection,
            "union": item_union,
        }

    ordered = sorted(breeds.items(), key=lambda item: item[1]["iou"])
    global_iou = float(intersection.sum() / max(union.sum(), 1))
    image_ious = intersection / np.maximum(union, 1)
    return {
        "global_iou": global_iou,
        "mean_image_iou": float(image_ious.mean()),
        "median_image_iou": float(np.median(image_ious)),
        "worst_breed": ordered[0][0],
        "worst_breed_iou": float(ordered[0][1]["iou"]),
        "best_breed": ordered[-1][0],
        "best_breed_iou": float(ordered[-1][1]["iou"]),
        "breeds_above_0_61": sum(item["iou"] > 0.61 for item in breeds.values()),
        "breed_count": len(breeds),
        "breeds": dict(ordered),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("unet", "mobilenet_unet", "efficientnet_unet"),
        default="mobilenet_unet",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=3e-4)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--imagenet-weights",
        choices=("imagenet", "none"),
        default="imagenet",
        help="Use ImageNet initialization (default) or train the backbone from scratch.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


class BestModelCheckpoint(keras.callbacks.Callback):
    """Persist the best val IoU checkpoint across any number of fit calls."""

    def __init__(self, filepath: Path) -> None:
        super().__init__()
        self.filepath = filepath
        self.best = -math.inf

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        current = float(logs.get("val_iou_score", -math.inf))
        if current > self.best:
            self.best = current
            self.model.save_weights(self.filepath)
            print(
                f"\nEpoch {epoch + 1}: val_iou_score improved to "
                f"{current:.6f}, saved {self.filepath.name}"
            )


def make_callbacks(
    output: Path,
    checkpoint: BestModelCheckpoint,
    *,
    append_csv: bool,
    patience: int,
) -> list[keras.callbacks.Callback]:
    return [
        checkpoint,
        keras.callbacks.CSVLogger(output / "history.csv", append=append_csv),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_iou_score",
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_iou_score",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run_start = time.time()

    keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()

    data_root = LAB_DIR / "lab8_work" / "data"
    image_dir = data_root / "images"
    trimap_dir = data_root / "annotations" / "trimaps"
    image_paths = sorted(image_dir.glob("*.jpg"))
    paired_paths = [(path, trimap_dir / f"{path.stem}.png") for path in image_paths]

    loader = DataLoader(image_size=(128, 128), batch_size=args.batch_size, seed=42)
    loader.process_data(paired_paths)

    shuffled_paths = list(paired_paths)
    np.random.default_rng(42).shuffle(shuffled_paths)
    val_paths = shuffled_paths[int(len(shuffled_paths) * 0.8) :]
    assert len(val_paths) == len(loader.X_val)

    train, val = build_datasets(loader, args.batch_size, augment=not args.no_augment)
    model = build_model(args.model, imagenet_weights=args.imagenet_weights == "imagenet")
    model(tf.zeros((1, 128, 128, 3)))
    model.summary()

    checkpoint_path = args.output / "best.weights.h5"
    checkpoint = BestModelCheckpoint(checkpoint_path)

    histories = []
    freeze_epochs = min(args.freeze_epochs, args.epochs)
    if freeze_epochs > 0 and args.model != "unet":
        set_backbone_trainable(model, False)
        compile_model(model, args.head_lr)
        print(f"\nPhase 1: frozen backbone, {freeze_epochs} epochs")
        histories.append(
            model.fit(
                train,
                validation_data=val,
                epochs=freeze_epochs,
                callbacks=make_callbacks(
                    args.output,
                    checkpoint,
                    append_csv=False,
                    patience=14,
                ),
                verbose=2,
            )
        )

    remaining_epochs = args.epochs - freeze_epochs
    if remaining_epochs > 0:
        if args.model != "unet":
            set_backbone_trainable(model, True)
        compile_model(model, args.backbone_lr if args.model != "unet" else args.head_lr)
        print(f"\nPhase 2: {remaining_epochs} epochs")
        initial_epoch = freeze_epochs if histories else 0
        histories.append(
            model.fit(
                train,
                validation_data=val,
                epochs=args.epochs,
                initial_epoch=initial_epoch,
                callbacks=make_callbacks(
                    args.output,
                    checkpoint,
                    append_csv=bool(histories),
                    patience=18,
                ),
                verbose=2,
            )
        )

    model.load_weights(checkpoint_path)
    val_loss, val_iou = model.evaluate(val, verbose=0)
    report = evaluate_per_breed(model, loader, val_paths, args.batch_size)
    report.update(
        {
            "model": args.model,
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "augment": not args.no_augment,
            "parameters": int(model.count_params()),
            "checkpoint": str(checkpoint_path),
            "validation_loss": float(val_loss),
            "validation_iou": float(val_iou),
            "elapsed_seconds": time.time() - run_start,
        }
    )
    with (args.output / "report.json").open("w") as report_file:
        json.dump(report, report_file, indent=2)

    print("\nValidation summary")
    print(f"  global IoU:       {report['global_iou']:.6f}")
    print(f"  validation IoU:   {report['validation_iou']:.6f}")
    print(f"  worst breed IoU:  {report['worst_breed_iou']:.6f} ({report['worst_breed']})")
    print(f"  best breed IoU:   {report['best_breed_iou']:.6f} ({report['best_breed']})")
    print(f"  breeds > 0.61:    {report['breeds_above_0_61']}/{report['breed_count']}")
    print(f"  elapsed:          {report['elapsed_seconds']:.1f}s")
    print("\nWorst 10 breeds")
    for breed, item in list(report["breeds"].items())[:10]:
        print(f"  {breed:28s} {item['iou']:.6f}  ({item['images']} images)")


if __name__ == "__main__":
    main()
