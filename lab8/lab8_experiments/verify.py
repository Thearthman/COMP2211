"""Verify the submitted Lab 8 checkpoint against the validation split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

EXPERIMENT_DIR = Path(__file__).resolve().parent
LAB_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(LAB_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR))

from lab8_tasks import BackgroundSeparater, DataLoader
from experiment import evaluate_per_breed


def main() -> None:
    weights_path = LAB_DIR / "lab8_work" / "model.weights.h5"
    data_root = LAB_DIR / "lab8_work" / "data"
    image_paths = sorted((data_root / "images").glob("*.jpg"))
    trimap_dir = data_root / "annotations" / "trimaps"
    paired_paths = [
        (image_path, trimap_dir / f"{image_path.stem}.png")
        for image_path in image_paths
    ]

    model = BackgroundSeparater()
    model(tf.zeros((1, 128, 128, 3)))
    model.load_weights(weights_path)
    print(f"Model parameters: {model.count_params():,}")

    batch_one = model(tf.zeros((1, 128, 128, 3)), training=False)
    batch_32 = model(tf.zeros((32, 128, 128, 3)), training=False)
    assert batch_one.shape == (1, 128, 128, 1)
    assert batch_32.shape == (32, 128, 128, 1)
    for batch in (batch_one, batch_32):
        assert np.isfinite(batch.numpy()).all()
        assert float(batch.numpy().min()) >= 0.0
        assert float(batch.numpy().max()) <= 1.0

    reload_path = EXPERIMENT_DIR / "reload_check.weights.h5"
    model.save_weights(reload_path)
    reloaded = BackgroundSeparater()
    reloaded(tf.zeros((1, 128, 128, 3)))
    reloaded.load_weights(reload_path)
    reference_batch = tf.random.uniform((2, 128, 128, 3), seed=123)
    before = model(reference_batch, training=False).numpy()
    after = reloaded(reference_batch, training=False).numpy()
    max_difference = float(np.abs(before - after).max())
    assert max_difference < 1e-6
    reload_path.unlink()

    loader = DataLoader(seed=42)
    loader.process_data(paired_paths)
    shuffled_paths = list(paired_paths)
    np.random.default_rng(42).shuffle(shuffled_paths)
    val_paths = shuffled_paths[int(len(shuffled_paths) * 0.8):]
    assert len(val_paths) == len(loader.X_val)

    report = evaluate_per_breed(model, loader, val_paths, batch_size=32)
    report.update(
        {
            "parameters": int(model.count_params()),
            "batch_one_output_shape": list(batch_one.shape),
            "batch_32_output_shape": list(batch_32.shape),
            "save_reload_max_difference": max_difference,
            "target_iou": 0.61,
            "target_met": bool(
                report["breeds_above_0_61"] == report["breed_count"]
                and report["worst_breed_iou"] > 0.61
            ),
        }
    )

    report_path = EXPERIMENT_DIR / "final_report.json"
    with report_path.open("w") as report_file:
        json.dump(report, report_file, indent=2)

    print(f"Global IoU:      {report['global_iou']:.6f}")
    print(f"Mean image IoU:  {report['mean_image_iou']:.6f}")
    print(
        f"Worst breed IoU: {report['worst_breed_iou']:.6f} "
        f"({report['worst_breed']})"
    )
    print(
        f"Breeds > 0.61:   {report['breeds_above_0_61']}/"
        f"{report['breed_count']}"
    )
    print(f"Save/reload max difference: {max_difference:.3e}")
    assert report["target_met"], "At least one breed is not above 0.61"
    print("Final verification: passed")


if __name__ == "__main__":
    main()
