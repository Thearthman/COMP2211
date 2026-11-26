"""Check the submission against the requirements stated on the Lab 8 page."""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import tensorflow as tf

EXPERIMENT_DIR = Path(__file__).resolve().parent
LAB_DIR = EXPERIMENT_DIR.parent
SUBMISSION_PATH = LAB_DIR / "lab8_task.py"
SOURCE_PATH = LAB_DIR / "lab8_tasks.py"
NOTEBOOK_PATH = LAB_DIR / "lab8_tasks.ipynb"
WEIGHTS_PATH = LAB_DIR / "lab8_work" / "model.weights.h5"
REPORT_PATH = EXPERIMENT_DIR / "final_report.json"

sys.path.insert(0, str(EXPERIMENT_DIR))
from sync_notebook import extract_code_cells


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    comparison = node.test
    if len(comparison.ops) != 1 or not isinstance(comparison.ops[0], ast.Eq):
        return False
    operands = [comparison.left, *comparison.comparators]
    return any(
        isinstance(operand, ast.Name) and operand.id == "__name__"
        for operand in operands
    )


def check_import_is_safe() -> dict:
    source = SUBMISSION_PATH.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.If) and _is_main_guard(node):
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        raise AssertionError(
            f"Unguarded top-level statement on line {node.lineno}: "
            f"{type(node).__name__}"
        )

    spec = importlib.util.spec_from_file_location(
        "lab8_submission_under_test", SUBMISSION_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create the submission import spec")
    module = importlib.util.module_from_spec(spec)
    output = io.StringIO()
    with redirect_stdout(output):
        spec.loader.exec_module(module)
    if output.getvalue():
        raise AssertionError(f"Import printed output: {output.getvalue()!r}")
    if "data_path" in vars(module) or "model" in vars(module):
        raise AssertionError("Import created runtime data/model globals")
    return module


def check_data_split(module) -> None:
    pairs = [(Path(f"image_{i}.jpg"), Path(f"mask_{i}.png")) for i in range(10)]
    shuffled = list(pairs)
    np.random.default_rng(42).shuffle(shuffled)
    expected_train = shuffled[: int(len(shuffled) * 0.8)]
    expected_val = shuffled[int(len(shuffled) * 0.8):]

    loader = module.DataLoader(seed=42)
    train_paths, val_paths = loader.load_data(pairs)
    assert train_paths == expected_train
    assert val_paths == expected_val
    assert pairs == [
        (Path(f"image_{i}.jpg"), Path(f"mask_{i}.png")) for i in range(10)
    ]

    repeated_train, repeated_val = module.DataLoader(seed=42).load_data(pairs)
    assert repeated_train == train_paths
    assert repeated_val == val_paths


def check_iou_metric(module) -> None:
    cases = [
        (
            tf.constant([[[[1.0], [0.0]], [[1.0], [0.0]]]]),
            tf.constant([[[[0.9], [0.1]], [[0.8], [0.2]]]]),
            1.0,
        ),
        (
            tf.constant([[[[1.0], [1.0]], [[0.0], [0.0]]]]),
            tf.constant([[[[0.8], [0.4]], [[0.7], [0.1]]]]),
            1.0 / 3.0,
        ),
        (
            tf.constant([[[[1.0], [1.0]], [[0.0], [0.0]]]]),
            tf.constant([[[[0.1], [0.2]], [[0.8], [0.7]]]]),
            0.0,
        ),
    ]
    for y_true, y_pred, expected in cases:
        metric = module.IoUScore()
        metric.update_state(y_true, y_pred)
        np.testing.assert_allclose(metric.result().numpy(), expected, atol=1e-6)

    accumulated = module.IoUScore()

    @tf.function
    def update(y_true, y_pred):
        accumulated.update_state(y_true, y_pred)

    update(
        tf.constant([[[[1.0], [0.0]]]]),
        tf.constant([[[[0.9], [0.8]]]]),
    )
    update(
        tf.constant([[[[1.0], [1.0]], [[1.0], [0.0]]]]),
        tf.constant([[[[0.9], [0.8]], [[0.1], [0.2]]]]),
    )
    assert float(accumulated.intersection.numpy()) == 3.0
    assert float(accumulated.union.numpy()) == 5.0
    np.testing.assert_allclose(accumulated.result().numpy(), 0.6, atol=1e-6)
    accumulated.reset_state()
    assert float(accumulated.result().numpy()) == 0.0

    threshold_metric = module.IoUScore()
    threshold_metric.update_state(
        tf.constant([[[[1.0]]]]),
        tf.constant([[[[0.5]]]]),
    )
    assert float(threshold_metric.intersection.numpy()) == 0.0


def check_model(module) -> dict:
    model = module.BackgroundSeparater()
    batch_one = model(tf.zeros((1, 128, 128, 3)), training=False)
    batch_32 = model(tf.zeros((32, 128, 128, 3)), training=False)
    assert batch_one.shape == (1, 128, 128, 1)
    assert batch_32.shape == (32, 128, 128, 1)
    for output in (batch_one, batch_32):
        values = output.numpy()
        assert np.isfinite(values).all()
        assert 0.0 <= float(values.min()) <= float(values.max()) <= 1.0

    model.load_weights(WEIGHTS_PATH)
    reference = tf.random.uniform((2, 128, 128, 3), seed=123)
    before = model(reference, training=False).numpy()

    reload_path = EXPERIMENT_DIR / "compliance_reload.weights.h5"
    try:
        model.save_weights(reload_path)
        reloaded = module.BackgroundSeparater()
        reloaded(tf.zeros((1, 128, 128, 3)))
        reloaded.load_weights(reload_path)
        after = reloaded(reference, training=False).numpy()
    finally:
        reload_path.unlink(missing_ok=True)

    max_difference = float(np.abs(before - after).max())
    assert max_difference < 1e-6
    parameter_count = int(model.count_params())
    assert parameter_count < 25_000_000
    assert WEIGHTS_PATH.stat().st_size < 100 * 1024 * 1024
    return {
        "parameters": parameter_count,
        "weights_megabytes": WEIGHTS_PATH.stat().st_size / (1024 * 1024),
        "save_reload_max_difference": max_difference,
    }


def check_notebook_sync() -> int:
    chunks = extract_code_cells(SOURCE_PATH.read_text())
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    cells = [
        cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
    ]
    assert len(chunks) == len(cells)
    for cell, chunk in zip(cells, chunks, strict=True):
        source = cell["source"]
        rendered = source if isinstance(source, str) else "".join(source)
        assert rendered == chunk, f"Notebook code cell {cell.get('id')} is out of sync"
        compile(rendered, f"{NOTEBOOK_PATH}:{cell.get('id')}", "exec")
        assert "NotImplementedError" not in rendered
    return len(cells)


def check_training_requirements() -> None:
    source = SUBMISSION_PATH.read_text()
    required_fragments = [
        "losses.BinaryCrossentropy",
        "metrics=[IoUScore()]",
        "monitor='val_iou_score'",
        "save_best_only=True",
        "save_weights_only=True",
        "tf.keras.utils.set_random_seed(42)",
    ]
    for fragment in required_fragments:
        assert fragment in source, f"Missing training requirement: {fragment}"


def check_performance_report() -> dict:
    report = json.loads(REPORT_PATH.read_text())
    assert report["target_met"] is True
    assert report["breed_count"] == 37
    assert report["breeds_above_0_61"] == 37
    assert report["worst_breed_iou"] > 0.61
    return report


def main() -> None:
    assert SUBMISSION_PATH.read_text() == SOURCE_PATH.read_text()
    module = check_import_is_safe()
    check_data_split(module)
    check_iou_metric(module)
    model_report = check_model(module)
    notebook_cells = check_notebook_sync()
    check_training_requirements()
    performance = check_performance_report()

    print("Lab 8 HTML compliance check: passed")
    print(f"Notebook code cells synchronized: {notebook_cells}")
    print(f"Model parameters: {model_report['parameters']:,}")
    print(f"Weights size: {model_report['weights_megabytes']:.2f} MiB")
    print(
        "Save/reload max difference: "
        f"{model_report['save_reload_max_difference']:.3e}"
    )
    print(
        f"Worst breed IoU: {performance['worst_breed_iou']:.6f} "
        f"({performance['worst_breed']})"
    )
    print(
        f"Breeds above 0.61: "
        f"{performance['breeds_above_0_61']}/{performance['breed_count']}"
    )


if __name__ == "__main__":
    main()
