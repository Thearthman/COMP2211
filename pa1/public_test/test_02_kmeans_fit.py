"""Two well-separated clusters (TODO-2-3)."""
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
TEST_DESCRIPTION = 'P02 | TODO-2-3 | KMeansClustering.fit()'
TEST_DETAILS = 'Checks nearest-centroid assignments and updated centroids.'

def check(s, output=None):
    X = np.array([[0.0, 0.0], [0.0, 2.0], [10.0, 0.0], [10.0, 2.0]])
    model = s.KMeansClustering(k=2)
    model.fit(X)
    if output is not None:
        print(f'centroids = {model.centroids}\nlabels = {model.labels}', file=output)
    centers = np.asarray(model.centroids)
    assert centers.shape == (2, 2), f'centroid shape\nExpected: {(2, 2)}\nActual: {centers.shape}'
    actual_result = centers[np.argsort(centers[:, 0])]
    expected_result = [[0.0, 1.0], [10.0, 1.0]]
    assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'centroids: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
    assert np.issubdtype(np.asarray(actual_result).dtype, np.number), f'centroids: expected numeric values\nExpected: numeric dtype\nActual: {np.asarray(actual_result).dtype}'
    assert np.all(np.isfinite(actual_result)), f'centroids: contains NaN or infinity\nExpected: finite values (no NaN or infinity)\nActual: {np.asarray(actual_result)}'
    assert np.allclose(actual_result, expected_result, rtol=1e-06, atol=1e-08), f'centroids: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
    labels = np.asarray(model.labels)
    assert labels.shape == (4,) and np.issubdtype(labels.dtype, np.integer), f'label shape/type\nExpected condition: labels.shape == (4,) and np.issubdtype(labels.dtype, np.integer)\nlabels: {labels}'
    assert np.all((labels >= 0) & (labels < 2)), f'label range\nExpected condition: np.all((labels >= 0) & (labels < 2))\nlabels: {labels}'
    actual_result = centers[labels]
    expected_result = [[0.0, 1.0], [0.0, 1.0], [10.0, 1.0], [10.0, 1.0]]
    assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'assigned centers: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
    assert np.issubdtype(np.asarray(actual_result).dtype, np.number), f'assigned centers: expected numeric values\nExpected: numeric dtype\nActual: {np.asarray(actual_result).dtype}'
    assert np.all(np.isfinite(actual_result)), f'assigned centers: contains NaN or infinity\nExpected: finite values (no NaN or infinity)\nActual: {np.asarray(actual_result)}'
    assert np.allclose(actual_result, expected_result, rtol=1e-06, atol=1e-08), f'assigned centers: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
if __name__ == '__main__':
    np.set_printoptions(precision=6, threshold=20, edgeitems=3)
    print(TEST_DESCRIPTION)
    print(TEST_DETAILS)
    print('Input:\nX = [[0, 0], [0, 2], [10, 0], [10, 2]]\nk = 2, max_iters = 100\nInitialization uses the provided random seed 42.')
    print('Expected:\ncentroids = [[0, 1], [10, 1]] (cluster order may differ)\nSamples 0 and 1 belong to the first centroid above, samples 2 and 3 to the second.')
    print("Actual:")
    output = sys.stdout
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            import pa1_tasks as student
            check(student, output=output)
    except BaseException as error:
        print(f'FAIL: {type(error).__name__}: {error}')
        if not isinstance(error, AssertionError):
            for frame in reversed(traceback.extract_tb(error.__traceback__)):
                if frame.filename.endswith('pa1_tasks.py'):
                    print(f'Student code: pa1_tasks.py, line {frame.lineno}, in {frame.name}')
                    if frame.line:
                        print(frame.line)
                    break
        sys.exit(1)
    print('PASS')
