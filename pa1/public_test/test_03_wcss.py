"""Hand-computable WCSS equals four (TODO-4)."""
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
TEST_DESCRIPTION = 'P03 | TODO-4 | KMeansClustering.compute_wcss()'
TEST_DETAILS = 'Checks the sum of squared Euclidean distances to assigned centroids.'

def check(s, output=None):
    model = s.KMeansClustering(k=2)
    model.centroids = np.array([[1.0, 0.0], [11.0, 0.0]])
    model.labels = np.array([0, 0, 1, 1])
    actual_result = model.compute_wcss(np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0]]))
    if output is not None:
        print(actual_result, file=output)
    expected_result = 4.0
    assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'WCSS: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
    assert np.issubdtype(np.asarray(actual_result).dtype, np.number), f'WCSS: expected numeric values\nExpected: numeric dtype\nActual: {np.asarray(actual_result).dtype}'
    assert np.all(np.isfinite(actual_result)), f'WCSS: contains NaN or infinity\nExpected: finite values (no NaN or infinity)\nActual: {np.asarray(actual_result)}'
    assert np.allclose(actual_result, expected_result, rtol=1e-06, atol=1e-08), f'WCSS: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
if __name__ == '__main__':
    np.set_printoptions(precision=6, threshold=20, edgeitems=3)
    print(TEST_DESCRIPTION)
    print(TEST_DETAILS)
    print('Input:\nX = [[0, 0], [2, 0], [10, 0], [12, 0]]\nk = 2\ncentroids = [[1, 0], [11, 0]]\nlabels = [0, 0, 1, 1]')
    print('Expected:\n4.0')
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
