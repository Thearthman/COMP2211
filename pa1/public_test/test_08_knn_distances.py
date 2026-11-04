"""3-4-5 Euclidean distances (TODO-9)."""
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
TEST_DESCRIPTION = 'P08 | TODO-9 | KNNClassifier.compute_distances()'
TEST_DETAILS = 'Checks the Euclidean distance matrix, including a distance of 5 rather than its square, 25.'

def check(s, output=None):
    model = s.KNNClassifier(k=1)
    model.fit(np.array([[0.0, 0.0], [6.0, 8.0]]), np.array([0, 1]))
    actual_result = model.compute_distances(np.array([[3.0, 4.0], [0.0, 0.0]]))
    if output is not None:
        print(actual_result, file=output)
    expected_result = [[5.0, 5.0], [0.0, 10.0]]
    assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'distances: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
    assert np.issubdtype(np.asarray(actual_result).dtype, np.number), f'distances: expected numeric values\nExpected: numeric dtype\nActual: {np.asarray(actual_result).dtype}'
    assert np.all(np.isfinite(actual_result)), f'distances: contains NaN or infinity\nExpected: finite values (no NaN or infinity)\nActual: {np.asarray(actual_result)}'
    assert np.allclose(actual_result, expected_result, rtol=1e-06, atol=1e-08), f'distances: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
if __name__ == '__main__':
    np.set_printoptions(precision=6, threshold=20, edgeitems=3)
    print(TEST_DESCRIPTION)
    print(TEST_DETAILS)
    print('Input:\nX_train = [[0, 0], [6, 8]]\ny_train = [0, 1]\nX_test = [[3, 4], [0, 0]]\nk = 1')
    print('Expected:\n[[5, 5], [0, 10]]')
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
