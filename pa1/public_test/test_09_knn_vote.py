"""K=3 majority vote on a fixed distance matrix (TODO-10)."""
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
TEST_DESCRIPTION = 'P09 | TODO-10 | KNNClassifier.predict()'
TEST_DETAILS = 'Checks majority voting for K=3 using predefined neighbor distances and labels. Neighbor selection is checked independently of distance calculation.'

def check(s, output=None):
    model = s.KNNClassifier(k=3)
    model.fit(np.array([[0.1], [0.2], [0.3], [0.4]]), np.array([0, 1, 1, 2]))
    model.compute_distances = lambda X: np.linalg.norm(X[:, None, :] - model.X_train[None, :, :], axis=2)
    actual_result = model.predict(np.array([[0.0]]))
    if output is not None:
        print(actual_result, file=output)
    expected_result = [1]
    assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'majority vote: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
    assert np.array_equal(actual_result, expected_result), f'majority vote: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
if __name__ == '__main__':
    np.set_printoptions(precision=6, threshold=20, edgeitems=3)
    print(TEST_DESCRIPTION)
    print(TEST_DETAILS)
    print('Input:\nX_train = [[0.1], [0.2], [0.3], [0.4]]\ny_train = [0, 1, 1, 2]\nX_test = [[0]]\nk = 3\nIndependent Euclidean distance calculation returns [[0.1, 0.2, 0.3, 0.4]].')
    print('Expected:\n[1]')
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
