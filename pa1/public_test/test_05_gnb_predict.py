"""Predict two samples from fixed model parameters (TODO-6)."""
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
TEST_DESCRIPTION = 'P05 | TODO-6 | GaussianNaiveBayes.predict()'
TEST_DETAILS = 'Checks GNB prediction using fixed means, variances and log priors.'

def check(s, output=None):
    model = s.GaussianNaiveBayes()
    model.classes = np.array([0, 1])
    model.mean = np.array([[0.0, 0.0], [10.0, 10.0]])
    model.var = np.ones((2, 2))
    model.log_priors = np.array([-0.6931471805599453] * 2)
    actual_result = model.predict(np.array([[1.0, 1.0], [9.0, 9.0]]))
    if output is not None:
        print(actual_result, file=output)
    expected_result = [0, 1]
    assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'predictions: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
    assert np.array_equal(actual_result, expected_result), f'predictions: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
if __name__ == '__main__':
    np.set_printoptions(precision=6, threshold=20, edgeitems=3)
    print(TEST_DESCRIPTION)
    print(TEST_DETAILS)
    print('Input:\nX = [[1, 1], [9, 9]]\nclasses = [0, 1]\nmean = [[0, 0], [10, 10]]\nvar = [[1, 1], [1, 1]]\nlog_priors = [-0.6931471805599453, -0.6931471805599453]')
    print('Expected:\n[0, 1]')
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
