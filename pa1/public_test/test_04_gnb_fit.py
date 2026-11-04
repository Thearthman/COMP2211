"""Two-class Gaussian parameter learning (TODO-5)."""
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
TEST_DESCRIPTION = 'P04 | TODO-5 | GaussianNaiveBayes.fit()'
TEST_DETAILS = 'Checks class labels, per-class means, variances and log priors.'

def check(s, output=None):
    model = s.GaussianNaiveBayes()
    model.fit(np.array([[0.0, 2.0], [2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]), np.array([0, 0, 1, 1]))
    if output is not None:
        print(f'classes = {model.classes}\nmean = {model.mean}\nvar = {model.var}\nlog_priors = {model.log_priors}', file=output)
    actual_result = model.classes
    expected_result = [0, 1]
    assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'classes: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
    assert np.array_equal(actual_result, expected_result), f'classes: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
    actual_result = model.mean
    expected_result = [[1.0, 3.0], [8.0, 10.0]]
    assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'means: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
    assert np.issubdtype(np.asarray(actual_result).dtype, np.number), f'means: expected numeric values\nExpected: numeric dtype\nActual: {np.asarray(actual_result).dtype}'
    assert np.all(np.isfinite(actual_result)), f'means: contains NaN or infinity\nExpected: finite values (no NaN or infinity)\nActual: {np.asarray(actual_result)}'
    assert np.allclose(actual_result, expected_result, rtol=1e-06, atol=1e-08), f'means: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
    actual_result = model.var
    expected_result = [[1.0, 1.0], [4.0, 4.0]]
    assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'variances: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
    assert np.issubdtype(np.asarray(actual_result).dtype, np.number), f'variances: expected numeric values\nExpected: numeric dtype\nActual: {np.asarray(actual_result).dtype}'
    assert np.all(np.isfinite(actual_result)), f'variances: contains NaN or infinity\nExpected: finite values (no NaN or infinity)\nActual: {np.asarray(actual_result)}'
    assert np.allclose(actual_result, expected_result, rtol=1e-06, atol=1e-08), f'variances: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
    actual_result = model.log_priors
    expected_result = [-0.6931471805599453, -0.6931471805599453]
    assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'log priors: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
    assert np.issubdtype(np.asarray(actual_result).dtype, np.number), f'log priors: expected numeric values\nExpected: numeric dtype\nActual: {np.asarray(actual_result).dtype}'
    assert np.all(np.isfinite(actual_result)), f'log priors: contains NaN or infinity\nExpected: finite values (no NaN or infinity)\nActual: {np.asarray(actual_result)}'
    assert np.allclose(actual_result, expected_result, rtol=1e-06, atol=1e-08), f'log priors: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
if __name__ == '__main__':
    np.set_printoptions(precision=6, threshold=20, edgeitems=3)
    print(TEST_DESCRIPTION)
    print(TEST_DETAILS)
    print('Input:\nX = [[0, 2], [2, 4], [6, 8], [10, 12]]\ny = [0, 0, 1, 1]')
    print('Expected:\nclasses = [0, 1]\nmean = [[1, 3], [8, 10]]\nvar = [[1, 1], [4, 4]]\nlog_priors = [-0.6931471805599453, -0.6931471805599453]')
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
