"""Small training, validation and test feature arrays (TODO-1)."""
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
TEST_DESCRIPTION = 'P01 | TODO-1 | standardize_features()'
TEST_DETAILS = 'Checks feature-wise standardization of training, validation and test arrays using training statistics only.'

def check(s, output=None):
    train = np.array([[1.0, 2.0], [3.0, 6.0]])
    val = np.array([[5.0, 10.0]])
    test = np.array([[-1.0, -2.0]])
    result = s.standardize_features(train, val, test)
    if output is not None:
        print(result, file=output)
    assert isinstance(result, (tuple, list)) and len(result) == 3, f'return three arrays\nExpected condition: isinstance(result, (tuple, list)) and len(result) == 3\nresult: {result}'
    for actual, expected in zip(result, [[[-1.0, -1.0], [1.0, 1.0]], [[3.0, 3.0]], [[-3.0, -3.0]]]):
        actual_result = actual
        expected_result = expected
        assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'standardized features: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
        assert np.issubdtype(np.asarray(actual_result).dtype, np.number), f'standardized features: expected numeric values\nExpected: numeric dtype\nActual: {np.asarray(actual_result).dtype}'
        assert np.all(np.isfinite(actual_result)), f'standardized features: contains NaN or infinity\nExpected: finite values (no NaN or infinity)\nActual: {np.asarray(actual_result)}'
        assert np.allclose(actual_result, expected_result, rtol=1e-06, atol=1e-08), f'standardized features: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
if __name__ == '__main__':
    np.set_printoptions(precision=6, threshold=20, edgeitems=3)
    print(TEST_DESCRIPTION)
    print(TEST_DETAILS)
    print('Input:\nX_train = [[1, 2], [3, 6]]\nX_val = [[5, 10]]\nX_test = [[-1, -2]]')
    print('Expected:\nX_train_scaled = [[-1, -1], [1, 1]]\nX_val_scaled = [[3, 3]]\nX_test_scaled = [[-3, -3]]')
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
