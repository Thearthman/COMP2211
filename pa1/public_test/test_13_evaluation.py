"""Four-class confusion matrix and Macro-F1 (TODO-14)."""
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
TEST_DESCRIPTION = 'P13 | TODO-14 | evaluate_multiclass()'
TEST_DETAILS = 'Checks the confusion matrix and Macro-F1 for 4 classes using fixed true and predicted labels.'

def check(s, output=None):
    matrix, macro = s.evaluate_multiclass(np.array([0, 0, 1, 1, 2, 2, 3, 3]), np.array([0, 1, 1, 1, 2, 0, 3, 2]), num_classes=4)
    if output is not None:
        print(f'confusion_matrix = {matrix}\nmacro_f1 = {macro}', file=output)
    actual_result = matrix
    expected_result = [[1, 1, 0, 0], [0, 2, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]]
    assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'confusion matrix: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
    assert np.array_equal(actual_result, expected_result), f'confusion matrix: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
    actual_result = macro
    expected_result = 0.6166666666666667
    assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'Macro-F1: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
    assert np.issubdtype(np.asarray(actual_result).dtype, np.number), f'Macro-F1: expected numeric values\nExpected: numeric dtype\nActual: {np.asarray(actual_result).dtype}'
    assert np.all(np.isfinite(actual_result)), f'Macro-F1: contains NaN or infinity\nExpected: finite values (no NaN or infinity)\nActual: {np.asarray(actual_result)}'
    assert np.allclose(actual_result, expected_result, rtol=1e-06, atol=1e-08), f'Macro-F1: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
if __name__ == '__main__':
    np.set_printoptions(precision=6, threshold=20, edgeitems=3)
    print(TEST_DESCRIPTION)
    print(TEST_DETAILS)
    print('Input:\ny_true = [0, 0, 1, 1, 2, 2, 3, 3]\ny_pred = [0, 1, 1, 1, 2, 0, 3, 2]\nnum_classes = 4')
    print('Expected:\nconfusion_matrix = [[1, 1, 0, 0], [0, 2, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]]\nmacro_f1 = 0.6166666666666667')
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
