"""Select K using controlled validation predictions (TODO-11)."""
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
from unittest.mock import patch
TEST_DESCRIPTION = 'P10 | TODO-11 | find_best_k()'
TEST_DETAILS = 'Checks fitting on the supplied training data and predicting the validation data for candidate K values [1, 3], then selecting the best K and returning accuracies. A controlled KNN model isolates this task.'


class MockKNN:
    """Complete test-owned KNN interface, independent of the submission's class."""

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.astype(int).copy()

    def compute_distances(self, X_test):
        return np.linalg.norm(X_test[:, None, :] - self.X_train[None, :, :], axis=2)

    def predict(self, X_test):
        neighbors = self.y_train[np.argsort(self.compute_distances(X_test), axis=1)[:, :self.k]]
        predictions = []
        for row in neighbors:
            classes, counts = np.unique(row, return_counts=True)
            predictions.append(classes[np.argmax(counts)])
        return np.array(predictions, dtype=int)


def check(s, output=None):
    train = np.array([[0.0], [0.2], [0.4], [2.0]])
    labels = np.array([0, 1, 1, 0])
    val = np.array([[0.01], [0.21]])

    class GivenKNN(MockKNN):

        def __init__(self, k=5):
            super().__init__(k=k)
            self.ready = False

        def fit(self, X, y):
            actual_result = X
            expected_result = train
            assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'training features: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
            assert np.array_equal(actual_result, expected_result), f'training features: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
            actual_result = y
            expected_result = labels
            assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'training labels: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
            assert np.array_equal(actual_result, expected_result), f'training labels: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
            super().fit(X, y)
            self.ready = True

        def predict(self, X):
            assert self.ready, f'fit before predict\nExpected condition: self.ready\nActual: {self.ready}'
            actual_result = X
            expected_result = val
            assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'validation features: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
            assert np.array_equal(actual_result, expected_result), f'validation features: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
            return super().predict(X)

    with patch.object(s, 'KNNClassifier', GivenKNN):
        best, accuracies = s.find_best_k(train, labels, val, np.array([1, 1]), [1, 3])
    if output is not None:
        print(f'best_k = {best}\naccuracies = {accuracies}', file=output)
    assert best == 3, f'best K\nExpected: {3}\nActual: {best}'
    actual_result = accuracies
    expected_result = [0.5, 1.0]
    assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'validation accuracies: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
    assert np.issubdtype(np.asarray(actual_result).dtype, np.number), f'validation accuracies: expected numeric values\nExpected: numeric dtype\nActual: {np.asarray(actual_result).dtype}'
    assert np.all(np.isfinite(actual_result)), f'validation accuracies: contains NaN or infinity\nExpected: finite values (no NaN or infinity)\nActual: {np.asarray(actual_result)}'
    assert np.allclose(actual_result, expected_result, rtol=1e-06, atol=1e-08), f'validation accuracies: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
if __name__ == '__main__':
    np.set_printoptions(precision=6, threshold=20, edgeitems=3)
    print(TEST_DESCRIPTION)
    print(TEST_DETAILS)
    print('Input:\nX_train = [[0], [0.2], [0.4], [2]]\ny_train = [0, 1, 1, 0]\nX_val = [[0.01], [0.21]]\ny_val = [1, 1]\nk_choices = [1, 3]\nIndependent KNN validation predictions: k=1 -> [0, 1], k=3 -> [1, 1].\nThe test also checks that fit() receives the training data before predict() receives X_val.')
    print('Expected:\nbest_k = 3\naccuracies = [0.5, 1.0]')
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
