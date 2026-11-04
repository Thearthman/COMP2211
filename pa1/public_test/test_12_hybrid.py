"""Only the low-margin sample uses KNN (TODO-13)."""
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
TEST_DESCRIPTION = 'P12 | TODO-13 | hybrid_predict()'
TEST_DETAILS = 'Checks hybrid predictions, a Boolean fallback mask and calling KNN only on the fallback subset.'


class MockGNB:
    """Complete test-owned GNB interface, independent of the submission's class."""

    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.log_priors = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = np.array([np.mean(X[y == c], axis=0) for c in self.classes])
        self.var = np.array([np.var(X[y == c], axis=0) for c in self.classes])
        self.log_priors = np.log([np.mean(y == c) for c in self.classes])

    def _log_scores(self, X):
        variance = self.var + 1e-9
        contributions = -0.5 * np.log(2 * np.pi * variance) - (X[:, None, :] - self.mean) ** 2 / (2 * variance)
        return self.log_priors + contributions.sum(axis=2)

    def predict(self, X):
        return self.classes[np.argmax(self._log_scores(X), axis=1)]

    def predict_proba(self, X):
        scores = self._log_scores(X)
        probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
        return probabilities / probabilities.sum(axis=1, keepdims=True)


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
    X = np.array([[1.5], [4.5]])
    gnb = MockGNB()
    gnb.classes = np.arange(4)
    gnb.mean = np.array([[1.0], [0.0], [-10.0], [-20.0]])
    gnb.var = np.ones((4, 1))
    gnb.log_priors = np.log([0.25, 0.25, 0.25, 0.25])
    knn = MockKNN(k=1)
    knn.fit(X.copy(), np.array([2, 1]))
    calls = []

    def predict(X):
        actual_result = X
        expected_result = [[1.5]]
        assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'fallback samples: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
        assert np.array_equal(actual_result, expected_result), f'fallback samples: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
        calls.append(1)
        nearest = np.linalg.norm(X[:, None, :] - knn.X_train[None, :, :], axis=2).argmin(axis=1)
        return knn.y_train[nearest]
    knn.predict = predict
    pred, mask = s.hybrid_predict(gnb, knn, X.copy(), 2.0)
    if output is not None:
        print(f'predictions = {pred}\nfallback_mask = {mask}\nKNN calls = {len(calls)}', file=output)
    actual_result = pred
    expected_result = [2, 0]
    assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'predictions: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
    assert np.array_equal(actual_result, expected_result), f'predictions: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
    actual_result = mask
    expected_result = [True, False]
    assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'mask: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
    assert np.array_equal(actual_result, expected_result), f'mask: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
    assert np.asarray(mask).dtype == np.bool_, f'boolean mask\nExpected: {np.bool_}\nActual: {np.asarray(mask).dtype}'
    assert len(calls) == 1, f'one subset prediction\nExpected: {1}\nActual: {len(calls)}'
if __name__ == '__main__':
    np.set_printoptions(precision=6, threshold=20, edgeitems=3)
    print(TEST_DESCRIPTION)
    print(TEST_DETAILS)
    print('Input:\nX = [[1.5], [4.5]]\nthreshold = 2\ngnb.classes = [0, 1, 2, 3]\ngnb.mean = [[1], [0], [-10], [-20]]\ngnb.var = [[1], [1], [1], [1]]\ngnb.log_priors = log([0.25, 0.25, 0.25, 0.25])\nGNB top-two margins are [1, 4].\nKNN: k=1, X_train=X, y_train=[2, 1]. Must be called once with [[1.5]], returning [2].')
    print('Expected:\npredictions = [2, 0]\nfallback_mask = [True, False]\nKNN calls = 1')
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
