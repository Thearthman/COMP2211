"""Best threshold is four (TODO-12)."""
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
TEST_DESCRIPTION = 'P11 | TODO-12 | select_fallback_threshold()'
TEST_DETAILS = 'Checks selection of the best fallback threshold from [0, 2, 4, 6].'


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
    X = np.array([[1.5], [3.5], [5.5]])
    gnb = MockGNB()
    gnb.classes = np.arange(4)
    gnb.mean = np.array([[1.0], [0.0], [-10.0], [-20.0]])
    gnb.var = np.ones((4, 1))
    gnb.log_priors = np.log([0.25, 0.25, 0.25, 0.25])
    knn = MockKNN(k=1)
    knn.fit(X.copy(), np.ones(3, dtype=int))
    result = s.select_fallback_threshold(gnb, knn, X.copy(), np.array([1, 1, 0]), [0, 2, 4, 6])
    if output is not None:
        print(result, file=output)
    assert result == 4, f'best threshold\nExpected: {4}\nActual: {result}'
if __name__ == '__main__':
    np.set_printoptions(precision=6, threshold=20, edgeitems=3)
    print(TEST_DESCRIPTION)
    print(TEST_DETAILS)
    print('Input:\nX_val = [[1.5], [3.5], [5.5]]\ny_val = [1, 1, 0]\ncandidate_thresholds = [0, 2, 4, 6]\ngnb.classes = [0, 1, 2, 3]\ngnb.mean = [[1], [0], [-10], [-20]]\ngnb.var = [[1], [1], [1], [1]]\ngnb.log_priors = log([0.25, 0.25, 0.25, 0.25])\nGNB top-two margins are [1, 3, 5].\nKNN: k=1, X_train=X_val, y_train=[1, 1, 1]. Every query is predicted as label 1.')
    print('Expected:\n4')
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
