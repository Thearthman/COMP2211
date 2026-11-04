"""Evidence and feature-name ordering for one sample (TODO-8)."""
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
TEST_DESCRIPTION = 'P07 | TODO-8 | explain_prediction()'
TEST_DETAILS = 'Checks the predicted class, feature ordering and Gaussian log-density contributions.'


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


def check(s, output=None):
    model = MockGNB()
    model.classes = np.array([0, 1])
    model.mean = np.array([[0.0, 0.0], [5.0, 5.0]])
    model.var = np.array([[1.0, 4.0], [2.0, 3.0]])
    model.log_priors = np.log([0.5, 0.5])
    pred, ranked = s.explain_prediction(model, np.array([0.0, 0.0]), np.array(['red', 'shape']))
    if output is not None:
        print(f'predicted_class = {pred}\nranked_features = {ranked}', file=output)
    assert pred == 0, f'predicted class\nExpected: {0}\nActual: {pred}'
    assert [r[0] for r in ranked] == ['red', 'shape'], f"feature ordering\nExpected: {['red', 'shape']}\nActual: {[r[0] for r in ranked]}"
    actual_result = [r[1] for r in ranked]
    expected_result = [-0.9189385337046727, -1.612085713889618]
    assert np.asarray(actual_result).shape == np.asarray(expected_result).shape, f'evidence: wrong shape\nExpected: {np.asarray(expected_result).shape}\nActual: {np.asarray(actual_result).shape}'
    assert np.issubdtype(np.asarray(actual_result).dtype, np.number), f'evidence: expected numeric values\nExpected: numeric dtype\nActual: {np.asarray(actual_result).dtype}'
    assert np.all(np.isfinite(actual_result)), f'evidence: contains NaN or infinity\nExpected: finite values (no NaN or infinity)\nActual: {np.asarray(actual_result)}'
    assert np.allclose(actual_result, expected_result, rtol=1e-06, atol=1e-08), f'evidence: wrong values\nExpected: {np.asarray(expected_result)}\nActual: {np.asarray(actual_result)}'
if __name__ == '__main__':
    np.set_printoptions(precision=6, threshold=20, edgeitems=3)
    print(TEST_DESCRIPTION)
    print(TEST_DETAILS)
    print("Input:\nx = [0, 0]\nfeature_names = ['red', 'shape']\ngnb.classes = [0, 1]\ngnb.mean = [[0, 0], [5, 5]]\ngnb.var = [[1, 4], [2, 3]]\ngnb.log_priors = log([0.5, 0.5])")
    print("Expected:\npredicted_class = 0\nranked_features = [('red', -0.9189385337046727), ('shape', -1.612085713889618)]")
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
