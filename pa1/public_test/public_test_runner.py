"""Run the unchanged ZINC public scripts against a notebook's current definitions."""
from pathlib import Path
import runpy
from unittest.mock import patch

PART_TESTS = {
    1: ['test_01_standardize.py'],
    2: ['test_02_kmeans_fit.py', 'test_03_wcss.py'],
    3: ['test_04_gnb_fit.py', 'test_05_gnb_predict.py'],
    4: ['test_06_feature_score.py', 'test_07_explain_prediction.py'],
    5: ['test_08_knn_distances.py', 'test_09_knn_vote.py', 'test_10_best_k.py'],
    6: ['test_11_threshold.py', 'test_12_hybrid.py'],
    7: ['test_13_evaluation.py'],
}


def run_public_tests(part, namespace):
    # Forward temporary model patches to the functions' actual global namespace.
    # This preserves the same TODO isolation as the standalone ZINC scripts.
    class NotebookSubmission:
        def __getattr__(self, name):
            if name not in namespace:
                raise AttributeError(name)
            return namespace[name]

        def __setattr__(self, name, value):
            namespace[name] = value

        def __delattr__(self, name):
            del namespace[name]

    scripts = PART_TESTS[part]
    passed = 0
    for filename in scripts:
        print('=' * 70)
        try:
            with patch.dict('sys.modules', {'pa1_tasks': NotebookSubmission()}):
                runpy.run_path(str(Path(__file__).parent / filename), run_name='__main__')
            passed += 1
        except SystemExit as error:
            if error.code not in (None, 0, 1):
                print(f'FAIL: Test exited with code {error.code}.')
        except Exception as error:
            print(f'FAIL: {type(error).__name__}: {error}')
    print(f'Part {part} public tests passed: {passed}/{len(scripts)}')
    return passed
