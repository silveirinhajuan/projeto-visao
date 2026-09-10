"""Tests for the research harness."""

import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_metrics():
    from research.metrics.metrics import mse, mae, accuracy, compute_forgetting

    y_true = np.array([[1.0], [2.0], [3.0]])
    y_pred = np.array([[1.1], [2.1], [2.9]])

    assert mse(y_true, y_pred) < 0.02
    assert mae(y_true, y_pred) < 0.15

    # Test forgetting
    perf = np.array([[0.9, 0.8, 0.7], [0.0, 0.85, 0.75], [0.0, 0.0, 0.8]])
    f = compute_forgetting(perf)
    assert f > 0
    print("✓ Metrics tests passed")


def test_statistics():
    from research.statistics.stats import mean, std, confidence_interval, cohens_d

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert mean(values) == 3.0
    assert std(values) > 0

    ci_low, ci_high = confidence_interval(values)
    assert ci_low < mean(values) < ci_high

    d = cohens_d([3, 4, 5], [1, 2, 3])
    assert d > 0
    print("✓ Statistics tests passed")


def test_models():
    from research.models.model_defs import MLPModel, GRUModel, LSTMModel, CfCModel

    X = np.random.randn(100, 2)
    Y = np.random.randn(100, 1)

    for ModelClass in [MLPModel, GRUModel, LSTMModel, CfCModel]:
        model = ModelClass(n_in=2, n_out=1, n_hidden=16, epochs=5, seed=42)
        result = model.fit(X, Y)
        assert "final_loss" in result
        assert model.is_trained

        preds = model.predict(X)
        assert preds.shape == Y.shape
        print(f"  ✓ {ModelClass.__name__}")

    print("✓ Model tests passed")


def test_config_loader():
    from research.config_loader import load_config, save_config
    import tempfile

    cfg = {"name": "test", "seed": 123, "type": "benchmark"}
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        import yaml
        yaml.dump(cfg, f)
        path = f.name

    loaded = load_config(path)
    assert loaded["seed"] == 123
    assert loaded["name"] == "test"
    print("✓ Config loader tests passed")


def test_datasets():
    from research.datasets.generators import sine_regression, mackey_glass, classification_task

    X_tr, Y_tr, X_te, Y_te = sine_regression(n_train=100, n_test=50, seed=42)
    assert len(X_tr) == 100
    assert len(X_te) == 50

    X_tr, Y_tr, X_te, Y_te = mackey_glass(n_train=200, n_test=50, seed=42)
    assert len(X_tr) == 200

    X_tr, Y_tr, X_te, Y_te = classification_task(n_train=100, n_test=50, seed=42)
    assert len(X_tr) == 100
    print("✓ Dataset tests passed")


if __name__ == "__main__":
    test_metrics()
    test_statistics()
    test_models()
    test_config_loader()
    test_datasets()
    print("\n✅ All tests passed!")
