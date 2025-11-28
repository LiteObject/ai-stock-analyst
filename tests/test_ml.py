import numpy as np
import pandas as pd
import pytest

from ml.features.technical import TechnicalFeatures
from ml.models.ensemble import EnsemblePredictor


@pytest.fixture
def sample_ohlcv():
    dates = pd.date_range(start="2022-01-01", periods=300, freq="D")
    df = pd.DataFrame(
        {
            "open": np.random.rand(300) * 100,
            "high": np.random.rand(300) * 100,
            "low": np.random.rand(300) * 100,
            "close": np.random.rand(300) * 100,
            "volume": np.random.randint(1000, 10000, 300),
        },
        index=dates,
    )
    return df


def test_technical_features(sample_ohlcv):
    df = TechnicalFeatures.add_all_features(sample_ohlcv)

    assert "sma_10" in df.columns
    assert "rsi" in df.columns
    assert "macd" in df.columns
    assert "bb_upper" in df.columns
    assert "atr" in df.columns
    assert "obv" in df.columns
    assert "target_direction" in df.columns


def test_ensemble_predictor_rf(sample_ohlcv):
    df = TechnicalFeatures.add_all_features(sample_ohlcv)
    df = df.dropna()

    features = df.drop(columns=["target_direction", "target_return"])
    # Ensure only numeric columns are used
    features = features.select_dtypes(include=[np.number])
    target = df["target_direction"]

    predictor = EnsemblePredictor(model_type="random_forest", task="classification")
    metrics = predictor.train(features, target)

    assert "accuracy" in metrics
    assert predictor.is_trained

    preds = predictor.predict(features)
    assert len(preds) == len(features)

    probs = predictor.predict_proba(features)
    assert len(probs) == len(features)


def test_ensemble_predictor_xgb(sample_ohlcv):
    df = TechnicalFeatures.add_all_features(sample_ohlcv)
    df = df.dropna()

    features = df.drop(columns=["target_direction", "target_return"])
    features = features.select_dtypes(include=[np.number])
    target = df["target_direction"]

    predictor = EnsemblePredictor(model_type="xgboost", task="classification")
    metrics = predictor.train(features, target)

    assert "accuracy" in metrics
    assert predictor.is_trained

    preds = predictor.predict(features)
    assert len(preds) == len(features)
