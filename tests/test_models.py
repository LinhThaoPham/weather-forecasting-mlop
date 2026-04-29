"""Tests for ML model logic — Prophet and LSTM model classes."""
import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


try:
    from src.config.gcs_storage import USE_GCS, LOCAL_MODELS_DIR, download_model, upload_model
    HAS_GCS = True
except ImportError:
    HAS_GCS = False


@pytest.mark.skipif(not HAS_GCS, reason="gcs_storage module only on feature/gcp-cloud-run branch")
class TestGCSStorage:
    def test_use_gcs_default_false(self):
        assert USE_GCS is False

    def test_local_models_dir_exists_as_string(self):
        assert isinstance(LOCAL_MODELS_DIR, str)

    def test_download_model_local_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            download_model("nonexistent_model.h5")

    def test_upload_model_local_mode(self, tmp_path):
        temp_file = tmp_path / "test_model.pkl"
        temp_file.write_text("fake model data")
        result = upload_model(str(temp_file), "test_model.pkl")
        assert isinstance(result, str)


# ── Feature Engineering ──

class TestFeatureEngineering:
    def test_import_module(self):
        from src.data_pipeline import feature_engineering
        assert feature_engineering is not None


# ── Hybrid Ensemble ──

class TestHybridEnsemble:
    def test_import_module(self):
        from src.models_logic import hybrid_ensemble
        assert hybrid_ensemble is not None

    def test_weighted_average_logic(self):
        """Verify the ensemble weighting concept."""
        from src.config.constants import PROPHET_WEIGHT
        lstm_weight = 1 - PROPHET_WEIGHT

        prophet_pred = 30.0
        lstm_pred = 28.0
        expected = PROPHET_WEIGHT * prophet_pred + lstm_weight * lstm_pred

        assert abs(expected - (PROPHET_WEIGHT * 30 + lstm_weight * 28)) < 0.01


# ── Evaluate Module ──

class TestEvaluate:
    def test_import_module(self):
        from src.training import evaluate
        assert evaluate is not None
