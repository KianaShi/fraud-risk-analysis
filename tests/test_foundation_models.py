"""Checkpoint-free tests for the Stage C Train -> Validation benchmark."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from fraud_detection.foundation_models import (
    MODEL_NAMES,
    TABICL_CHECKPOINT,
    TABPFN_CHECKPOINT,
    evaluate_foundation_models,
    prepare_stage_c_data,
    representation_audit,
    run_stage_c,
)
from fraud_detection.modeling import DEFAULT_FEATURES, BenchmarkPartitions


class FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    @staticmethod
    def empty_cache():
        return None

    @staticmethod
    def reset_peak_memory_stats():
        return None

    @staticmethod
    def synchronize():
        return None

    @staticmethod
    def max_memory_allocated():
        return 10 * 1048576

    @staticmethod
    def max_memory_reserved():
        return 12 * 1048576

    @staticmethod
    def current_device():
        return 0

    @staticmethod
    def get_device_name(index):
        del index
        return "fake CUDA"


class FakeTorch:
    cuda = FakeCuda()


class RecordingModel:
    def __init__(self, name):
        self.name = name
        self.model_path = "auto"
        self.model_path_ = f"cache/{name}.ckpt"
        self.fit_frames = []
        self.predict_frames = []

    def fit(self, X, y):
        self.fit_frames.append(X.copy())
        self.fit_labels = y.copy()
        return self

    def predict_proba(self, X):
        self.predict_frames.append(X.copy())
        probability = np.linspace(0.1, 0.9, len(X))
        return np.column_stack([1 - probability, probability])

    def get_params(self, deep=False):
        del deep
        return {"n_estimators": 8, "device": "cuda"}


class FoundationModelTests(unittest.TestCase):
    def setUp(self):
        size = 100
        values = np.arange(size)
        self.raw = pd.DataFrame(
            {
                "is_fraud": values % 2,
                "ea_score": values.astype(float),
                "identity_rank": values.astype(float),
                "reputation_level": values % 3,
                "volume_score": values % 2,
                "result_number": values % 5,
                "email_days": values * 2,
                "is_valid": values % 2,
                "is_connected": values % 2,
                "personal_device": values % 2,
                "receiving_mail": values % 2,
                "area_code": np.where(values % 2, "212", "415"),
                "device_browser_type": np.where(values % 2, "CHROME", "FIREFOX"),
                "ip_address_loc_country": np.where(values % 2, "US", "CA"),
                "type": np.where(values % 2, "M", "L"),
            }
        )

    def _builders(self):
        models = {name: RecordingModel(name) for name in MODEL_NAMES}
        return models, lambda name, groups: models[name]

    def test_test_features_are_not_selected_or_cleaned(self):
        target = pd.Series(self.raw["is_fraud"].to_numpy(), index=self.raw.index)
        partitions = BenchmarkPartitions(
            train=pd.Index(range(70)),
            validation=pd.Index(range(70, 85)),
            test=pd.Index(range(85, 100)),
        )
        selected_indices = []

        from fraud_detection import foundation_models as module

        real_prepare = module.prepare_model_data

        def recording_prepare(frame):
            selected_indices.extend(frame.index)
            return real_prepare(frame)

        with patch.object(module, "stratified_split", return_value=partitions), patch.object(
            module, "prepare_model_data", side_effect=recording_prepare
        ):
            train, _, validation, _, _ = prepare_stage_c_data(self.raw)
        self.assertEqual(set(selected_indices), set(partitions.train) | set(partitions.validation))
        self.assertFalse(set(selected_indices) & set(partitions.test))
        self.assertEqual(len(train), 70)
        self.assertEqual(len(validation), 15)
        self.assertEqual(target.loc[partitions.test].sum(), 8)

    def test_same_unencoded_14_feature_frames_reach_both_models(self):
        train, y_train, validation, y_validation, groups = prepare_stage_c_data(self.raw)
        models, builder = self._builders()
        results, metadata = evaluate_foundation_models(
            train,
            y_train,
            validation,
            y_validation,
            groups,
            model_builder=builder,
            torch_module=FakeTorch(),
        )
        self.assertEqual(results["model"].tolist(), list(MODEL_NAMES))
        for model in models.values():
            pd.testing.assert_frame_equal(model.fit_frames[0], train)
            pd.testing.assert_frame_equal(model.predict_frames[0], validation)
            self.assertEqual(model.fit_frames[0].shape[1], 14)
            self.assertEqual(model.fit_frames[0]["type"].dtype, object)
        self.assertTrue(metadata["representation"]["information_set_identical_for_both_models"])
        self.assertFalse(metadata["representation"]["external_encoding_or_imputation"])

    def test_predict_proba_metrics_and_checkpoint_metadata_are_recorded(self):
        train, y_train, validation, y_validation, groups = prepare_stage_c_data(self.raw)
        _, builder = self._builders()
        with patch(
            "fraud_detection.foundation_models.package_versions",
            return_value={"torch": "x", "tabpfn": "8.1.0", "tabicl": "2.1.1"},
        ):
            results, metadata = evaluate_foundation_models(
                train,
                y_train,
                validation,
                y_validation,
                groups,
                model_builder=builder,
                torch_module=FakeTorch(),
            )
        required = {"pr_auc", "roc_auc", "precision", "recall", "f1", "balanced_accuracy", "accuracy"}
        self.assertTrue(required <= set(results.columns))
        self.assertEqual(results["classification_threshold"].unique().tolist(), [0.5])
        self.assertEqual(metadata["models"]["tabpfn_3"]["checkpoint"], TABPFN_CHECKPOINT)
        self.assertEqual(metadata["models"]["tabicl_v2"]["checkpoint"], TABICL_CHECKPOINT)

    def test_representation_audit_lists_frozen_feature_groups(self):
        train, _, validation, _, groups = prepare_stage_c_data(self.raw)
        audit = representation_audit(train, validation, groups)
        self.assertEqual(audit["train"]["column_names"], DEFAULT_FEATURES.all)
        self.assertEqual(audit["train"]["features"], 14)
        self.assertEqual(audit["categorical_columns"], list(DEFAULT_FEATURES.categorical))
        self.assertEqual(audit["numeric_columns"], list(DEFAULT_FEATURES.numeric))
        self.assertEqual(audit["boolean_columns"], list(DEFAULT_FEATURES.boolean))

    def test_stage_c_writes_validation_artifacts_only(self):
        _, builder = self._builders()
        with tempfile.TemporaryDirectory() as directory, patch(
            "fraud_detection.foundation_models.package_versions",
            return_value={"torch": "x", "tabpfn": "8.1.0", "tabicl": "2.1.1"},
        ):
            root = Path(directory)
            output = root / "foundation_model_validation.csv"
            metadata = root / "foundation_model_validation_metadata.json"
            run_stage_c(
                self.raw,
                output,
                metadata,
                model_builder=builder,
                torch_module=FakeTorch(),
            )
            self.assertTrue(output.is_file())
            self.assertTrue(metadata.is_file())
            self.assertFalse(any("test" in path.name.lower() for path in root.iterdir()))


if __name__ == "__main__":
    unittest.main()
