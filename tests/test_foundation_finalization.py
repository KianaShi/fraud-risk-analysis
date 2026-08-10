"""Checkpoint-free tests for gated Stage D freezing and Stage E confirmation."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from fraud_detection.foundation_finalization import (
    build_freeze_document,
    evaluate_frozen_test_models,
    load_frozen_configs,
    persist_freeze_artifact,
    prepare_stage_e_data,
    validate_freeze_document,
)
from fraud_detection.foundation_models import MODEL_NAMES, prepare_stage_c_data, run_stage_c
from fraud_detection.modeling import split_membership_fingerprints, stratified_split
from tests.test_foundation_models import FakeTorch, RecordingModel
import finalize_foundation_models as finalization_cli


class FoundationFinalizationTests(unittest.TestCase):
    def setUp(self):
        size = 100
        values = pd.Series(range(size))
        self.raw = pd.DataFrame({
            "id": values.map(lambda x: f"record-{x:03d}"),
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
            "area_code": values.map(lambda x: "212" if x % 2 else "415"),
            "device_browser_type": values.map(lambda x: "CHROME" if x % 2 else "FIREFOX"),
            "ip_address_loc_country": values.map(lambda x: "US" if x % 2 else "CA"),
            "type": values.map(lambda x: "M" if x % 2 else "L"),
        })
        metrics = {
            "pr_auc": 0.9, "roc_auc": 0.9, "precision": 0.8, "recall": 0.8,
            "f1": 0.8, "balanced_accuracy": 0.8, "accuracy": 0.8,
        }
        self.validation = pd.DataFrame([
            {"model": name, "status": "passed", "classification_threshold": 0.5, **metrics}
            for name in MODEL_NAMES
        ])
        defaults = {
            "tabpfn_3": {"n_estimators": 8, "device": "cuda"},
            "tabicl_v2": {"n_estimators": 8, "device": "cuda"},
        }
        self.metadata = {
            "stage": "C",
            "packages": {"tabpfn": "8.1.0", "tabicl": "2.1.1"},
            "representation": {"train": {"column_names": [
                "ea_score", "identity_rank", "reputation_level", "volume_score",
                "result_number", "email_days", "is_valid", "is_connected",
                "personal_device", "receiving_mail", "area_code",
                "device_browser_type", "ip_address_loc_country", "type",
            ]}},
            "models": {
                name: {"defaults": defaults[name], "native_preprocessing": "native"}
                for name in MODEL_NAMES
            },
        }
        partitions = stratified_split(
            self.raw["is_fraud"], stable_ids=self.raw["id"]
        )
        self.metadata["split_membership"] = split_membership_fingerprints(
            self.raw["id"], partitions
        )

    def _persist(self, root):
        validation = root / "validation.csv"
        metadata = root / "metadata.json"
        freeze = root / "foundation_model_configs.json"
        self.validation.to_csv(validation, index=False)
        metadata.write_text(json.dumps(self.metadata), encoding="utf-8")
        persist_freeze_artifact(validation, metadata, freeze)
        return freeze

    def test_both_configs_are_frozen_and_reloaded_before_stage_e(self):
        with tempfile.TemporaryDirectory() as directory:
            freeze = self._persist(Path(directory))
            reloaded = json.loads(freeze.read_text(encoding="utf-8"))
            validate_freeze_document(reloaded)
            self.assertEqual(reloaded["schema_version"], 2)
            self.assertEqual(
                reloaded["split_membership"], self.metadata["split_membership"]
            )
            self.assertEqual(
                {config["configuration_status"] for config in reloaded["models"].values()},
                {"frozen"},
            )

    def test_normal_stage_c_to_d_to_e_membership_flow_succeeds(self):
        models = {name: RecordingModel(name) for name in MODEL_NAMES}
        with tempfile.TemporaryDirectory() as directory, patch(
            "fraud_detection.foundation_models.package_versions",
            return_value={"torch": "x", "tabpfn": "8.1.0", "tabicl": "2.1.1"},
        ):
            root = Path(directory)
            validation = root / "validation.csv"
            metadata = root / "metadata.json"
            freeze = root / "freeze.json"
            run_stage_c(
                self.raw,
                validation,
                metadata,
                model_builder=lambda name, unused: models[name],
                torch_module=FakeTorch(),
            )
            persist_freeze_artifact(validation, metadata, freeze)
            prepared = prepare_stage_e_data(self.raw, load_frozen_configs(freeze))
        self.assertEqual(len(prepared[0]), 85)
        self.assertEqual(len(prepared[2]), 15)

    def test_loaded_frozen_configs_are_recursively_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen = load_frozen_configs(self._persist(Path(directory)))
            with self.assertRaises(TypeError):
                frozen["models"]["tabpfn_3"]["n_estimators"] = 1
            with self.assertRaises(TypeError):
                frozen["models"]["tabicl_v2"]["classification_threshold"] = 0.6

    def test_stage_e_preserves_features_threshold_and_one_prediction_per_model(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen = load_frozen_configs(self._persist(Path(directory)))
            train_validation, y_train_validation, test, y_test, groups = prepare_stage_e_data(
                self.raw, frozen
            )
            models = {name: RecordingModel(name) for name in MODEL_NAMES}
            for name, model in models.items():
                model.get_params = lambda deep=False, name=name: {
                    "n_estimators": 8,
                    "random_state": 42,
                    "device": "cuda",
                    "categorical_features_indices": [10, 11, 12, 13],
                    "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt",
                }
            results = evaluate_frozen_test_models(
                train_validation,
                y_train_validation,
                test,
                y_test,
                groups,
                frozen,
                model_builder=lambda name, unused: models[name],
                torch_module=FakeTorch(),
            )
            self.assertEqual(results["classification_threshold"].unique().tolist(), [0.5])
            for model in models.values():
                self.assertEqual(len(model.fit_frames), 1)
                self.assertEqual(len(model.predict_frames), 1)
                self.assertEqual(model.fit_frames[0].shape[1], 14)
                self.assertEqual(model.predict_frames[0].shape[1], 14)

    def test_invalid_freeze_blocks_test_feature_preparation(self):
        document = build_freeze_document(self.validation, self.metadata)
        document["models"]["tabpfn_3"]["configuration_status"] = "candidate"
        with self.assertRaisesRegex(ValueError, "not frozen"):
            prepare_stage_e_data(self.raw, document)

    def test_stage_e_membership_is_invariant_to_row_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen = load_frozen_configs(self._persist(Path(directory)))
            original = prepare_stage_e_data(self.raw, frozen)
            reordered = self.raw.sample(frac=1, random_state=19).reset_index(drop=True)
            repeated = prepare_stage_e_data(reordered, frozen)
        self.assertEqual(set(original[0]["ea_score"]), set(repeated[0]["ea_score"]))
        self.assertEqual(set(original[2]["ea_score"]), set(repeated[2]["ea_score"]))

    def test_changed_removed_or_additional_membership_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen = load_frozen_configs(self._persist(Path(directory)))
            changed = self.raw.copy()
            changed.loc[0, "id"] = "different-record"
            removed = self.raw.iloc[:-1].copy()
            additional = pd.concat([
                self.raw,
                self.raw.iloc[[-1]].assign(id="additional-record"),
            ], ignore_index=True)
            for name, frame in (
                ("changed", changed),
                ("removed", removed),
                ("additional", additional),
            ):
                with self.subTest(case=name), self.assertRaisesRegex(
                    ValueError, "membership fingerprint"
                ):
                    prepare_stage_e_data(frame, frozen)

    def test_wrong_frozen_membership_is_rejected(self):
        document = build_freeze_document(self.validation, self.metadata)
        document["split_membership"]["test_membership_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "membership fingerprint"):
            prepare_stage_e_data(self.raw, document)

    def test_legacy_freeze_remains_loadable_but_cannot_authorize_stage_e(self):
        document = build_freeze_document(self.validation, self.metadata)
        document["schema_version"] = 1
        del document["split_membership"]
        validate_freeze_document(document)
        with self.assertRaisesRegex(ValueError, "[Ll]egacy.*membership"):
            prepare_stage_e_data(self.raw, document)

    def test_cli_persists_freeze_before_loading_real_data(self):
        events = []
        args = SimpleNamespace(
            test_only=False,
            freeze_only=False,
            input=Path("real.csv"),
            results_dir=Path("results"),
            figures_dir=Path("figures"),
        )

        def fake_persist(*unused):
            events.append("freeze_persisted_and_reloaded")
            return {
                "models": {
                    name: {"configuration_status": "frozen"} for name in MODEL_NAMES
                }
            }

        def fake_read_csv(path):
            self.assertEqual(path, args.input)
            self.assertEqual(events, ["freeze_persisted_and_reloaded"])
            events.append("real_data_loaded")
            return self.raw

        def fake_stage_e(raw, *unused, **unused_kwargs):
            self.assertIs(raw, self.raw)
            events.append("stage_e")
            return pd.DataFrame([{"status": "passed"}]), pd.DataFrame()

        with patch.object(finalization_cli, "parse_args", return_value=args), patch.object(
            finalization_cli, "persist_freeze_artifact", side_effect=fake_persist
        ), patch.object(finalization_cli.pd, "read_csv", side_effect=fake_read_csv), patch.object(
            finalization_cli, "run_stage_e", side_effect=fake_stage_e
        ):
            finalization_cli.main()
        self.assertEqual(
            events, ["freeze_persisted_and_reloaded", "real_data_loaded", "stage_e"]
        )


if __name__ == "__main__":
    unittest.main()
