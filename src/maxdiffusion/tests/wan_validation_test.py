# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0

from types import SimpleNamespace
import tempfile
import unittest

import jax.numpy as jnp
import numpy as np

from ..generate_wan_2_2_fun_camera import build_memory_inputs_from_config
from ..models.wan.memory_pose import build_memory_pose_tokens
from ..pipelines.wan.wan_pipeline import (
    _canonical_restored_param_path,
    _canonicalize_restored_flat_params,
    _is_memory_param_path,
    _optional_param_group_is_absent,
)
from ..trainers.wan_trainer import WanTrainer
from ..trainers.wan_vace_trainer import WanVaceTrainer
from ..trainers.wan_2_1_fun_camera_trainer import Wan2_1FunCameraTrainer
from ..trainers.wan_2_2_fun_camera_trainer import Wan2_2FunCameraTrainer


class WanDatasetValidationTest(unittest.TestCase):

    def test_cached_tfrecord_contract_rejects_either_invalid_capability(self):
        trainers = (WanTrainer, WanVaceTrainer, Wan2_1FunCameraTrainer, Wan2_2FunCameraTrainer)
        invalid = (
            SimpleNamespace(dataset_type="other", cache_latents_text_encoder_outputs=True),
            SimpleNamespace(dataset_type="tfrecord", cache_latents_text_encoder_outputs=False),
        )
        for trainer_cls in trainers:
            for config in invalid:
                trainer = trainer_cls.__new__(trainer_cls)
                trainer.config = config
                with self.assertRaises(ValueError, msg=f"{trainer_cls.__name__}: {config}"):
                    trainer.load_dataset(mesh=None, pipeline=None, is_training=True)


class MemoryPoseValidationTest(unittest.TestCase):

    @staticmethod
    def _valid_inputs(k=2):
        extr_key = np.repeat(np.eye(4, dtype=np.float64)[None, :3], k, axis=0)
        intr_key = np.repeat(np.eye(3, dtype=np.float64)[None], k, axis=0)
        extr_query = np.eye(4, dtype=np.float64)[:3]
        intr_query = np.eye(3, dtype=np.float64)
        return extr_key, intr_key, extr_query, intr_query

    def test_valid_pose_shapes(self):
        target, key = build_memory_pose_tokens(*self._valid_inputs(), n_key=2)
        self.assertEqual(target.shape, (1, 1, 9))
        self.assertEqual(key.shape, (1, 2, 9))

    def test_rejects_more_keys_than_poses(self):
        with self.assertRaisesRegex(ValueError, "requires at least"):
            build_memory_pose_tokens(*self._valid_inputs(k=1), n_key=2)

    def test_rejects_nonfinite_pose(self):
        inputs = list(self._valid_inputs())
        inputs[0][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_memory_pose_tokens(*inputs, n_key=2)


class MemoryCheckpointValidationTest(unittest.TestCase):

    def test_restored_path_canonicalizes_nested_list_indices(self):
        path = ("memory_retriever", "retrieval_blocks_list", "0", "attn", "kernel", "value")
        self.assertEqual(
            _canonical_restored_param_path(path),
            ("memory_retriever", "retrieval_blocks_list", 0, "attn", "kernel"),
        )
        self.assertEqual(
            _canonical_restored_param_path(("blocks", "7", "memory_cross_attn", "q", "kernel", "value")),
            ("blocks", 7, "memory_cross_attn", "q", "kernel"),
        )

    def test_memory_param_classification(self):
        self.assertTrue(_is_memory_param_path(("memory_emb_0", "kernel")))
        self.assertTrue(_is_memory_param_path(("memory_retriever", "learnable_query")))
        self.assertTrue(_is_memory_param_path(("blocks", "memory_cross_attn", "q", "kernel")))
        self.assertTrue(_is_memory_param_path(("blocks", 0, "norm_memory", "scale")))
        self.assertFalse(_is_memory_param_path(("blocks", 0, "attn1", "query", "kernel")))

    def test_restored_flat_tree_is_canonical_before_optional_merge(self):
        restored = {
            ("blocks", "0", "attn1", "query", "kernel", "value"): "base",
            ("memory_retriever", "retrieval_blocks_list", "0", "attn", "kernel", "value"): "memory",
        }
        canonical = _canonicalize_restored_flat_params(restored)
        self.assertEqual(
            set(canonical),
            {
                ("blocks", 0, "attn1", "query", "kernel"),
                ("memory_retriever", "retrieval_blocks_list", 0, "attn", "kernel"),
            },
        )
        # Adding a new optional block key now stays in one int-index namespace.
        canonical[("blocks", 0, "memory_cross_attn", "q", "kernel")] = "optional"
        self.assertNotIn("0", {path[1] for path in canonical if path[0] == "blocks"})
        with self.assertRaisesRegex(ValueError, "colliding"):
            _canonicalize_restored_flat_params(
                {
                    ("blocks", "0", "attn1", "query", "kernel", "value"): 1,
                    ("blocks", 0, "attn1", "query", "kernel"): 2,
                }
            )

    def test_optional_group_complete_absent_partial(self):
        expected = {("memory_emb_0", "kernel"), ("memory_emb_2", "kernel")}
        self.assertTrue(_optional_param_group_is_absent(expected, set(), "static-memory"))
        self.assertFalse(_optional_param_group_is_absent(expected, expected, "static-memory"))
        with self.assertRaisesRegex(ValueError, "1/2"):
            _optional_param_group_is_absent(expected, {("memory_emb_0", "kernel")}, "static-memory")
        with self.assertRaisesRegex(ValueError, "zero matching"):
            _optional_param_group_is_absent(set(), set(), "static-memory")


class MemoryInputValidationTest(unittest.TestCase):

    @staticmethod
    def _config(path, n_key):
        return SimpleNamespace(use_memory=True, memory_path=path, memory_n_key=n_key)

    def test_rejects_wrong_memory_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/memory.npy"
            np.save(path, np.zeros((1, 4, 8, 16), dtype=np.float16))
            with self.assertRaisesRegex(ValueError, r"\[K,4,782,1024\]"):
                build_memory_inputs_from_config(self._config(path, 0), jnp.float32)

    def test_rejects_requested_keys_beyond_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/memory.npy"
            np.save(path, np.zeros((1, 4, 782, 1024), dtype=np.float16))
            with self.assertRaisesRegex(ValueError, "exceeds available"):
                build_memory_inputs_from_config(self._config(path, 2), jnp.float32)

    def test_rejects_nonfinite_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/memory.npy"
            memory = np.zeros((1, 4, 782, 1024), dtype=np.float16)
            memory[0, 0, 0, 0] = np.nan
            np.save(path, memory)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                build_memory_inputs_from_config(self._config(path, 0), jnp.float32)


if __name__ == "__main__":
    unittest.main()
