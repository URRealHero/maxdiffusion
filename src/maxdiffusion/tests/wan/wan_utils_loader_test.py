"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0
"""

import json
import os
import tempfile
from unittest import mock

from absl.testing import absltest
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from safetensors.torch import save_file
import torch

from maxdiffusion.models.wan import wan_utils


_QUERY_PATH = ("blocks", "attn1", "query", "kernel")


class WanUtilsLoaderTest(absltest.TestCase):

    def _scan_shapes(self, include_optional=False):
        flat = {_QUERY_PATH: jax.ShapeDtypeStruct((2, 3, 2), jnp.float32)}
        if include_optional:
            flat[("blocks", "attn1", "lora_q", "lora_A", "kernel")] = (
                jax.ShapeDtypeStruct((2, 3, 1), jnp.float32)
            )
            flat[("memory_retriever", "proj", "kernel")] = jax.ShapeDtypeStruct(
                (3, 3), jnp.float32
            )
        return unflatten_dict(flat)

    def _non_scan_shapes(self):
        return unflatten_dict(
            {
                ("blocks", 0, "attn1", "query", "kernel"): jax.ShapeDtypeStruct(
                    (3, 2), jnp.float32
                ),
                ("blocks", 1, "attn1", "query", "kernel"): jax.ShapeDtypeStruct(
                    (3, 2), jnp.float32
                ),
            }
        )

    def _query_tensors(self, dtype=torch.float32):
        return {
            "blocks.0.attn1.to_q.weight": torch.arange(6, dtype=dtype).reshape(2, 3),
            "blocks.1.attn1.to_q.weight": (torch.arange(6, dtype=dtype) + 10).reshape(
                2, 3
            ),
        }

    def _write_single(self, directory, tensors, subfolder=""):
        target_dir = os.path.join(directory, subfolder)
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, "diffusion_pytorch_model.safetensors")
        save_file(tensors, path)
        return path

    def _load(
        self, directory, eval_shapes, scan_layers=True, subfolder="", cast_dtype_fn=None
    ):
        return wan_utils.load_base_wan_transformer(
            directory,
            eval_shapes,
            "cpu",
            hf_download=False,
            num_layers=2,
            scan_layers=scan_layers,
            subfolder=subfolder,
            cast_dtype_fn=cast_dtype_fn,
        )

    def test_scanned_single_file_matches_expected_and_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_single(directory, self._query_tensors())
            fast = self._load(directory, self._scan_shapes())
            with mock.patch.dict(os.environ, {"MAXDIFFUSION_WAN_LEGACY_LOADER": "1"}):
                legacy = self._load(directory, self._scan_shapes())

        fast_query = flatten_dict(fast)[_QUERY_PATH]
        legacy_query = np.asarray(flatten_dict(legacy)[_QUERY_PATH])
        expected = np.stack(
            [
                np.arange(6, dtype=np.float32).reshape(2, 3).T,
                (np.arange(6, dtype=np.float32) + 10).reshape(2, 3).T,
            ]
        )
        np.testing.assert_array_equal(fast_query, expected)
        np.testing.assert_array_equal(fast_query, legacy_query)

    def test_non_scan_preserves_integer_block_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_single(directory, self._query_tensors())
            loaded = flatten_dict(
                self._load(directory, self._non_scan_shapes(), scan_layers=False)
            )

        self.assertIn(("blocks", 0, "attn1", "query", "kernel"), loaded)
        self.assertIn(("blocks", 1, "attn1", "query", "kernel"), loaded)
        np.testing.assert_array_equal(
            loaded[("blocks", 1, "attn1", "query", "kernel")],
            (np.arange(6, dtype=np.float32) + 10).reshape(2, 3).T,
        )

    def test_bfloat16_conversion_is_bit_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            tensors = self._query_tensors(torch.bfloat16)
            self._write_single(directory, tensors)
            loaded = flatten_dict(self._load(directory, self._scan_shapes()))[
                _QUERY_PATH
            ]

        self.assertEqual(loaded.dtype, np.dtype(ml_dtypes.bfloat16))
        expected = np.stack(
            [
                tensors["blocks.0.attn1.to_q.weight"].T.view(torch.uint16).numpy(),
                tensors["blocks.1.attn1.to_q.weight"].T.view(torch.uint16).numpy(),
            ]
        )
        np.testing.assert_array_equal(loaded.view(np.uint16), expected)

    def test_dtype_policy_preserves_sensitive_groups(self):
        for path in (
            ("blocks", "norm1", "scale"),
            ("condition_embedder", "time_embedder", "kernel"),
            ("blocks", "adaln_scale_shift_table"),
            ("blocks", "lora_q", "lora_A", "kernel"),
            ("memory_retriever", "proj", "kernel"),
        ):
            self.assertEqual(
                wan_utils.wan_param_dtype(path, jnp.bfloat16), np.dtype(jnp.float32)
            )
        self.assertEqual(
            wan_utils.wan_param_dtype(
                ("blocks", "attn1", "query", "kernel"), jnp.bfloat16
            ),
            np.dtype(jnp.bfloat16),
        )

    def test_optional_lora_and_memory_shapes_may_be_wholly_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_single(directory, self._query_tensors())
            loaded = flatten_dict(
                self._load(directory, self._scan_shapes(include_optional=True))
            )

        self.assertEqual(set(loaded), {_QUERY_PATH})

    def test_missing_scanned_layer_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            tensors = {"blocks.0.attn1.to_q.weight": torch.ones((2, 3))}
            self._write_single(directory, tensors)
            with self.assertRaisesRegex(
                ValueError, "Incomplete scanned WAN parameters"
            ):
                self._load(directory, self._scan_shapes())

    def test_duplicate_destination_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            tensors = self._query_tensors()
            tensors["blocks.0.attn1.query.weight"] = torch.full((2, 3), 7.0)
            self._write_single(directory, tensors)
            with self.assertRaisesRegex(
                ValueError, "Duplicate WAN checkpoint destination"
            ):
                self._load(directory, self._scan_shapes())

    def test_out_of_range_block_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            tensors = self._query_tensors()
            tensors["blocks.2.attn1.to_q.weight"] = torch.full((2, 3), 7.0)
            self._write_single(directory, tensors)
            with self.assertRaisesRegex(ValueError, "outside"):
                self._load(directory, self._scan_shapes())

    def test_local_indexed_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            subfolder = "transformer"
            target = os.path.join(directory, subfolder)
            os.makedirs(target)
            tensors = self._query_tensors()
            save_file(
                {"blocks.0.attn1.to_q.weight": tensors["blocks.0.attn1.to_q.weight"]},
                os.path.join(target, "a.safetensors"),
            )
            save_file(
                {"blocks.1.attn1.to_q.weight": tensors["blocks.1.attn1.to_q.weight"]},
                os.path.join(target, "b.safetensors"),
            )
            with open(
                os.path.join(target, "diffusion_pytorch_model.safetensors.index.json"),
                "w",
                encoding="utf-8",
            ) as index_file:
                json.dump(
                    {
                        "weight_map": {
                            "blocks.0.attn1.to_q.weight": "a.safetensors",
                            "blocks.1.attn1.to_q.weight": "b.safetensors",
                        }
                    },
                    index_file,
                )

            loaded = flatten_dict(
                self._load(directory, self._scan_shapes(), subfolder=subfolder)
            )

        self.assertEqual(set(loaded), {_QUERY_PATH})

    def test_requested_subfolder_falls_back_to_root_single_file(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_single(directory, self._query_tensors())
            loaded = flatten_dict(
                self._load(directory, self._scan_shapes(), subfolder="transformer")
            )

        self.assertEqual(set(loaded), {_QUERY_PATH})


if __name__ == "__main__":
    absltest.main()
