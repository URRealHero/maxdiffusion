"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import tempfile
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
import tensorflow as tf

from maxdiffusion.input_pipeline import _tfds_data_processing


class _Config(SimpleNamespace):

    def get_keys(self):
        return vars(self).keys()


class TfrecordDataProcessingTest(absltest.TestCase):

    def _write_records(self, directory, sample_ids):
        path = os.path.join(directory, "records.tfrec")
        with tf.io.TFRecordWriter(path) as writer:
            for sample_id in sample_ids:
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "sample_id": tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[sample_id.encode("utf-8")]
                                )
                            )
                        }
                    )
                )
                writer.write(example.SerializeToString())

    def _make_dataset(self, directory, excluded=None, global_batch_size=2):
        self._write_records(directory, ["keep", "drop"])
        exclude_path = ""
        if excluded is not None:
            exclude_path = os.path.join(directory, "exclude.txt")
            with open(exclude_path, "w", encoding="utf-8") as exclude_file:
                exclude_file.write("\n".join(excluded))

        config = _Config(
            cache_latents_text_encoder_outputs=False,
            dataset_save_location=directory,
            dataset_type="tfrecord",
            exclude_sample_ids_path=exclude_path,
        )
        feature_description = {"sample_id": tf.io.FixedLenFeature([], tf.string)}
        with mock.patch.object(
            _tfds_data_processing.multihost_dataloading,
            "MultiHostDataLoadIterator",
            side_effect=lambda ds, mesh: ds,
        ):
            return _tfds_data_processing._make_tfrecord_iterator(
                config=config,
                dataloading_host_index=0,
                dataloading_host_count=1,
                mesh=None,
                global_batch_size=global_batch_size,
                feature_description_fn=feature_description,
                prepare_sample_fn=lambda sample: sample,
                dataset_path=directory,
                is_training=True,
            )

    def test_repeat_before_batch_fills_small_post_filter_shard(self):
        with tempfile.TemporaryDirectory() as tempdir:
            dataset = self._make_dataset(tempdir, excluded=["drop"])
            batch = next(iter(dataset))

        self.assertEqual(batch["sample_id"].numpy().tolist(), [b"keep", b"keep"])

    def test_fully_excluded_shard_fails_before_multihost_iteration(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertRaisesRegex(ValueError, "host 0 has no training records"):
                self._make_dataset(tempdir, excluded=["keep", "drop"])

    def test_unfiltered_records_still_form_complete_batches(self):
        with tempfile.TemporaryDirectory() as tempdir:
            dataset = self._make_dataset(tempdir)
            batch = next(iter(dataset))

        self.assertCountEqual(batch["sample_id"].numpy().tolist(), [b"keep", b"drop"])


if __name__ == "__main__":
    absltest.main()
