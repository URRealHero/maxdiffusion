"""
Copyright 2024 Google LLC

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
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from datasets import load_dataset, load_from_disk
import jax
from maxdiffusion import multihost_dataloading, max_logging

AUTOTUNE = tf.data.AUTOTUNE
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_as_tf_dataset(dataset, global_batch_size, shuffle, dataloading_host_count):
  dataset = dataset.with_format("tensorflow")[:]
  tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)

  if shuffle:
    tf_dataset = tf_dataset.shuffle(len(tf_dataset))
  tf_dataset = tf_dataset.batch(global_batch_size // dataloading_host_count, drop_remainder=True)
  tf_dataset = tf_dataset.prefetch(AUTOTUNE)
  tf_dataset = tf_dataset.repeat(-1)

  return tf_dataset


def make_tf_iterator(
    config, dataloading_host_index, dataloading_host_count, mesh, global_batch_size, tokenize_fn, image_transforms_fn
):
  if config.cache_latents_text_encoder_outputs and os.path.isdir(config.dataset_save_location):
    train_ds = load_from_disk(config.dataset_save_location)
  else:
    train_ds = load_dataset(config.dataset_name, split=config.train_split)
    train_ds = train_ds.select_columns([config.caption_column, config.image_column])
    train_ds = train_ds.map(
        function=tokenize_fn,
        batched=True,
        remove_columns=[config.caption_column],
        num_proc=None,
        desc="Running tokenizer on train dataset",
    )
    # need to do it before load_as_tf_dataset
    # since raw images are different sizes
    # will break from_tensor_slices
    train_ds = train_ds.map(
        function=image_transforms_fn,
        batched=True,
        remove_columns=[config.image_column],
        num_proc=None,
        desc="Transforming images",
    )
    if config.cache_latents_text_encoder_outputs:
      train_ds.save_to_disk(config.dataset_save_location)
      # Only process 0 should attempt to clean up cache files
      if jax.process_index() == 0:
        try:
          train_ds.cleanup_cache_files()
        except FileNotFoundError:
          # Ignore FileNotFoundError as files may have been cleaned up by another process
          pass
  train_ds = load_as_tf_dataset(train_ds, global_batch_size, True, dataloading_host_count)
  train_ds = train_ds.shard(num_shards=dataloading_host_count, index=dataloading_host_index)

  train_iter = multihost_dataloading.MultiHostDataLoadIterator(train_ds, mesh)
  return train_iter


# TODO - https://github.com/google/array_record/blob/main/beam/examples/example_gcs_conversion.py
def _make_tfrecord_iterator(
    config,
    dataloading_host_index,
    dataloading_host_count,
    mesh,
    global_batch_size,
    feature_description_fn,
    prepare_sample_fn,
    dataset_path,
    is_training: bool,
):
  # set load_tfrecord_cached to True in config to use pre-processed tfrecord dataset.
  # pedagogical_examples/dataset_tf_cache_to_tfrecord.py to convert tf preprocessed dataset to tfrecord.
  # Dataset cache in github runner test doesn't contain all the features since its shared, Use the default tfrecord iterator.
  # if is_training is True, loads the training dataset. If False, loads the evaluation dataset.

  # checks that the dataset path is valid. In case of gcs, the existence of the dir is not checked.
  is_dataset_dir_valid = "gs://" in config.dataset_save_location or os.path.isdir(config.dataset_save_location)

  # Determine whether to use the "cached" dataset, which requires externally
  # provided parsing functions, or the default one with its internal parsing logic.
  make_cached_tfrecord_iterator = (
      config.cache_latents_text_encoder_outputs
      and is_dataset_dir_valid
      and "load_tfrecord_cached" in config.get_keys()
      and config.load_tfrecord_cached
  )

  feature_description = {
      "moments": tf.io.FixedLenFeature([], tf.string),
      "clip_embeddings": tf.io.FixedLenFeature([], tf.string),
  }

  used_feature_description = (
      feature_description_fn if (make_cached_tfrecord_iterator or config.dataset_type == "tfrecord") else feature_description
  )

  def _parse_tfrecord_fn(example):
    return tf.io.parse_single_example(example, used_feature_description)

  def prepare_sample(features):
    moments = tf.io.parse_tensor(tnp.asarray(features["moments"]), out_type=tf.float32)
    clip_embeddings = tf.io.parse_tensor(tnp.asarray(features["clip_embeddings"]), out_type=tf.float32)
    return {"pixel_values": moments, "input_ids": clip_embeddings}

  # Prefer .tfrec files so sidecar metadata (manifest .jsonl, _done markers) in the
  # same directory is not parsed as TFRecords. Sort for a deterministic order across
  # hosts — sharding below assumes every host sees the same file sequence.
  filenames = sorted(tf.io.gfile.glob(os.path.join(dataset_path, "*.tfrec")))
  if not filenames:
    filenames = sorted(tf.io.gfile.glob(os.path.join(dataset_path, "*.tfrecord")))
  if not filenames:
    filenames = sorted(tf.io.gfile.glob(os.path.join(dataset_path, "*")))

  used_prepare_sample = (
      prepare_sample_fn if (make_cached_tfrecord_iterator or config.dataset_type == "tfrecord") else prepare_sample
  )

  if is_training:
    # File-level sharding: each host reads only its slice of the TFRecord files.
    # Record-level shard had every host open ALL files (num_parallel_reads=AUTOTUNE)
    # and discard (1 - 1/host_count) of the records — amplifying GCS reads by
    # host_count, which throttles/stalls the iterator inside iterator_get_next
    # for large datasets. Sharding the file list means each host only opens its
    # own files.
    files_ds = tf.data.Dataset.from_tensor_slices(filenames)
    if len(filenames) >= dataloading_host_count:
      files_ds = files_ds.shard(num_shards=dataloading_host_count, index=dataloading_host_index)
      files_ds = files_ds.shuffle(len(filenames))  # reshuffle file order each epoch
      ds = files_ds.interleave(
          lambda f: tf.data.TFRecordDataset(f),
          cycle_length=4,
          num_parallel_calls=4,  # bounded; avoids AUTOTUNE opening many GCS connections
          deterministic=False,
      )
    else:
      # Fewer files than hosts: file-level sharding would leave some hosts with
      # zero files -> infinite empty dataset -> multi-host hang at the collective.
      # Small file count means opening all files per host is cheap; shard records.
      max_logging.log(
          f"Found {len(filenames)} TFRecord file(s) for {dataloading_host_count} dataloading host(s); "
          "falling back to record-level sharding to avoid starving hosts."
      )
      files_ds = files_ds.shuffle(len(filenames))
      ds = files_ds.interleave(
          lambda f: tf.data.TFRecordDataset(f),
          cycle_length=4,
          num_parallel_calls=4,
          deterministic=False,
      ).shard(num_shards=dataloading_host_count, index=dataloading_host_index)
    ds = ds.map(_parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
    # Held-out training: drop records whose sample_id is in the exclude list (e.g. the test
    # set) so the model never trains on them. Opt-in via config.exclude_sample_ids_path; the
    # trainer adds 'sample_id' to the feature_description when this is set.
    _exclude_path = config.exclude_sample_ids_path if "exclude_sample_ids_path" in config.get_keys() else ""
    if _exclude_path:
      with tf.io.gfile.GFile(_exclude_path, "r") as _f:
        _bad = [ln.strip() for ln in _f if ln.strip()]
      _excl = tf.lookup.StaticHashTable(
          tf.lookup.KeyValueTensorInitializer(
              tf.constant(_bad, dtype=tf.string), tf.ones([len(_bad)], dtype=tf.int64)
          ),
          default_value=tf.constant(0, dtype=tf.int64),
      )
      max_logging.log(f"Held-out training: excluding {len(_bad)} sample_ids from training ({_exclude_path})")
      ds = ds.filter(lambda x: tf.equal(_excl.lookup(x["sample_id"]), 0))
    ds = (
        ds.map(used_prepare_sample, num_parallel_calls=AUTOTUNE)
        .shuffle(global_batch_size * 10)
        .batch(global_batch_size // dataloading_host_count, drop_remainder=True)
        .repeat(-1)
        .prefetch(AUTOTUNE)
    )
  # For Evaluation: keep record-level sharding (low volume) + padding logic.
  else:
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    num_eval_samples = 0
    for _ in ds:
      num_eval_samples += 1

    remainder = num_eval_samples % global_batch_size
    if remainder != 0:
      num_to_pad = global_batch_size - remainder
      # Create a dataset of padding samples from the beginning
      padding_ds = ds.take(num_to_pad)
      # Add the padding samples to the end
      ds = ds.concatenate(padding_ds)
      max_logging.log(f"Padded evaluation dataset with {num_to_pad} samples.")

    ds = (
        ds.shard(num_shards=dataloading_host_count, index=dataloading_host_index)
        .map(_parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
        .map(used_prepare_sample, num_parallel_calls=AUTOTUNE)
        .batch(global_batch_size // dataloading_host_count, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

  iter = multihost_dataloading.MultiHostDataLoadIterator(ds, mesh)
  return iter


def make_tfrecord_iterator(
    config,
    dataloading_host_index,
    dataloading_host_count,
    mesh,
    global_batch_size,
    feature_description,
    prepare_sample_fn,
    is_training,
):
  """Iterator for TFRecord format. For Laion dataset,
  check out preparation script
  maxdiffusion/pedagogical_examples/to_tfrecords.py
  """
  # Currently only support evaluation on tfrecord. To avoid influencing previous reference, judge whether is training dataset.
  # TODO: refactor to support evaluation on all dataset format.
  dataset_path = config.train_data_dir if is_training else config.eval_data_dir
  return _make_tfrecord_iterator(
      config,
      dataloading_host_index,
      dataloading_host_count,
      mesh,
      global_batch_size,
      feature_description,
      prepare_sample_fn,
      dataset_path,
      is_training,
  )
