import os
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

from tpu_tfrecord_encoder.encode_tv2v import ShardedWriter


class _FakeRecordWriter:
    opened = []

    def __init__(self, path):
        self.path = path
        self.records = []
        self.closed = False
        self.opened.append(self)

    def write(self, value):
        self.records.append(value)

    def close(self):
        self.closed = True


class EncoderHelpersTest(unittest.TestCase):

    def test_sharded_writer_returns_path_across_rollover(self):
        _FakeRecordWriter.opened.clear()
        fake_tf = SimpleNamespace(
            io=SimpleNamespace(
                gfile=SimpleNamespace(makedirs=lambda _: None),
                TFRecordWriter=_FakeRecordWriter,
            )
        )
        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            writer = ShardedWriter("/tmp/encoded", records_per_shard=2, host_index=3, run_id="run-a")
            paths = [writer.write(b"a"), writer.write(b"b"), writer.write(b"c")]
            writer.close()

        self.assertEqual(
            [os.path.basename(p) for p in paths],
            [
                "host_003_run_run-a_file_000000.tfrec",
                "host_003_run_run-a_file_000000.tfrec",
                "host_003_run_run-a_file_000001.tfrec",
            ],
        )
        self.assertTrue(all(paths))
        self.assertEqual([w.records for w in _FakeRecordWriter.opened], [[b"a", b"b"], [b"c"]])

    def test_sidecar_regex_accepts_launcher_run_ids(self):
        fake_tf = SimpleNamespace()
        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            from tpu_tfrecord_encoder.merge_clip_sidecar import META_RE, SHARD_RE

        cases = ("1234", "wan21concatcam-20260710-120000", "manual_retry_2")
        for run_id in cases:
            shard = SHARD_RE.fullmatch(f"host_007_run_{run_id}_file_000042.tfrec")
            meta = META_RE.fullmatch(f"metadata_host_007_run_{run_id}.jsonl")
            self.assertIsNotNone(shard, run_id)
            self.assertIsNotNone(meta, run_id)
            self.assertEqual(shard.groups(), ("007", run_id, "000042"))
            self.assertEqual(meta.groups(), ("007", run_id))


if __name__ == "__main__":
    unittest.main()
