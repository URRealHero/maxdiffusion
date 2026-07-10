# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0

from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp

from ..trainers.base_wan_trainer import (
    BaseWanTrainer,
    _advance_training_rng,
    _checkpoint_payload,
    _checkpoint_restore_args,
)


class _SchedulerOnlyTrainer(BaseWanTrainer):

    def _get_checkpointer(self):
        return None

    def get_data_shardings(self, mesh):
        raise NotImplementedError

    def get_eval_data_shardings(self, mesh):
        raise NotImplementedError

    def load_dataset(self, mesh, pipeline=None, is_training=True):
        raise NotImplementedError

    def get_train_step(self, pipeline, mesh, state_shardings, data_shardings):
        raise NotImplementedError

    def get_eval_step(self, pipeline, mesh, state_shardings, eval_data_shardings):
        raise NotImplementedError


class BaseWanTrainerSchedulerTest(unittest.TestCase):

    @staticmethod
    def _trainer(flow_shift, train_flow_shift=-1.0):
        return _SchedulerOnlyTrainer(
            SimpleNamespace(
                train_text_encoder=False,
                flow_shift=flow_shift,
                train_flow_shift=train_flow_shift,
            )
        )

    def test_training_scheduler_honors_flow_shift(self):
        scheduler_3, state_3 = self._trainer(3.0).create_scheduler()
        scheduler_5, state_5 = self._trainer(5.0).create_scheduler()
        self.assertEqual(float(scheduler_3.config.shift), 3.0)
        self.assertEqual(float(scheduler_5.config.shift), 5.0)
        self.assertFalse(jnp.allclose(state_3.sigmas, state_5.sigmas))

    def test_explicit_training_shift_override(self):
        scheduler, _ = self._trainer(5.0, train_flow_shift=3.0).create_scheduler()
        self.assertEqual(float(scheduler.config.shift), 3.0)

    def test_resume_rng_matches_uninterrupted_recurrence(self):
        rng, _ = jax.random.split(jax.random.key(123))
        expected = rng
        for _ in range(17):
            expected = jax.random.split(expected, num=4)[1]
        actual = _advance_training_rng(rng, 17)
        self.assertTrue(jnp.array_equal(actual, expected))
        self.assertTrue(jnp.array_equal(_advance_training_rng(rng, 0), rng))
        with self.assertRaises(ValueError):
            _advance_training_rng(rng, -1)

    def test_checkpoint_step_zero_restores(self):
        opt_state = object()
        restore = _checkpoint_restore_args(opt_state, 0)
        self.assertIs(restore["opt_state"], opt_state)
        self.assertEqual(restore["step"], 0)
        self.assertEqual(_checkpoint_restore_args(None, 0), {})
        self.assertEqual(_checkpoint_restore_args(opt_state, None), {})

    def test_final_checkpoint_payload_matches_periodic_payload(self):
        params = object()
        state = SimpleNamespace(params=params)
        self.assertIs(_checkpoint_payload(state, True), state)
        self.assertIs(_checkpoint_payload(state, False), params)


if __name__ == "__main__":
    unittest.main()
