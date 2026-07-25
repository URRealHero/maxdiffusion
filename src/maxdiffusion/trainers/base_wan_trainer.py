"""
Copyright 2025 Google LLC

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

import abc
from concurrent.futures import ThreadPoolExecutor
import datetime
import os
import pprint
import socket
import threading
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from maxdiffusion import max_logging, max_utils, train_utils
from maxdiffusion.generate_wan import inference_generate_video
from maxdiffusion.generate_wan import run as generate_wan
from maxdiffusion.pipelines.wan.wan_pipeline import WanPipeline
from maxdiffusion.schedulers import FlaxFlowMatchScheduler
from maxdiffusion.train_utils import (_metrics_queue, _tensorboard_writer_worker, load_next_batch)
from maxdiffusion.utils import load_video
from maxdiffusion.video_processor import VideoProcessor
import numpy as np
from skimage.metrics import structural_similarity as ssim


class TrainState(train_state.TrainState):
  graphdef: nnx.GraphDef
  rest_of_state: nnx.State


def _to_array(x):
  if not isinstance(x, jax.Array):
    x = jnp.asarray(x)
  return x


def _advance_training_rng(rng, steps):
  """Advance the WAN train-step PRNG recurrence by the requested updates.

  Every WAN train step splits its input key into four keys and returns element
  one as the next step's key. Replaying that small recurrence on resume keeps
  timestep/noise/dropout randomness aligned with an uninterrupted run without
  putting a PRNG key into older checkpoint schemas.
  """
  steps = int(steps)
  if steps < 0:
    raise ValueError(f"steps must be non-negative, got {steps}")
  if steps == 0:
    return rng
  return jax.jit(
      lambda key: jax.lax.fori_loop(0, steps, lambda _, k: jax.random.split(k, num=4)[1], key)
  )(rng)


def _checkpoint_restore_args(opt_state, step):
  """Build restore args without treating valid step zero as false."""
  if opt_state is None or step is None:
    return {}
  return {"opt_state": opt_state, "step": step}


def _checkpoint_payload(state, save_optimizer):
  """Mirror periodic/final checkpoint payload selection."""
  return state if save_optimizer else state.params


def generate_sample(config, pipeline, filename_prefix):
  """
  Generates a video to validate training did not corrupt the model
  """
  if not hasattr(pipeline, "vae"):
    wan_vae, vae_cache = WanPipeline.load_vae(
        pipeline.mesh.devices, pipeline.mesh, nnx.Rngs(jax.random.key(config.seed)), config
    )
    pipeline.vae = wan_vae
    pipeline.vae_cache = vae_cache
  return generate_wan(config, pipeline, filename_prefix)


def print_ssim(pretrained_video_path, posttrained_video_path):
  video_processor = VideoProcessor()
  pretrained_video = load_video(pretrained_video_path[0])
  pretrained_video = video_processor.preprocess_video(pretrained_video)
  pretrained_video = np.array(pretrained_video)
  pretrained_video = np.transpose(pretrained_video, (0, 2, 3, 4, 1))
  pretrained_video = np.uint8((pretrained_video + 1) * 255 / 2)

  posttrained_video = load_video(posttrained_video_path[0])
  posttrained_video = video_processor.preprocess_video(posttrained_video)
  posttrained_video = np.array(posttrained_video)
  posttrained_video = np.transpose(posttrained_video, (0, 2, 3, 4, 1))
  posttrained_video = np.uint8((posttrained_video + 1) * 255 / 2)

  ssim_compare = ssim(pretrained_video[0], posttrained_video[0], multichannel=True, channel_axis=-1, data_range=255)

  max_logging.log(f"SSIM score after training is {ssim_compare}")


class BaseWanTrainer(abc.ABC):
  _profiler: max_utils.Profiler | None = None

  def __init__(self, config):
    if config.train_text_encoder:
      raise ValueError("this script currently doesn't support training text_encoders")
    self.config = config
    self.checkpointer = self._get_checkpointer()

  @abc.abstractmethod
  def _get_checkpointer(self):
    """Returns the checkpointer for the trainer."""

  def post_training_steps(self, pipeline, params, train_states, msg=""):
    pass

  def create_scheduler(self):
    """Creates and initializes the Flow Match scheduler for training."""
    # Inference has always consumed config.flow_shift, but training silently
    # constructed the scheduler with its default shift=3.0. Honour an explicit
    # train_flow_shift when supplied; otherwise keep one shared flow_shift
    # contract for training and inference.
    train_shift = float(getattr(self.config, "train_flow_shift", -1.0) or -1.0)
    if train_shift <= 0:
      train_shift = float(getattr(self.config, "flow_shift", 3.0))
    if not np.isfinite(train_shift) or train_shift <= 0:
      raise ValueError(f"Training flow shift must be finite and > 0, got {train_shift}")
    noise_scheduler = FlaxFlowMatchScheduler(shift=train_shift, dtype=jnp.float32)
    noise_scheduler_state = noise_scheduler.create_state()
    noise_scheduler_state = noise_scheduler.set_timesteps(noise_scheduler_state, num_inference_steps=1000, training=True)
    max_logging.log(f"Training FlowMatch scheduler: shift={train_shift}")
    return noise_scheduler, noise_scheduler_state

  @staticmethod
  def calculate_tflops(pipeline):
    maxdiffusion_config = pipeline.config
    # Model configuration
    height = pipeline.config.height
    width = pipeline.config.width
    num_frames = pipeline.config.num_frames

    # Transformer dimensions
    transformer_config = pipeline.transformer.config
    num_layers = transformer_config.num_layers
    heads = pipeline.transformer.config.num_attention_heads
    head_dim = pipeline.transformer.config.attention_head_dim
    ffn_dim = transformer_config.ffn_dim
    seq_len = int(((height / 8) * (width / 8) * ((num_frames - 1) // pipeline.vae_scale_factor_temporal + 1)) / 4)
    text_encoder_dim = 512
    # Attention FLOPS
    # Self
    self_attn_qkv_proj_flops = 3 * (2 * seq_len * (heads * head_dim) ** 2)
    self_attn_qk_v_flops = 2 * (2 * seq_len**2 * (heads * head_dim))
    # Cross
    cross_attn_kv_proj_flops = 3 * (2 * text_encoder_dim * (heads * head_dim) ** 2)
    cross_attn_q_proj_flops = 1 * (2 * seq_len * (heads * head_dim) ** 2)
    cross_attention_qk_v_flops = 2 * (2 * seq_len * text_encoder_dim * (heads * head_dim))

    # Output_projection from attention
    attn_output_proj_flops = 2 * (2 * seq_len * (heads * head_dim) ** 2)

    total_attn_flops = (
        self_attn_qkv_proj_flops
        + self_attn_qk_v_flops
        + cross_attn_kv_proj_flops
        + cross_attn_q_proj_flops
        + cross_attention_qk_v_flops
        + attn_output_proj_flops
    )

    # FFN
    ffn_flops = 2 * (2 * seq_len * (heads * head_dim) * ffn_dim)

    flops_per_block = total_attn_flops + ffn_flops

    total_transformer_flops = flops_per_block * num_layers

    tflops = maxdiffusion_config.per_device_batch_size * total_transformer_flops / 1e12
    train_tflops = 3 * tflops

    max_logging.log(f"Calculated TFLOPs per pass: {train_tflops:.4f}")
    return train_tflops, total_attn_flops, seq_len

  @abc.abstractmethod
  def get_data_shardings(self, mesh):
    """Returns data shardings for training."""

  @abc.abstractmethod
  def get_eval_data_shardings(self, mesh):
    """Returns data shardings for evaluation."""

  @abc.abstractmethod
  def load_dataset(self, mesh, pipeline=None, is_training=True):
    """Loads the dataset."""

  @abc.abstractmethod
  def get_train_step(self, pipeline, mesh, state_shardings, data_shardings):
    """Returns the training step function."""

  @abc.abstractmethod
  def get_eval_step(self, pipeline, mesh, state_shardings, eval_data_shardings):
    """Returns the evaluation step function."""

  def start_training(self):
    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      pipeline, opt_state, step = self.checkpointer.load_checkpoint()
    restore_args = _checkpoint_restore_args(opt_state, step)
    if restore_args:
      del opt_state
    if self.config.enable_ssim:
      # Generate a sample before training to compare against generated sample after training.
      pretrained_video_path = generate_sample(self.config, pipeline, filename_prefix="pre-training-")

    if self.config.eval_every == -1 or (not self.config.enable_generate_video_for_eval):
      # save some memory.
      del pipeline.vae
      del pipeline.vae_cache

    mesh = pipeline.mesh
    train_data_iterator = self.load_dataset(mesh, pipeline=pipeline, is_training=True)

    # Load FlowMatch scheduler
    scheduler, scheduler_state = self.create_scheduler()
    pipeline.scheduler = scheduler
    pipeline.scheduler_state = scheduler_state
    optimizer, learning_rate_scheduler = self.checkpointer._create_optimizer(
        pipeline.transformer, self.config, self.config.learning_rate
    )
    # Returns pipeline with trained transformer state
    pipeline = self.training_loop(pipeline, optimizer, learning_rate_scheduler, train_data_iterator, restore_args)

    if self.config.enable_ssim:
      posttrained_video_path = generate_sample(self.config, pipeline, filename_prefix="post-training-")
      print_ssim(pretrained_video_path, posttrained_video_path)

  def _generate_eval_videos(self, pipeline, mesh, example_batch, step):
    """Generate eval videos mid-training. Default = prompt-only T2V.

    Subclasses override to condition on a sampled dataset record (e.g. the Fun
    camera trainer). `example_batch` is the current (shuffled) training batch.
    """
    inference_generate_video(self.config, pipeline, filename_prefix=f"{step}-train_steps-")

  def eval(self, mesh, eval_rng_key, step, p_eval_step, state, scheduler_state, writer):
    eval_data_iterator = self.load_dataset(mesh, is_training=False)
    eval_rng = eval_rng_key
    eval_losses_by_timestep = {}
    # Loop indefinitely until the iterator is exhausted
    while True:
      try:
        eval_start_time = datetime.datetime.now()
        eval_batch = load_next_batch(eval_data_iterator, None, self.config)
        with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
          metrics, eval_rng = p_eval_step(state, eval_batch, eval_rng, scheduler_state)
          metrics["scalar"]["learning/eval_loss"].block_until_ready()
        losses = metrics["scalar"]["learning/eval_loss"]
        timesteps = eval_batch["timesteps"]
        gathered_losses = multihost_utils.process_allgather(losses, tiled=True)
        gathered_losses = jax.device_get(gathered_losses)
        gathered_timesteps = multihost_utils.process_allgather(timesteps, tiled=True)
        gathered_timesteps = jax.device_get(gathered_timesteps)
        if jax.process_index() == 0:
          for t, l in zip(gathered_timesteps.flatten(), gathered_losses.flatten()):
            timestep = int(t)
            if timestep not in eval_losses_by_timestep:
              eval_losses_by_timestep[timestep] = []
            eval_losses_by_timestep[timestep].append(l)
          eval_end_time = datetime.datetime.now()
          eval_duration = eval_end_time - eval_start_time
          max_logging.log(f"Eval time: {eval_duration.total_seconds():.2f} seconds.")
      except StopIteration:
        # This block is executed when the iterator has no more data
        break
    # Check if any evaluation was actually performed
    if eval_losses_by_timestep and jax.process_index() == 0:
      mean_per_timestep = []
      if jax.process_index() == 0:
        max_logging.log(f"Step {step}, calculating mean loss per timestep...")
      for timestep, losses in sorted(eval_losses_by_timestep.items()):
        losses = jnp.array(losses)
        losses = losses[: min(self.config.eval_max_number_of_samples_in_bucket, len(losses))]
        mean_loss = jnp.mean(losses)
        max_logging.log(f"  Mean eval loss for timestep {timestep}: {mean_loss:.4f}")
        mean_per_timestep.append(mean_loss)
      final_eval_loss = jnp.mean(jnp.array(mean_per_timestep))
      max_logging.log(f"Step {step}, Final Average Eval loss: {final_eval_loss:.4f}")
      if writer:
        writer.add_scalar("learning/eval_loss", final_eval_loss, step)

  def training_loop(self, pipeline, optimizer, learning_rate_scheduler, train_data_iterator, restore_args: dict = {}):
    mesh = pipeline.mesh
    graphdef, params, rest_of_state = nnx.split(pipeline.transformer, nnx.Param, ...)

    with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      state = TrainState.create(
          apply_fn=graphdef.apply, params=params, tx=optimizer, graphdef=graphdef, rest_of_state=rest_of_state
      )
      if restore_args:
        # Checkpoint label semantics: a checkpoint labeled N is saved at the END of
        # loop iteration N (params/opt_state AFTER update N), so a resumed run must
        # execute step N+1 first. Without the +1 the loop re-ran step N — one
        # duplicate update per resume — and immediately overwrote checkpoint N.
        # TrainState.step counts applied updates (N+1 after updates 0..N), so it
        # gets the same +1.
        resume_step = restore_args.get("step", 0) + 1
        restore_args["step"] = resume_step
        max_logging.log(f"Restoring optimizer from checkpoint step {resume_step - 1}; resuming at step {resume_step}")
      # Shard the freshly created state first — it is entirely TPU-resident, so
      # the constraint below is always valid. Restored checkpoint values are
      # spliced in afterwards, directly onto their target shardings.
      state = jax.tree.map(_to_array, state)
      state_spec = nnx.get_partition_spec(state)
      state = jax.lax.with_sharding_constraint(state, state_spec)
      state_shardings = nnx.get_named_sharding(state, mesh)
      if restore_args:
        # flax.struct.replace() is FUNCTIONAL — returns a new object, does NOT mutate
        # in place. Without reassignment the restored opt_state (Adam m/v moments +
        # internal step count) was silently discarded, zeroing optimizer momentum on
        # every resume (weights + step still restored via other paths). Reassign.
        # Two further constraints shape this splice:
        # (1) the orbax restore returns opt_state as a plain nested dict — the
        #     optax NamedTuple node types are not recoverable without an
        #     abstract target, and apply_gradients needs them. The freshly
        #     sharded state.opt_state is the authoritative structure, so the
        #     restored VALUES are transplanted into it leaf-by-leaf. Leaf order
        #     is stable (dicts flatten key-sorted; every optax state in this
        #     chain has alphabetical field order); shape+dtype are validated so
        #     structure drift is a hard failure, never silent corruption.
        # (2) the restored values sit replicated on a host-local CPU mesh, and
        #     in multi-controller JAX no primitive may move arrays between
        #     device sets — each value is therefore rebuilt as a global array
        #     on its template leaf's exact sharding (every host holds the full
        #     value, so it can serve any shard).
        template_leaves, template_def = jax.tree.flatten(state.opt_state)
        restored_leaves = jax.tree.leaves(restore_args.get("opt_state"))
        if len(restored_leaves) != len(template_leaves):
          raise ValueError(
              f"restored opt_state has {len(restored_leaves)} leaves, freshly "
              f"initialized optimizer has {len(template_leaves)} — checkpoint "
              "and optimizer definition do not match"
          )
        for i, (t, r) in enumerate(zip(template_leaves, restored_leaves)):
          t_shape, r_shape = tuple(getattr(t, "shape", ())), tuple(getattr(r, "shape", ()))
          t_dtype, r_dtype = getattr(t, "dtype", None), getattr(r, "dtype", None)
          if t_shape != r_shape or t_dtype != r_dtype:
            raise ValueError(
                f"restored opt_state leaf {i} mismatch: checkpoint {r_shape}/{r_dtype} "
                f"vs optimizer {t_shape}/{t_dtype}"
            )

        def _global_from_host(value, template_leaf):
          host = np.asarray(jax.device_get(value))
          return jax.make_array_from_callback(host.shape, template_leaf.sharding, lambda idx, _h=host: _h[idx])

        restored_opt_state = jax.tree.unflatten(
            template_def, [_global_from_host(r, t) for r, t in zip(restored_leaves, template_leaves)]
        )
        step_value = _global_from_host(np.asarray(resume_step, dtype=state.step.dtype), state.step)
        state = state.replace(opt_state=restored_opt_state, step=step_value)
        del restore_args["opt_state"]
        del optimizer
      if jax.process_index() == 0 and restore_args:
        max_logging.log("--- Optimizer State Sharding Spec (opt_state) ---")
        pretty_string = pprint.pformat(state_spec.opt_state, indent=4, width=60)
        max_logging.log(pretty_string)
        max_logging.log("------------------------------------------------")
    if self.config.hardware != "gpu":
      max_utils.delete_pytree(params)
    data_shardings = self.get_data_shardings(mesh)
    eval_data_shardings = self.get_eval_data_shardings(mesh)

    writer = max_utils.initialize_summary_writer(self.config)
    writer_thread = threading.Thread(target=_tensorboard_writer_worker, args=(writer, self.config), daemon=True)
    writer_thread.start()

    num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
    max_utils.add_text_to_summary_writer("number_model_parameters", str(num_model_parameters), writer)
    max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ.get("LIBTPU_INIT_ARGS", ""), writer)
    max_utils.add_config_to_summary_writer(self.config, writer)

    max_logging.log(
        f"multihost: this host is jax process {jax.process_index()} of {jax.process_count()} "
        f"(hostname {socket.gethostname()}); every host logs '[proc N] completed step' lines"
    )
    if jax.process_index() == 0:
      max_logging.log("***** Running training *****")
      max_logging.log(f"  Instantaneous batch size per device = {self.config.per_device_batch_size}")
      max_logging.log(f"  Total train batch size (w. parallel & distributed) = {self.config.global_batch_size_to_train_on}")
      max_logging.log(f"  Total optimization steps = {self.config.max_train_steps}")

    p_train_step = self.get_train_step(pipeline, mesh, state_shardings, data_shardings)
    p_eval_step = self.get_eval_step(pipeline, mesh, state_shardings, eval_data_shardings)

    rng = jax.random.key(self.config.seed)
    rng, eval_rng_key = jax.random.split(rng)
    start_step = 0
    last_step_completion = datetime.datetime.now()
    local_metrics_file = open(self.config.metrics_file, "a", encoding="utf8") if self.config.metrics_file else None
    running_gcs_metrics = [] if self.config.gcs_metrics else None
    first_profiling_step = self.config.skip_first_n_steps_for_profiler
    if max_utils.profiler_enabled(self.config) and first_profiling_step >= self.config.max_train_steps:
      raise ValueError("Profiling requested but initial profiling step set past training final step")
    last_profiling_step = np.clip(
        first_profiling_step + self.config.profiler_steps - 1, first_profiling_step, self.config.max_train_steps - 1
    )
    start_step = restore_args.get("step", 0)
    if start_step:
      max_logging.log(f"Resuming training from step {start_step}")
      rng = _advance_training_rng(rng, start_step)
    per_device_tflops, _, _ = BaseWanTrainer.calculate_tflops(pipeline)
    scheduler_state = pipeline.scheduler_state
    example_batch = load_next_batch(train_data_iterator, None, self.config)

    with ThreadPoolExecutor(max_workers=1) as executor:
      for step in np.arange(start_step, self.config.max_train_steps):
        if max_utils.profiler_enabled(self.config) and step == first_profiling_step:
          self._profiler = max_utils.Profiler(self.config)
          self._profiler.start()
        start_step_time = datetime.datetime.now()

        next_batch_future = executor.submit(load_next_batch, train_data_iterator, example_batch, self.config)
        with (
            jax.profiler.StepTraceAnnotation("train", step_num=step),
            pipeline.mesh,
            nn_partitioning.axis_rules(self.config.logical_axis_rules),
        ):
          state, scheduler_state, train_metric, rng = p_train_step(state, example_batch, rng, scheduler_state)
          train_metric["scalar"]["learning/loss"].block_until_ready()
          # Hard-stop on non-finite (NaN/Inf) loss so a diverged or misconfigured
          # run fails loudly instead of silently burning compute. Trainers that
          # intentionally skip bad batches (frame_concat_skip_nonfinite_update)
          # opt out; disable globally with stop_on_nonfinite_loss=False.
          _stop_on_nonfinite = str(getattr(self.config, "stop_on_nonfinite_loss", True)).lower() == "true"
          _skip_nonfinite = str(getattr(self.config, "frame_concat_skip_nonfinite_update", False)).lower() == "true"
          if _stop_on_nonfinite and not _skip_nonfinite:
            _loss_value = float(train_metric["scalar"]["learning/loss"])
            if not np.isfinite(_loss_value):
              raise RuntimeError(
                  f"Non-finite training loss ({_loss_value}) at step {step}; stopping training. "
                  "Disable this guard with stop_on_nonfinite_loss=False."
              )
            _trainable_grads_all_finite = train_metric["scalar"].get(
                "debug/trainable_grads_all_finite_before_sanitize", None
            )
            if _trainable_grads_all_finite is not None and float(_trainable_grads_all_finite) == 0.0:
              raise RuntimeError(
                  f"Non-finite trainable gradients at step {step}; stopping training. "
                  "Disable this guard with stop_on_nonfinite_loss=False."
              )
        last_step_completion = datetime.datetime.now()

        if max_utils.profiler_enabled(self.config) and step == last_profiling_step:
          if self._profiler:
            self._profiler.stop()

        train_utils.record_scalar_metrics(
            train_metric, last_step_completion - start_step_time, per_device_tflops, learning_rate_scheduler(step)
        )
        if self.config.write_metrics:
          train_utils.write_metrics(writer, local_metrics_file, running_gcs_metrics, train_metric, step, self.config)

        if self.config.eval_every > 0 and (step + 1) % self.config.eval_every == 0:
          if self.config.enable_generate_video_for_eval:
            pipeline.transformer = nnx.merge(state.graphdef, state.params, state.rest_of_state)
            self._generate_eval_videos(pipeline, mesh, example_batch, step + 1)
          # Re-create the iterator each time you start evaluation to reset it
          # This assumes your data loading logic can be called to get a fresh iterator.
          # p_eval_step is None for trainers that don't compute eval-loss (e.g. the
          # Fun camera trainer, whose dataset has no per-record `timesteps`).
          if p_eval_step is not None:
            self.eval(mesh, eval_rng_key, step, p_eval_step, state, scheduler_state, writer)

        example_batch = next_batch_future.result()
        if step != 0 and self.config.checkpoint_every != -1 and step % self.config.checkpoint_every == 0:
          max_logging.log(f"Saving checkpoint for step {step}")
          self.checkpointer.save_checkpoint(step, pipeline, _checkpoint_payload(state, self.config.save_optimizer))

      _metrics_queue.put(None)
      writer_thread.join()
      if writer:
        writer.flush()
      if self.config.save_final_checkpoint:
        final_step = self.config.max_train_steps - 1
        max_logging.log(f"Saving final checkpoint for step {final_step}")
        # Keep final checkpoints structurally identical to periodic ones. In
        # mixed-precision runs the full state can contain fp32 master weights
        # as well as optimizer moments needed for a faithful resume.
        self.checkpointer.save_checkpoint(
            final_step, pipeline, _checkpoint_payload(state, self.config.save_optimizer)
        )
        self.checkpointer.checkpoint_manager.wait_until_finished()
      # load new state for trained transformer
      pipeline.transformer = nnx.merge(state.graphdef, state.params, state.rest_of_state)
      return pipeline
