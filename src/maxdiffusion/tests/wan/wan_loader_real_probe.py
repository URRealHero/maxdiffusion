"""Manual real-checkpoint parity/timing probe for the WAN transformer loader."""

import argparse
import hashlib
import inspect
import json
import os
import resource
import time

from flax import nnx
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from maxdiffusion.models.wan.transformers.transformer_wan import WanModel
from maxdiffusion.models.wan.wan_utils import load_wan_transformer


_FP32_KEYWORDS = ("norm", "condition_embedder", "scale_shift_table", "lora_", "memory")


def _final_dtype(path):
    path_str = ".".join(map(str, path)).lower()
    if any(keyword in path_str for keyword in _FP32_KEYWORDS):
        return np.dtype(np.float32)
    return np.dtype(ml_dtypes.bfloat16)


def _build_eval_shapes(repo, subfolder):
    config = WanModel.load_config(repo, subfolder=subfolder)
    config["scan_layers"] = True
    config["dtype"] = jnp.bfloat16
    config["weights_dtype"] = jnp.float32
    model = nnx.eval_shape(
        lambda rngs: WanModel(**config, rngs=rngs), rngs=nnx.Rngs(jax.random.key(0))
    )
    _, state, _ = nnx.split(model, nnx.Param, ...)
    return config, state.to_pure_dict()


def _fingerprint(params):
    digest = hashlib.sha256()
    dtype_counts = {}
    total_bytes = 0
    flat = flatten_dict(params)
    for path in sorted(flat, key=lambda key: tuple(map(str, key))):
        array = np.asarray(flat[path])
        target_dtype = _final_dtype(path)
        if array.dtype != target_dtype:
            array = array.astype(target_dtype)
        array = np.ascontiguousarray(array)
        path_str = "/".join(map(str, path))
        header = f"{path_str}|{array.shape}|{array.dtype}".encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(memoryview(array.view(np.uint8)))
        dtype_counts[str(array.dtype)] = dtype_counts.get(str(array.dtype), 0) + 1
        total_bytes += array.nbytes
    return {
        "sha256": digest.hexdigest(),
        "leaf_count": len(flat),
        "total_bytes": total_bytes,
        "dtype_counts": dtype_counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--subfolder", default="transformer")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    shape_start = time.perf_counter()
    config, eval_shapes = _build_eval_shapes(args.repo, args.subfolder)
    shape_seconds = time.perf_counter() - shape_start

    load_kwargs = {
        "pretrained_model_name_or_path": args.repo,
        "eval_shapes": eval_shapes,
        "device": "cpu",
        "num_layers": int(config["num_layers"]),
        "scan_layers": True,
        "subfolder": args.subfolder,
    }
    if "cast_dtype_fn" in inspect.signature(load_wan_transformer).parameters:
        load_kwargs["cast_dtype_fn"] = _final_dtype

    load_start = time.perf_counter()
    params = load_wan_transformer(**load_kwargs)
    load_seconds = time.perf_counter() - load_start

    hash_start = time.perf_counter()
    result = _fingerprint(params)
    result.update(
        {
            "repo": args.repo,
            "subfolder": args.subfolder,
            "revision_source": os.environ.get("PYTHONPATH", ""),
            "shape_seconds": shape_seconds,
            "load_seconds": load_seconds,
            "hash_seconds": time.perf_counter() - hash_start,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
    )
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
