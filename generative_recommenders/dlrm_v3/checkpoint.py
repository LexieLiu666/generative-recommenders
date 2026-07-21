# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
"""
Checkpoint utilities for saving and loading DLRMv3 model checkpoints.

This module provides functions for saving and loading distributed model checkpoints,
including both sparse (embedding) and dense (non-embedding) components.
"""

import gc
import glob
import hashlib
import json
import os
import uuid
from typing import Any, Dict, List, Optional, Set

import gin
import torch
from generative_recommenders.dlrm_v3.utils import MetricsLogger
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.optimizer import Optimizer
from torchrec.distributed.types import ShardedTensor


_RECSTORE_SCHEMA_VERSION = 1
_RECSTORE_MANIFEST = "recstore_manifest.json"
_RECSTORE_DENSE_KEY = "recstore_checkpoint"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalized_json(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _atomic_write_json(path: str, value: Dict[str, Any]) -> None:
    temporary_path = f"{path}.tmp-{uuid.uuid4().hex}"
    try:
        with open(temporary_path, "w", encoding="utf-8") as output:
            output.write(_canonical_json(value) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _atomic_torch_save(path: str, value: Dict[str, Any]) -> None:
    temporary_path = f"{path}.tmp-{uuid.uuid4().hex}"
    try:
        with open(temporary_path, "wb") as output:
            torch.save(value, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _require_single_rank_recstore_checkpoint() -> None:
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() != 1
    ):
        raise RuntimeError(
            "RecStore checkpointing currently supports WORLD_SIZE=1 only"
        )


def _find_recstore_module(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    modules = [
        module
        for module in model.modules()
        if bool(getattr(module, "is_recstore_sparse_module", False))
    ]
    if len(modules) > 1:
        raise RuntimeError(
            "RecStore checkpointing currently supports exactly one sparse module"
        )
    return modules[0] if modules else None


def _module_checkpoint_config(module: torch.nn.Module) -> Dict[str, Any]:
    config_fn = getattr(module, "recstore_checkpoint_config", None)
    if not callable(config_fn):
        raise RuntimeError(
            "RecStore sparse module does not provide recstore_checkpoint_config()"
        )
    config = _normalized_json(config_fn())
    required = {"tables", "fusion", "clamp_ids", "need_indices"}
    if not isinstance(config, dict) or set(config) != required:
        raise RuntimeError(
            "RecStore checkpoint config must contain tables, fusion, clamp_ids, "
            "and need_indices"
        )
    if not isinstance(config["tables"], list) or not config["tables"]:
        raise RuntimeError("RecStore checkpoint config must contain non-empty tables")
    if not isinstance(config["fusion"], dict):
        raise RuntimeError("RecStore checkpoint config must contain fusion metadata")
    if not isinstance(config["clamp_ids"], bool) or not isinstance(
        config["need_indices"], bool
    ):
        raise RuntimeError("RecStore clamp_ids and need_indices must be booleans")
    return config


def _runtime_checkpoint_config() -> Dict[str, Any]:
    config_path = os.environ.get("RECSTORE_CONFIG")
    if not config_path:
        raise RuntimeError("RECSTORE_CONFIG is required for RecStore checkpointing")
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            root = json.load(config_file)
        cache_ps = root["cache_ps"]
        optimizer = _normalized_json(cache_ps["optimizer"])
        distributed_client = root["distributed_client"]
        num_shards = distributed_client["num_shards"]
        hash_method = distributed_client["hash_method"]
        shard_ids = sorted(
            int(server["shard"]) for server in distributed_client["servers"]
        )
        cache_shard_ids = sorted(int(server["shard"]) for server in cache_ps["servers"])
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Invalid RecStore checkpoint config {config_path}"
        ) from error
    if not isinstance(optimizer, dict) or not optimizer:
        raise RuntimeError("RecStore optimizer checkpoint config must be non-empty")
    if (
        isinstance(num_shards, bool)
        or not isinstance(num_shards, int)
        or num_shards <= 0
        or shard_ids != list(range(num_shards))
        or cache_ps.get("num_shards") != num_shards
        or cache_shard_ids != shard_ids
    ):
        raise RuntimeError("RecStore cache and client sharding configs do not match")
    if not isinstance(hash_method, str) or not hash_method:
        raise RuntimeError("RecStore hash_method must be a non-empty string")
    return {
        "optimizer": optimizer,
        "sharding": {
            "num_shards": num_shards,
            "hash_method": hash_method,
            "shard_ids": shard_ids,
        },
    }


def _train_global_step(metric_logger: MetricsLogger) -> int:
    global_step = metric_logger.global_step
    train_step = global_step.get("train") if isinstance(global_step, dict) else None
    if (
        isinstance(train_step, bool)
        or not isinstance(train_step, int)
        or train_step < 0
    ):
        raise RuntimeError("MetricsLogger train global step must be non-negative")
    return train_step


def _build_recstore_metadata(
    module: torch.nn.Module,
    checkpoint_root: str,
    directory_step: int,
    global_step: int,
) -> Dict[str, Any]:
    module_config = _module_checkpoint_config(module)
    runtime_config = _runtime_checkpoint_config()
    run_id = os.path.basename(os.path.abspath(os.path.normpath(checkpoint_root)))
    if not run_id:
        raise RuntimeError("RecStore checkpoint root needs a run directory name")
    identity = {
        "schema_version": _RECSTORE_SCHEMA_VERSION,
        "backend": "recstore",
        "run_id": run_id,
        "generation": uuid.uuid4().hex,
        "directory_step": directory_step,
        "global_step": global_step,
        **module_config,
        **runtime_config,
    }
    checkpoint_id = hashlib.sha256(
        _canonical_json(identity).encode("ascii")
    ).hexdigest()
    return {"identity": identity, "checkpoint_id": checkpoint_id}


def _validate_identity(identity: Any, checkpoint_path: str) -> Dict[str, Any]:
    required = {
        "schema_version",
        "backend",
        "run_id",
        "generation",
        "directory_step",
        "global_step",
        "tables",
        "fusion",
        "clamp_ids",
        "need_indices",
        "optimizer",
        "sharding",
    }
    if not isinstance(identity, dict) or set(identity) != required:
        raise RuntimeError("RecStore checkpoint identity has an invalid schema")
    if identity["schema_version"] != _RECSTORE_SCHEMA_VERSION:
        raise RuntimeError("Unsupported RecStore checkpoint schema version")
    if identity["backend"] != "recstore":
        raise RuntimeError("Checkpoint backend is not RecStore")
    expected_run_id = os.path.basename(
        os.path.dirname(os.path.normpath(checkpoint_path))
    )
    if identity["run_id"] != expected_run_id:
        raise RuntimeError(
            f"RecStore run_id mismatch: expected {expected_run_id}, "
            f"got {identity['run_id']}"
        )
    generation = identity["generation"]
    try:
        parsed_generation = uuid.UUID(hex=generation)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("RecStore checkpoint generation is invalid") from error
    if parsed_generation.hex != generation:
        raise RuntimeError("RecStore checkpoint generation is not canonical")
    for field in ("directory_step", "global_step"):
        value = identity[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"RecStore checkpoint {field} must be non-negative")
    actual_directory = os.path.basename(os.path.normpath(checkpoint_path))
    if actual_directory != str(identity["directory_step"]):
        raise RuntimeError(
            f"RecStore directory step mismatch: expected {identity['directory_step']}, "
            f"got {actual_directory}"
        )
    return identity


def _metadata_from_manifest(
    manifest: Dict[str, Any], checkpoint_path: str
) -> Dict[str, Any]:
    if set(manifest) != {"identity", "checkpoint_id", "shards"}:
        raise RuntimeError("RecStore checkpoint manifest has an invalid schema")
    identity = _validate_identity(manifest["identity"], checkpoint_path)
    expected_id = hashlib.sha256(_canonical_json(identity).encode("ascii")).hexdigest()
    if manifest["checkpoint_id"] != expected_id:
        raise RuntimeError("RecStore checkpoint_id does not match its identity")
    return {"identity": identity, "checkpoint_id": expected_id}


def _validate_shards(
    sparse_path: str,
    identity: Dict[str, Any],
    records: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    expected = {
        f"recstore-shard-{shard_id}.bin"
        for shard_id in identity["sharding"]["shard_ids"]
    }
    if records is None:
        names = {
            os.path.basename(path)
            for path in glob.glob(os.path.join(sparse_path, "recstore-shard-*.bin"))
        }
        if names != expected:
            raise RuntimeError(
                f"RecStore shard files mismatch: expected {sorted(expected)}, "
                f"got {sorted(names)}"
            )
        records = [
            {
                "filename": name,
                "size": os.path.getsize(os.path.join(sparse_path, name)),
            }
            for name in sorted(names)
        ]
    if not isinstance(records, list):
        raise RuntimeError("RecStore checkpoint shards must be a list")
    normalized_records = _normalized_json(records)
    names = set()
    for record in normalized_records:
        if not isinstance(record, dict) or set(record) != {"filename", "size"}:
            raise RuntimeError("RecStore shard record has an invalid schema")
        filename = record["filename"]
        size = record["size"]
        if (
            not isinstance(filename, str)
            or filename != os.path.basename(filename)
            or filename in names
            or filename not in expected
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise RuntimeError("RecStore shard record is invalid")
        shard_path = os.path.join(sparse_path, filename)
        if not os.path.isfile(shard_path) or os.path.getsize(shard_path) != size:
            raise RuntimeError(
                f"RecStore shard {filename} is missing or has the wrong size"
            )
        names.add(filename)
    if names != expected:
        raise RuntimeError("RecStore checkpoint is missing shard files")
    return normalized_records


def _load_manifest(checkpoint_path: str) -> Dict[str, Any]:
    sparse_path = os.path.join(checkpoint_path, "sparse")
    manifest_path = os.path.join(sparse_path, _RECSTORE_MANIFEST)
    try:
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"RecStore checkpoint is incomplete: cannot read {manifest_path}"
        ) from error
    if not isinstance(manifest, dict):
        raise RuntimeError("RecStore checkpoint manifest must be a dictionary")
    metadata = _metadata_from_manifest(manifest, checkpoint_path)
    _validate_shards(sparse_path, metadata["identity"], manifest["shards"])
    return manifest


def _validate_dense_checkpoint_metadata(
    checkpoint_path: str, expected_metadata: Dict[str, Any]
) -> None:
    dense_path = os.path.join(checkpoint_path, "non_sparse.ckpt")
    dense_state = torch.load(dense_path, map_location="cpu")
    if dense_state.get(_RECSTORE_DENSE_KEY) != expected_metadata:
        raise RuntimeError(
            "Dense and RecStore sparse checkpoint identities do not match"
        )
    del dense_state
    gc.collect()


def _validate_current_config(module: torch.nn.Module, identity: Dict[str, Any]) -> None:
    current = {**_module_checkpoint_config(module), **_runtime_checkpoint_config()}
    for field in (
        "tables",
        "fusion",
        "clamp_ids",
        "need_indices",
        "optimizer",
        "sharding",
    ):
        if current[field] != identity[field]:
            raise RuntimeError(
                f"RecStore checkpoint {field} mismatch: "
                f"expected {_canonical_json(identity[field])}, "
                f"got {_canonical_json(current[field])}"
            )


def _checkpoint_client(module: torch.nn.Module) -> Any:
    client = getattr(module, "kv_client", None)
    if client is None:
        raise RuntimeError("RecStore sparse module has no kv_client")
    return client


class SparseState(Stateful):
    """
    Stateful wrapper for sparse (embedding) tensors in a model.

    This class implements the Stateful interface for distributed checkpointing,
    allowing sparse tensors to be saved and loaded separately from dense tensors.

    Args:
        model: The PyTorch model containing sparse tensors.
        sparse_tensor_keys: Set of keys identifying sparse tensors in the model's state dict.
    """

    def __init__(self, model: torch.nn.Module, sparse_tensor_keys: Set[str]) -> None:
        self.model = model
        self.sparse_tensor_keys = sparse_tensor_keys

    def state_dict(self) -> Dict[str, torch.Tensor]:
        out_dict: Dict[str, torch.Tensor] = {}
        is_sharded_tensor: Optional[bool] = None
        for k, v in self.model.state_dict().items():
            if k in self.sparse_tensor_keys:
                if is_sharded_tensor is None:
                    is_sharded_tensor = isinstance(v, ShardedTensor)
                assert is_sharded_tensor == isinstance(v, ShardedTensor)
                out_dict[k] = v
        return out_dict

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
        assert not incompatible_keys.unexpected_keys


def is_sparse_key(k: str, v: torch.Tensor) -> bool:
    return isinstance(v, ShardedTensor) or "embedding_collection" in k


def load_dense_state_dict(model: torch.nn.Module, state_dict: Dict[str, Any]) -> None:
    own_state = model.state_dict()
    own_state_dense_keys = {k for k, v in own_state.items() if not is_sparse_key(k, v)}
    state_dict_dense_keys = {
        k for k, v in state_dict.items() if not is_sparse_key(k, v)
    }
    assert own_state_dense_keys == state_dict_dense_keys, (
        f"expects {own_state_dense_keys} but gets {state_dict_dense_keys}"
    )
    for name in state_dict_dense_keys:
        param = state_dict[name]
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)


def _nonsparse_checkpoint_state(
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    sparse_tensor_keys: Set[str],
    recstore_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state = {
        "dense_dict": {
            k: v
            for k, v in model.state_dict().items()
            if not isinstance(v, ShardedTensor)
        },
        "optimizer_dict": optimizer.state_dict(),
        "class_metrics": {
            "train": [m.state_dict() for m in metric_logger.class_metrics["train"]],
            "eval": [m.state_dict() for m in metric_logger.class_metrics["eval"]],
        },
        "reg_metrics": {
            "train": [
                m.state_dict() for m in metric_logger.regression_metrics["train"]
            ],
            "eval": [
                m.state_dict() for m in metric_logger.regression_metrics["eval"]
            ],
        },
        "global_step": metric_logger.global_step,
        "sparse_tensor_keys": sparse_tensor_keys,
    }
    if recstore_metadata is not None:
        state[_RECSTORE_DENSE_KEY] = recstore_metadata
    return state


def _save_recstore_checkpoint(
    model: torch.nn.Module,
    module: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    rank: int,
    directory_step: int,
    checkpoint_root: str,
) -> None:
    _require_single_rank_recstore_checkpoint()
    if rank != 0:
        raise RuntimeError("Single-rank RecStore checkpoint save requires rank=0")
    checkpoint_path = os.path.join(checkpoint_root, str(directory_step))
    sparse_path = os.path.join(checkpoint_path, "sparse")
    non_sparse_path = os.path.join(checkpoint_path, "non_sparse.ckpt")
    manifest_path = os.path.join(sparse_path, _RECSTORE_MANIFEST)
    sparse_tensor_keys = {
        k for k, v in model.state_dict().items() if isinstance(v, ShardedTensor)
    }
    if sparse_tensor_keys:
        raise RuntimeError(
            "Mixed RecStore and TorchRec sparse checkpointing is unsupported"
        )
    os.makedirs(sparse_path, exist_ok=True)
    if os.path.exists(manifest_path):
        os.unlink(manifest_path)
    metadata = _build_recstore_metadata(
        module=module,
        checkpoint_root=checkpoint_root,
        directory_step=directory_step,
        global_step=_train_global_step(metric_logger),
    )
    save_checkpoint = getattr(_checkpoint_client(module), "save_checkpoint", None)
    if not callable(save_checkpoint):
        raise RuntimeError("RecStore client does not provide save_checkpoint()")
    save_checkpoint(sparse_path, _canonical_json(metadata))
    shards = _validate_shards(sparse_path, metadata["identity"])
    _atomic_torch_save(
        non_sparse_path,
        _nonsparse_checkpoint_state(
            model=model,
            optimizer=optimizer,
            metric_logger=metric_logger,
            sparse_tensor_keys=sparse_tensor_keys,
            recstore_metadata=metadata,
        ),
    )
    _atomic_write_json(manifest_path, {**metadata, "shards": shards})


@gin.configurable
def save_dmp_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    rank: int,
    batch_idx: int,
    path: str = "",
) -> None:
    """
    Save a distributed model checkpoint including sparse and dense components.

    Saves the model's sparse tensors using distributed checkpointing and dense
    tensors, optimizer state, and metrics using standard PyTorch serialization.

    Args:
        model: The model to checkpoint.
        optimizer: The optimizer whose state should be saved.
        metric_logger: The metrics logger containing training/eval metrics.
        rank: The current process rank in distributed training.
        batch_idx: The current batch index (used for checkpoint naming).
        path: Base path for saving the checkpoint. If empty, no checkpoint is saved.
    """
    if path == "":
        return
    recstore_module = _find_recstore_module(model)
    if recstore_module is not None:
        _save_recstore_checkpoint(
            model=model,
            module=recstore_module,
            optimizer=optimizer,
            metric_logger=metric_logger,
            rank=rank,
            directory_step=batch_idx,
            checkpoint_root=path,
        )
        print("checkpoint successfully saved")
        return

    path = os.path.join(path, str(batch_idx))
    if not os.path.exists(path) and rank == 0:
        os.makedirs(path)
    sparse_path = os.path.join(path, "sparse")
    if not os.path.exists(sparse_path) and rank == 0:
        os.makedirs(sparse_path)
    non_sparse_ckpt = os.path.join(path, "non_sparse.ckpt")

    sparse_tensor_keys = {
        k for k, v in model.state_dict().items() if isinstance(v, ShardedTensor)
    }
    if rank == 0:
        torch.save(
            _nonsparse_checkpoint_state(
                model=model,
                optimizer=optimizer,
                metric_logger=metric_logger,
                sparse_tensor_keys=sparse_tensor_keys,
            ),
            non_sparse_ckpt,
        )
    torch.distributed.barrier()
    sparse_dict = {"sparse_dict": SparseState(model, sparse_tensor_keys)}
    torch.distributed.checkpoint.save(
        sparse_dict,
        storage_writer=torch.distributed.checkpoint.FileSystemWriter(sparse_path),
    )
    torch.distributed.barrier()
    print("checkpoint successfully saved")


@gin.configurable
def load_sparse_checkpoint(
    model: torch.nn.Module,
    path: str = "",
) -> None:
    if path == "":
        return
    sparse_path = os.path.join(path, "sparse")
    recstore_module = _find_recstore_module(model)
    if recstore_module is not None:
        _require_single_rank_recstore_checkpoint()
        manifest = _load_manifest(path)
        metadata = _metadata_from_manifest(manifest, path)
        _validate_current_config(recstore_module, metadata["identity"])
        _validate_dense_checkpoint_metadata(path, metadata)
        load_checkpoint = getattr(
            _checkpoint_client(recstore_module), "load_checkpoint", None
        )
        if not callable(load_checkpoint):
            raise RuntimeError("RecStore client does not provide load_checkpoint()")
        load_checkpoint(sparse_path, _canonical_json(metadata))
        print("sparse checkpoint successfully loaded")
        return

    sparse_tensor_keys = {
        k for k, v in model.state_dict().items() if is_sparse_key(k, v)
    }
    sparse_dict = {"sparse_dict": SparseState(model, sparse_tensor_keys)}
    gc.collect()
    torch.distributed.checkpoint.load(
        sparse_dict,
        storage_reader=torch.distributed.checkpoint.FileSystemReader(sparse_path),
    )
    gc.collect()
    print("sparse checkpoint successfully loaded")


@gin.configurable
def load_nonsparse_checkpoint(
    model: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[Optimizer] = None,
    metric_logger: Optional[MetricsLogger] = None,
    path: str = "",
) -> None:
    """
    Load non-sparse (dense) components from a checkpoint.

    Loads dense model parameters, and optionally optimizer state and metrics.

    Args:
        model: The model to load dense parameters into.
        device: The device to load tensors onto.
        optimizer: Optional optimizer to restore state for.
        metric_logger: Optional metrics logger to restore state for.
        path: Base path of the checkpoint. If empty, no loading is performed.
    """
    if path == "":
        return
    non_sparse_ckpt = f"{path}/non_sparse.ckpt"

    non_sparse_state_dict = torch.load(non_sparse_ckpt, map_location=device)
    manifest_path = os.path.join(path, "sparse", _RECSTORE_MANIFEST)
    if os.path.exists(manifest_path):
        manifest = _load_manifest(path)
        expected_metadata = _metadata_from_manifest(manifest, path)
        saved_metadata = non_sparse_state_dict.get(_RECSTORE_DENSE_KEY)
        if saved_metadata != expected_metadata:
            raise RuntimeError(
                "Dense and RecStore sparse checkpoint identities do not match"
            )
    elif _RECSTORE_DENSE_KEY in non_sparse_state_dict:
        raise RuntimeError("RecStore sparse checkpoint manifest is missing")
    load_dense_state_dict(model, non_sparse_state_dict["dense_dict"])
    print("dense checkpoint successfully loaded")
    if optimizer is not None:
        optimizer.load_state_dict(non_sparse_state_dict["optimizer_dict"])
        print("optimizer checkpoint successfully loaded")
    if metric_logger is not None:
        metric_logger.global_step = non_sparse_state_dict["global_step"]
        class_metric_state_dict = non_sparse_state_dict["class_metrics"]
        regression_metric_state_dict = non_sparse_state_dict["reg_metrics"]
        for i, m in enumerate(metric_logger.class_metrics["train"]):
            m.load_state_dict(class_metric_state_dict["train"][i])
        for i, m in enumerate(metric_logger.class_metrics["eval"]):
            m.load_state_dict(class_metric_state_dict["eval"][i])
        for i, m in enumerate(metric_logger.regression_metrics["train"]):
            m.load_state_dict(regression_metric_state_dict["train"][i])
        for i, m in enumerate(metric_logger.regression_metrics["eval"]):
            m.load_state_dict(regression_metric_state_dict["eval"][i])


@gin.configurable
def load_dmp_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    device: torch.device,
    path: str = "",
) -> None:
    """
    Load a complete distributed model checkpoint (both sparse and dense components).

    This is a convenience function that calls both load_sparse_checkpoint and
    load_nonsparse_checkpoint.

    Args:
        model: The model to load the checkpoint into.
        optimizer: The optimizer to restore state for.
        metric_logger: The metrics logger to restore state for.
        device: The device to load tensors onto.
        path: Base path of the checkpoint. If empty, no loading is performed.
    """
    load_sparse_checkpoint(model=model, path=path)
    load_nonsparse_checkpoint(
        model=model,
        optimizer=optimizer,
        metric_logger=metric_logger,
        path=path,
        device=device,
    )
