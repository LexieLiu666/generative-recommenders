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

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from generative_recommenders.dlrm_v3.checkpoint import (
    load_nonsparse_checkpoint,
    load_sparse_checkpoint,
    save_dmp_checkpoint,
)
from generative_recommenders.dlrm_v3.inference.main import Runner
from generative_recommenders.dlrm_v3.inference.model_family import (
    ModelFamilySparseDist,
)


class _CheckpointClient:
    def __init__(self) -> None:
        self.loaded_metadata = None

    def save_checkpoint(self, path: str, metadata: str) -> None:
        checkpoint_path = os.path.dirname(path)
        if os.path.exists(os.path.join(checkpoint_path, "non_sparse.ckpt")):
            raise AssertionError("dense checkpoint was committed before sparse state")
        if os.path.exists(os.path.join(path, "recstore_manifest.json")):
            raise AssertionError("manifest was committed before sparse state")
        with open(os.path.join(path, "recstore-shard-0.bin"), "wb") as output:
            output.write(metadata.encode("ascii"))

    def load_checkpoint(self, path: str, metadata: str) -> None:
        self.loaded_metadata = json.loads(metadata)


class _RecStoreModule(torch.nn.Module):
    is_recstore_sparse_module = True

    def __init__(self, client: _CheckpointClient) -> None:
        super().__init__()
        self.kv_client = client

    def recstore_checkpoint_config(self):
        return {
            "tables": [
                {
                    "name": "table",
                    "num_embeddings": 16,
                    "embedding_dim": 4,
                    "base_offset": 0,
                    "feature_names": ["feature"],
                    "embedding_names": ["feature"],
                }
            ],
            "fusion": {"enabled": True, "fusion_k": 30},
            "clamp_ids": False,
            "need_indices": False,
        }


class _Model(torch.nn.Module):
    def __init__(self, client: _CheckpointClient, dense_value: float) -> None:
        super().__init__()
        self.embedding_collection = _RecStoreModule(client)
        self.dense = torch.nn.Parameter(torch.tensor([dense_value]))


class RecStoreInferenceBackendTest(unittest.TestCase):
    def test_runner_preserves_prediction_error(self) -> None:
        prediction_error = RuntimeError("RecStore lookup failed")
        model = MagicMock()
        model.predict.side_effect = prediction_error
        runner = Runner(
            model=model,
            ds=MagicMock(),
            num_queries=2,
            data_producer_threads=1,
        )
        query = SimpleNamespace(query_ids=[101, 102], samples=MagicMock())

        with patch(
            "generative_recommenders.dlrm_v3.inference.main.lg.QuerySampleResponse",
            side_effect=lambda query_id, data, size: (query_id, data, size),
        ), patch(
            "generative_recommenders.dlrm_v3.inference.main.lg.QuerySamplesComplete"
        ) as complete:
            runner.run_one_item(query)

        complete.assert_called_once_with([(101, 0, 0), (102, 0, 0)])
        self.assertEqual(runner.result_timing, [])
        self.assertEqual(runner.result_batches, [])
        runner.finish()
        with self.assertRaisesRegex(RuntimeError, "inference prediction failed") as caught:
            runner.raise_if_failed()
        self.assertIs(caught.exception.__cause__, prediction_error)

    def test_load_restores_recstore_checkpoint(self) -> None:
        hstu_config = MagicMock()
        table_config = {"table": MagicMock()}
        manager = ModelFamilySparseDist(
            hstu_config=hstu_config,
            table_config=table_config,
            embedding_collection_backend="recstore",
        )
        sparse_arch = MagicMock()

        with patch(
            "generative_recommenders.dlrm_v3.inference.model_family."
            "HSTUSparseInferenceModule",
            return_value=sparse_arch,
        ) as module_factory, patch(
            "generative_recommenders.dlrm_v3.inference.model_family."
            "load_sparse_checkpoint"
        ) as load_sparse_checkpoint_mock:
            manager.load("checkpoint")

        module_factory.assert_called_once_with(
            table_config=table_config,
            hstu_config=hstu_config,
            embedding_collection_backend="recstore",
            recstore_initialize_values=False,
        )
        load_sparse_checkpoint_mock.assert_called_once_with(
            model=sparse_arch._hstu_model,
            path="checkpoint",
        )
        self.assertIs(manager.module, sparse_arch)

    def test_checkpoint_round_trip_and_config_validation(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            config_path = os.path.join(root, "recstore.json")
            config = {
                "cache_ps": {
                    "num_shards": 1,
                    "servers": [{"shard": 0}],
                    "optimizer": {
                        "type": "RowWiseAdagrad",
                        "learning_rate": 0.001,
                        "epsilon": 1e-8,
                    },
                },
                "distributed_client": {
                    "num_shards": 1,
                    "hash_method": "city_hash",
                    "servers": [{"shard": 0}],
                },
            }
            with open(config_path, "w", encoding="utf-8") as output:
                json.dump(config, output)

            source_client = _CheckpointClient()
            source = _Model(source_client, dense_value=3.0)
            source_optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
            metrics = SimpleNamespace(
                global_step={"train": 10, "eval": 0},
                class_metrics={"train": [], "eval": []},
                regression_metrics={"train": [], "eval": []},
            )
            with patch.dict(os.environ, {"RECSTORE_CONFIG": config_path}):
                save_dmp_checkpoint(
                    model=source,
                    optimizer=source_optimizer,
                    metric_logger=metrics,
                    rank=0,
                    batch_idx=10,
                    path=root,
                )

                checkpoint_path = os.path.join(root, "10")
                sparse_path = os.path.join(checkpoint_path, "sparse")
                self.assertFalse(
                    any(name.endswith(".distcp") for name in os.listdir(sparse_path))
                )
                with open(
                    os.path.join(sparse_path, "recstore_manifest.json"),
                    "r",
                    encoding="utf-8",
                ) as manifest_file:
                    manifest = json.load(manifest_file)
                dense_state = torch.load(
                    os.path.join(checkpoint_path, "non_sparse.ckpt"),
                    map_location="cpu",
                )
                expected_metadata = {
                    "identity": manifest["identity"],
                    "checkpoint_id": manifest["checkpoint_id"],
                }
                self.assertEqual(
                    dense_state["recstore_checkpoint"], expected_metadata
                )

                restored_client = _CheckpointClient()
                restored = _Model(restored_client, dense_value=0.0)
                restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1)
                load_sparse_checkpoint(model=restored, path=checkpoint_path)
                load_nonsparse_checkpoint(
                    model=restored,
                    optimizer=restored_optimizer,
                    metric_logger=None,
                    path=checkpoint_path,
                    device=torch.device("cpu"),
                )
                self.assertEqual(restored_client.loaded_metadata, expected_metadata)
                torch.testing.assert_close(restored.dense, source.dense)

                config["cache_ps"]["optimizer"]["learning_rate"] = 0.5
                with open(config_path, "w", encoding="utf-8") as output:
                    json.dump(config, output)
                mismatch_client = _CheckpointClient()
                mismatch = _Model(mismatch_client, dense_value=0.0)
                with self.assertRaisesRegex(RuntimeError, "optimizer mismatch"):
                    load_sparse_checkpoint(model=mismatch, path=checkpoint_path)
                self.assertIsNone(mismatch_client.loaded_metadata)

    def test_recstore_rejects_torchrec_sparse_quantization(self) -> None:
        with self.assertRaisesRegex(ValueError, "only supported by TorchRec"):
            ModelFamilySparseDist(
                hstu_config=MagicMock(),
                table_config={},
                quant=True,
                embedding_collection_backend="recstore",
            )


if __name__ == "__main__":
    unittest.main()
