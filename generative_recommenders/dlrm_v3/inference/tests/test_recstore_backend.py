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

import unittest
from unittest.mock import MagicMock, patch

from generative_recommenders.dlrm_v3.inference.model_family import (
    ModelFamilySparseDist,
)


class RecStoreInferenceBackendTest(unittest.TestCase):
    def test_load_uses_live_recstore_state(self) -> None:
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
        load_sparse_checkpoint_mock.assert_not_called()
        self.assertIs(manager.module, sparse_arch)

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
