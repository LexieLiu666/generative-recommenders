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

# pyre-unsafe

import abc

import torch
from generative_recommenders.research.modeling.initialization import truncated_normal

try:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
except ModuleNotFoundError:  # pragma: no cover - dependency is required at runtime
    KeyedJaggedTensor = None


class EmbeddingModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class RecStoreEmbeddingModule(EmbeddingModule):
    """Sequential ``[... ] -> [..., D]`` adapter backed by RecStore."""

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        table_name: str = "hstu_items",
        initialize_values: bool = False,
    ) -> None:
        super().__init__()
        if KeyedJaggedTensor is None:
            raise RuntimeError("RecStoreEmbeddingModule requires torchrec")
        try:
            from torchrec.modules.embedding_configs import EmbeddingConfig
            from torchrec_kv import RecStoreEmbeddingCollection
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "RecStoreEmbeddingModule requires RecStore/src/python/pytorch "
                "on PYTHONPATH"
            ) from error

        self._item_embedding_dim = int(item_embedding_dim)

        def init_func(shape, dtype):
            item_emb = torch.nn.Embedding(
                shape[0], shape[1], padding_idx=0, dtype=dtype
            )
            truncated_normal(item_emb.weight, mean=0.0, std=0.02)
            return item_emb.weight.detach()

        self._embedding_collection = RecStoreEmbeddingCollection(
            tables=[
                EmbeddingConfig(
                    name=table_name,
                    embedding_dim=self._item_embedding_dim,
                    num_embeddings=int(num_items) + 1,
                    feature_names=["item_id"],
                )
            ],
            need_indices=False,
            initialize_values=initialize_values,
            init_func=init_func if initialize_values else None,
        )

    @property
    def recstore_embedding_collection(self) -> torch.nn.Module:
        return self._embedding_collection

    def debug_str(self) -> str:
        return f"recstore_emb_d{self._item_embedding_dim}"

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        original_shape = item_ids.shape
        flat_ids = item_ids.reshape(-1).to(dtype=torch.int64)
        lengths = torch.ones(
            (flat_ids.numel(),), dtype=torch.int32, device=flat_ids.device
        )
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["item_id"], values=flat_ids, lengths=lengths
        )
        embeddings = self._embedding_collection(features)["item_id"].values()
        embeddings = embeddings.reshape(*original_shape, self._item_embedding_dim)
        if item_ids.numel() > 0:
            padding = item_ids.to(device=embeddings.device).unsqueeze(-1).eq(0)
            embeddings = torch.where(padding, embeddings.detach(), embeddings)
        return embeddings

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.get_item_embeddings(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class CategoricalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        # pyrefly: ignore [bad-index]
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
