#!/usr/bin/env python3

import argparse
import os

import torch
from torchrec.optim import RowWiseAdagrad

from generative_recommenders.research.modeling.sequential.embedding_modules import (
    LocalEmbeddingModule,
    RecStoreEmbeddingModule,
)
from recstore.optimizer import SparseRowWiseAdagrad


def _check(label: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    expected = expected.detach().cpu()
    actual = actual.detach().cpu()
    if not torch.allclose(expected, actual, rtol=1e-6, atol=1e-7):
        error = (expected - actual).abs().max().item()
        raise AssertionError(f"{label} mismatch: max_abs_error={error:.10g}")
    error = (expected - actual).abs().max().item()
    print(f"{label}=PASS max_abs_error={error:.10g}")


def _trace_grads(module, embedding_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    trace = module.recstore_embedding_collection._trace
    ids = torch.cat([entry["ids"].detach().cpu() for entry in trace])
    grads = torch.cat([entry["grads"].detach().cpu() for entry in trace])
    unique_ids, inverse = torch.unique(ids, return_inverse=True)
    summed = torch.zeros((unique_ids.numel(), embedding_dim), dtype=torch.float32)
    summed.index_add_(0, inverse, grads)
    return unique_ids, summed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-items", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epsilon", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--table-name", default=f"hstu_alignment_{os.getpid()}")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    local = LocalEmbeddingModule(args.num_items, args.embedding_dim)
    torch.manual_seed(args.seed)
    recstore = RecStoreEmbeddingModule(
        args.num_items,
        args.embedding_dim,
        table_name=args.table_name,
        initialize_values=True,
    )

    device = torch.device(args.device)
    local = local.to(device)
    recstore = recstore.to(device)
    all_ids = torch.arange(args.num_items + 1, dtype=torch.int64)
    initial = recstore.recstore_embedding_collection.kv_client.pull(
        args.table_name, all_ids
    )
    _check("initial_weights", local._item_emb.weight, initial)

    local_optimizer = RowWiseAdagrad(
        [local._item_emb.weight], lr=args.learning_rate, eps=args.epsilon
    )
    recstore_optimizer = SparseRowWiseAdagrad(
        [recstore.recstore_embedding_collection],
        lr=args.learning_rate,
        eps=args.epsilon,
    )
    item_ids = torch.tensor(
        [[0, 1, 2, 2], [3, 0, 4, 2]], dtype=torch.int64, device=device
    )

    for step in range(2):
        local_optimizer.zero_grad()
        recstore_optimizer.zero_grad()
        local_output = local.get_item_embeddings(item_ids)
        recstore_output = recstore.get_item_embeddings(item_ids)
        _check(f"step_{step}_forward", local_output, recstore_output)

        upstream = torch.arange(
            local_output.numel(), dtype=torch.float32, device=device
        ).reshape_as(local_output)
        upstream = upstream.div(local_output.numel()).add(step + 1)
        local_output.backward(upstream)
        recstore_output.backward(upstream)

        unique_ids, recstore_grads = _trace_grads(recstore, args.embedding_dim)
        local_grads = local._item_emb.weight.grad.detach().cpu()[unique_ids]
        _check(f"step_{step}_gradients", local_grads, recstore_grads)

        local_optimizer.step()
        recstore_optimizer.step()
        recstore_optimizer.flush()
        updated = recstore.recstore_embedding_collection.kv_client.pull(
            args.table_name, unique_ids
        )
        _check(f"step_{step}_updated_weights", local._item_emb.weight[unique_ids], updated)

    print("HSTU_RECSTORE_ALIGNMENT=PASS")


if __name__ == "__main__":
    main()
