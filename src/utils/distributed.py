from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedState:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


def _cuda_device_from_args(args: Any, local_rank: int) -> torch.device:
    gpu_id = int(getattr(args, "gpu_id", -1) if getattr(args, "gpu_id", -1) is not None else -1)
    if not torch.cuda.is_available() or gpu_id < 0:
        return torch.device("cpu")

    device_index = local_rank if bool(getattr(args, "distributed", False)) else gpu_id
    torch.cuda.set_device(device_index)
    return torch.device(f"cuda:{device_index}")


def setup_distributed(args: Any) -> DistributedState:
    local_rank = int(getattr(args, "local_rank", 0) or 0)
    rank = int(getattr(args, "rank", 0) or 0)
    world_size = int(getattr(args, "world_size", 1) or 1)

    if bool(getattr(args, "distributed", False)):
        import os

        local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
        rank = int(os.environ.get("RANK", rank))
        world_size = int(os.environ.get("WORLD_SIZE", world_size))

    device = _cuda_device_from_args(args, local_rank)
    use_distributed = bool(getattr(args, "distributed", False)) and world_size > 1

    if use_distributed and not dist.is_initialized():
        backend = str(getattr(args, "dist_backend", "nccl") or "nccl")
        if backend == "nccl" and device.type != "cuda":
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        use_distributed = world_size > 1

    return DistributedState(
        distributed=use_distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def is_main_process(state: DistributedState) -> bool:
    return int(state.rank) == 0


def log0(state: DistributedState, message: str) -> None:
    if is_main_process(state):
        print(message, flush=True)


def barrier(state: DistributedState) -> None:
    if state.distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_scalars(
    state: DistributedState,
    value: float,
    improved: bool,
    should_stop: bool,
) -> Tuple[float, bool, bool]:
    if not state.distributed or not dist.is_available() or not dist.is_initialized():
        return float(value), bool(improved), bool(should_stop)

    payload = torch.tensor(
        [float(value), float(bool(improved)), float(bool(should_stop))],
        dtype=torch.float64,
        device=state.device,
    )
    dist.broadcast(payload, src=0)
    return (
        float(payload[0].item()),
        bool(round(float(payload[1].item()))),
        bool(round(float(payload[2].item()))),
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
