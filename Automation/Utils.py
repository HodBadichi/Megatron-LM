import os
import torch
from datetime import timedelta


def is_faulty_gpu():
    current_gpu_rank = torch.distributed.get_rank()
    faulty_gpu_rank = get_faulty_gpu_rank()
    return current_gpu_rank == faulty_gpu_rank


def is_detached_gpu_on():
    return int(os.getenv("DetachedGPU_ON", "0"))


def get_failed_iteration():
    return int(os.getenv('FAILED_ITERATION'))


def set_detached_gpu_on():
    os.environ['DetachedGPU_ON'] = "1"


def get_faulty_gpu_rank():
    return int(os.getenv("FAILED_GPU"))


def get_new_working_group():
    timeout = timedelta(minutes=0.0001)  # 8 seconds
    ranks = [0]
    return torch.distributed.new_group(ranks, timeout=timeout)

