#!/bin/bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --cap-add SYS_PTRACE --privileged --ulimit stack=67108864 -it -v ~/Megatron-LM:/workspace/megatron nvcr.io/nvidia/pytorch:24.04-py3 bash
