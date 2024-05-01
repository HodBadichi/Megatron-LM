#!/bin/bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --cap-add SYS_PTRACE --privileged --ulimit stack=67108864 -it -v ~/Megatron-LM:/workspace/megatron megatron:version1 bash -c "cd megatron && git config --global --add safe.directory /workspace/megatron"
