#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
GPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
VOCAB_FILE=vocab.json
MERGE_FILE=merges.txt
DATA_PATH=wikipedia_text_document
GPT_ARGS="--num-layers 12
--hidden-size 768
--num-attention-heads 12
--seq-length 1024
--max-position-embeddings 1024
--micro-batch-size 12
--global-batch-size 192
--lr 0.0005
--train-iters 150000
--lr-decay-iters 150000
--lr-decay-style cosine
--lr-warmup-iters 2000
--weight-decay .1
--adam-beta2 .999
--fp16
--log-interval 10
--save-interval 2000
--eval-interval 200
--eval-iters 10
--attention-softmax-in-fp32
--use-mcore-models
"

export FAILED_GPU=1
export FAILED_ITERATION=5
export ZeroGPU_TEST_MODE=1
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
        ../pretrain_gpt.py \
        --tensor-model-parallel-size 2 \
        --pipeline-model-parallel-size 1 \
        $GPT_ARGS \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --data-path $DATA_PATH \
        $TENSORBOARD_ARGS
