#!/bin/bash

pip install nltk
pip install datasets

python3 preprocess_helper.py



python tools/preprocess_data.py \
       --input wikipedia.json \
       --output-prefix wikipedia \
       --vocab-file vocab.json \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file merges.txt \
       --json-keys text \
       --workers 32 \
       --append-eod
