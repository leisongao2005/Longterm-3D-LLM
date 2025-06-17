#!/bin/bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7

YAML_FILE=finetune_longterm_embodied_task.yaml

torchrun --nnodes=1 --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/$YAML_FILE


