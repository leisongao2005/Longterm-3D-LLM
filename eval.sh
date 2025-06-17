#!/bin/bash

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

python 3DLLM_BLIP2-base/evaluate.py --cfg-path 3DLLM_BLIP2-base/lavis/projects/blip2/train/finetune_longterm.yaml