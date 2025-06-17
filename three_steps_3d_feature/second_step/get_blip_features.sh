#!/bin/bash


export CUDA_DEVICE_ORDER="PCI_BUS_ID"

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7

RBG_DIR="/local1/leisongao/data/3dllm/rgb_features_eval/"

MASK_DIR_PATH="/local1/leisongao/data/3dllm/masks_eval/"

SAVE_PATH="/local1/leisongao/data/3dllm/blip_features_eval/"

python blip_sam.py --scene_dir_path $RBG_DIR --mask_dir_path $MASK_DIR_PATH --save_dir_path $SAVE_PATH