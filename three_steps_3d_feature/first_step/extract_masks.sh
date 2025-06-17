#!/bin/bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7

RBG_DIR="/local1/leisongao/data/3dllm/rgb_features_eval/"

SAVE_PATH="/local1/leisongao/data/3dllm/masks_eval/"


# python maskformer_mask.py --scene_dir_path $RBG_DIR --save_dir_path $SAVE_PATH
python sam_mask.py --scene_dir_path $RBG_DIR --save_dir_path $SAVE_PATH