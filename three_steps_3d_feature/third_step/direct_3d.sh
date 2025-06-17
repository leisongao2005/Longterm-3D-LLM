#!/bin/bash

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

export OMP_NUM_THREADS=4
# export CUDA_VISIBLE_DEVICES=0

RBG_DIR="/local1/leisongao/data/3dllm/rgb_features_eval/"

DEPTH_DIR="/local1/leisongao/data/3dllm/depth_features_eval/"

FEATURE_DIR="/local1/leisongao/data/3dllm/blip_features_eval/"

python direct_3d.py --data_dir_path $RBG_DIR --depth_dir_path $DEPTH_DIR --feat_dir_path $FEATURE_DIR