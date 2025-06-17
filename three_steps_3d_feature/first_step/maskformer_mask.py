import torch
import types
import os
from tqdm import tqdm
import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
import glob
from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
import matplotlib
import matplotlib.pyplot as plt
import argparse

import json


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


MASK2FORMER_CONFIG_FILE = "./maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
MASK2FORMER_WEIGHTS_FILE = "./model_final_e5f453.pkl"

import torch
import os
import types

def batch_process_images(rgb_list, save_dir, scan, MASK2FORMER_CONFIG_FILE, MASK2FORMER_WEIGHTS_FILE, LOAD_IMG_HEIGHT=512, LOAD_IMG_WIDTH=512, batch_size=8):
    """
    Process a batch of images and save the mask predictions for each image in the batch.

    Args:
        rgb_list (list): List of image file paths to process.
        save_dir (str): Directory where masks will be saved.
        scan (str): Scan identifier for constructing save paths.
        MASK2FORMER_CONFIG_FILE (str): Path to the Mask2Former config file.
        MASK2FORMER_WEIGHTS_FILE (str): Path to the Mask2Former weights file.
        LOAD_IMG_HEIGHT (int): Height to which images should be resized.
        LOAD_IMG_WIDTH (int): Width to which images should be resized.
        batch_size (int): Number of images to process in each batch.
    """
    # Load images in batches
    for i in range(0, len(rgb_list), batch_size):
        batch_rgb_images = []

        # Load the batch of images
        for img_name in rgb_list[i:i+batch_size]:
            try:
                IMGFILE = img_name
                MASK_LOAD_FILE = os.path.join(save_dir, scan, os.path.basename(img_name).replace(".jpg", ".pt"))
                
                # Read image in BGR format
                img = read_image(IMGFILE, format="BGR")
                batch_rgb_images.append(img)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")

        # Convert list of images to a numpy array (B, H, W, C) format
        if batch_rgb_images:
            batch_rgb_images = np.stack(batch_rgb_images, axis=0)  # Shape: (B, H, W, C)
            batch_rgb_images = batch_rgb_images.astype(np.uint8)

            # Set up the config and demo
            cfgargs = types.SimpleNamespace()
            cfgargs.config_file = MASK2FORMER_CONFIG_FILE
            cfgargs.opts = ["MODEL.WEIGHTS", MASK2FORMER_WEIGHTS_FILE]
            cfg = setup_cfg(cfgargs)
            demo = VisualizationDemo(cfg)

            # Run batch processing
            try:
                predictions_batch, vis_outputs_batch = demo.run_on_images_batch(batch_rgb_images)

                # Save results for each image in the batch
                for j, masks in enumerate(predictions_batch):
                    MASK_LOAD_FILE = os.path.join(save_dir, scan, os.path.basename(rgb_list[i+j]).replace(".jpg", ".pt"))
                    mask = torch.nn.functional.interpolate(
                        masks["instances"].pred_masks.unsqueeze(0), [LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH], mode="nearest"
                    )
                    mask = mask.half()
                    torch.save(mask[0].detach().cpu(), MASK_LOAD_FILE)
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")



if __name__ == "__main__":
    torch.autograd.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Specify dirs")
    parser.add_argument("--scene_dir_path", default="./masked_rdp_data/", type=str)
    parser.add_argument("--save_dir_path", default="./maskformer_masks/", type=str)
    args = parser.parse_args()

    scene_dir = args.scene_dir_path
    save_dir = args.save_dir_path

    os.makedirs(os.path.join(save_dir), exist_ok=True)
    # Set up the config and demo
    cfgargs = types.SimpleNamespace()
    cfgargs.config_file = MASK2FORMER_CONFIG_FILE
    cfgargs.opts = ["MODEL.WEIGHTS", MASK2FORMER_WEIGHTS_FILE]
    cfg = setup_cfg(cfgargs)
    demo = VisualizationDemo(cfg, parallel=True)

    for scan in tqdm(os.listdir(scene_dir)):
        os.makedirs(os.path.join(save_dir, scan), exist_ok=True)

        rgb_list = glob.glob(os.path.join(scene_dir, scan, "*jpg"))
        # print(rgb_list)

        batch_size = 8
        
        # Load images in batches
        for i in range(0, len(rgb_list), batch_size):
            batch_rgb_images = []
            LOAD_IMG_HEIGHT = 512
            LOAD_IMG_WIDTH = 512

            # Load the batch of images
            for img_name in rgb_list[i:i+batch_size]:
                try:
                    IMGFILE = img_name
                    MASK_LOAD_FILE = os.path.join(save_dir, scan, os.path.basename(img_name).replace(".jpg", ".pt"))
                    
                    # Read image in BGR format
                    img = read_image(IMGFILE, format="BGR")
                    batch_rgb_images.append(img)
                except Exception as e:
                    print(f"Error loading image {img_name}: {e}")

            # Convert list of images to a numpy array (B, H, W, C) format
            if batch_rgb_images:
                batch_rgb_images = np.stack(batch_rgb_images, axis=0)  # Shape: (B, H, W, C)
                batch_rgb_images = batch_rgb_images.astype(np.uint8)

                # Run batch processing
                try:
                    print("got here")
                    print(batch_rgb_images.shape)
                    predictions_batch, vis_outputs_batch = demo.run_on_batch(batch_rgb_images)
                    print("got predictions")

                    # Save results for each image in the batch
                    for j, masks in enumerate(predictions_batch):
                        MASK_LOAD_FILE = os.path.join(save_dir, scan, os.path.basename(rgb_list[i+j]).replace(".jpg", ".pt"))
                        mask = torch.nn.functional.interpolate(
                            masks["instances"].pred_masks.unsqueeze(0), [LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH], mode="nearest"
                        )
                        mask = mask.half()
                        torch.save(mask[0].detach().cpu(), MASK_LOAD_FILE)
                except Exception as e:
                    print(f"Error processing batch starting at index {i}: {e}")



        # for img_name in rgb_list:
        #     try:
        #         IMGFILE = img_name
        #         MASK_LOAD_FILE = os.path.join(save_dir, scan, os.path.basename(img_name).replace(".jpg", ".pt"))
        #         LOAD_IMG_HEIGHT = 512
        #         LOAD_IMG_WIDTH = 512

        #         cfgargs = types.SimpleNamespace()
        #         cfgargs.imgfile = IMGFILE
        #         cfgargs.config_file = MASK2FORMER_CONFIG_FILE
        #         cfgargs.opts = ["MODEL.WEIGHTS", MASK2FORMER_WEIGHTS_FILE]

        #         cfg = setup_cfg(cfgargs)
        #         demo = VisualizationDemo(cfg)

        #         img = read_image(IMGFILE, format="BGR")

        #         predictions, visualized_output = demo.run_on_image(img)
        #         masks = torch.nn.functional.interpolate(
        #             predictions["instances"].pred_masks.unsqueeze(0), [LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH], mode="nearest"
        #         )
        #         masks = masks.half()
        #         torch.save(masks[0].detach().cpu(), MASK_LOAD_FILE)
        #     except Exception as e:
        #         print(f"Excepting parsing {img_name} {e}")