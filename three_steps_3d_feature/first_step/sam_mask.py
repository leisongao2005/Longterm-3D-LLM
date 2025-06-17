import os
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from tqdm import tqdm, trange

import glob
import argparse
from tqdm import tqdm
import time
from multiprocessing import Process

mask_generator = None
save_dir = None
dataset_dir = None

def is_room_done(room):
    expected_images = glob.glob(os.path.join(dataset_dir, room, "*.jpg"))
    output_masks = glob.glob(os.path.join(save_dir, room, "*.pt"))
    return len(expected_images) == len(output_masks)


def process_chunk(rooms, gpu_id):
    torch.autograd.set_grad_enabled(False)

    sam = sam_model_registry["vit_h"](checkpoint=Path("sam_vit_h_4b8939.pth"))
    sam.to(device=f"cuda:{gpu_id}")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=8,
        pred_iou_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )

    os.makedirs(save_dir, exist_ok=True)

    print("Extracting SAM masks...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)


    for room in tqdm(rooms, position=gpu_id, desc=f"Worker {gpu_id+1}", dynamic_ncols=True):
        os.makedirs(save_dir + room, exist_ok=True)
        # dataset_path = dataset_dir + room + "/*png"
        dataset_path = dataset_dir + room + "/*jpg"
        data_list = glob.glob(dataset_path)

        for img_name in data_list:
            img_base_name = os.path.basename(img_name)

            try:
                savefile = os.path.join(
                    save_dir,
                    room,
                    # os.path.basename(img_name).replace(".png", ".pt"),
                    os.path.basename(img_name).replace(".jpg", ".pt"),
                )
                if os.path.exists(savefile):
                    continue

                imgfile = img_name
                img = cv2.imread(imgfile)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (512, 512))

                masks = mask_generator.generate(img)

                cur_mask = masks[0]["segmentation"]
                _savefile = os.path.join(
                    save_dir,
                    room,
                    os.path.splitext(os.path.basename(imgfile))[0] + ".pt",
                )

                mask_list = []
                for mask_item in masks:
                    mask_list.append(mask_item["segmentation"])

                mask_np = np.asarray(mask_list)
                mask_torch = torch.from_numpy(mask_np)
                torch.save(mask_torch, _savefile)

                del masks, mask_np, mask_torch, mask_list
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing {imgfile}: {e}")

def split_data(lst, n):
    """Split list `lst` into `n` approximately equal-sized chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def collect_unprocessed_rooms(dataset_dir, save_dir):
    unprocessed_rooms = []
    for room in os.listdir(dataset_dir):
        image_paths = glob.glob(os.path.join(dataset_dir, room, "*.jpg"))
        save_paths = glob.glob(os.path.join(save_dir, room, "*.pt"))
        if len(save_paths) < len(image_paths):
            unprocessed_rooms.append(room)
    return unprocessed_rooms


def main():
    parser = argparse.ArgumentParser(description="Specify dirs")
    parser.add_argument("--scene_dir_path", default="./masked_rdp_data/", type=str)
    parser.add_argument("--save_dir_path", default="./sam_masks/", type=str)
    args = parser.parse_args()

    global dataset_dir 
    global save_dir 

    dataset_dir = args.scene_dir_path
    save_dir = args.save_dir_path
    os.makedirs(save_dir, exist_ok=True)

    print("Extracting SAM masks...")
    jobs = collect_unprocessed_rooms(dataset_dir, save_dir)
    print(f"Total jobs: {len(jobs)}")

    # process_chunk(jobs, 0)
    # print("All jobs completed.")
    # return

    num_processes = 4
    chunks = split_data(jobs, num_processes)

    processes = []
    for i in range(num_processes):
        p = Process(target=process_chunk, args=(chunks[i], i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
