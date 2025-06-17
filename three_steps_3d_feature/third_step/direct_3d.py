# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import argparse

import numpy as np
import copy
import cv2

import time
import torch

import quaternion
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import sys
from tqdm import tqdm
from multiprocessing import Pool

import shutil


from tools import Application

# is this the only import necessary?
# from habitat.utils.geometry_utils import quaternion_to_list
def quaternion_to_list(q: quaternion.quaternion):
    return q.imag.tolist() + [q.real]



def robot2world(position, u, v, heading):
    x0, y0, z0 = position
    x1 = x0 + v * np.cos(heading + np.pi / 2)
    z1 = -(-z0 + v * np.sin(heading + np.pi / 2))
    x2 = x1 + u * np.cos(heading + np.pi / 2 - np.pi / 2)
    z2 = -(-z1 + u * np.sin(heading + np.pi / 2 - np.pi / 2))
    return [x2, y0, z2]


def transformation_quatrtnion2heading(rotation: quaternion):
    quat = quaternion_to_list(rotation)
    q = R.from_quat(quat)
    heading = q.as_rotvec()[1]
    return heading


def main(room_name, data_root_dir, depth_dir, feat_dir, sample_num):
    feature_dir = os.path.join(feat_dir, room_name)
    data_dir = os.path.join(data_root_dir, room_name)

    print("starting processing: ", room_name)

    # print("does the path exist:", os.path.exists(os.path.join(feat_dir, room_name)))
    try:
        if (
            os.path.exists(os.path.join(data_dir, "pcd_feat.pt"))
            and torch.load(os.path.join(data_dir, "pcd_feat.pt"), weights_only=False).shape[0] == sample_num
        ):
            print("already processed", room_name, "removing now ...")
            if (os.path.exists(os.path.join(feat_dir, room_name))):
                shutil.rmtree(os.path.join(feat_dir, room_name))
                print("removed the dir")
            
            print(f"exiting main for {room_name} ...")
            return
    except Exception as e:
        print(e)

    print(f"{room_name} was not processed yet, contining ... ")

    depth_dir = os.path.join(depth_dir, room_name)
    api = Application((512, 512), 90, 1, 0.005, 600, 1.5, 1, 2)
    pc_pos_list = []
    pc_feat_list = []


    # for file in os.listdir(feature_dir):
    for file in tqdm(os.listdir(feature_dir), desc="single room", position=1, dynamic_ncols=True):
        try:
            feature_map = torch.load(os.path.join(feature_dir, file)).detach().cpu().numpy()
        except Exception as e:
            print(e)
            continue

        pose_file = json.load(open(os.path.join(data_dir, file.replace("image_", "pose_").replace(".pt", ".json"))))

        house_name = room_name.split("_")[0]
        ky = room_name.split("_")[1]

        # changed to our bboxes
        # bbox = json.load(open(os.path.join("room_bboxes_with_walls_revised_axis", house_name + ".json")))[ky]
        bbox = json.load(open(os.path.join("/local1/whu/data/hm3d/new_hm3d_obj_bbox/room_bboxes_revised_axis_200", house_name + ".json")))[ky]

        
        min_x = bbox[0][0]
        min_y = bbox[0][1]
        min_z = bbox[0][2]
        max_x = bbox[1][0]
        max_y = bbox[1][1]
        max_z = bbox[1][2]

        # old data format
        # rotation_0 = pose_file["rotation"][0]
        # rotation_1 = pose_file["rotation"][1]
        # rotation_2 = pose_file["rotation"][2]
        # rotation_3 = pose_file["rotation"][3]
        # position = pose_file["translation"]
        # heading = transformation_quatrtnion2heading(np.quaternion(rotation_0, rotation_1, rotation_2, rotation_3))

        # using our data format
        rotation_0 = [i[0] for i in pose_file["pose"][:3]]
        rotation_1 = [i[1] for i in pose_file["pose"][:3]]
        rotation_2 = [i[2] for i in pose_file["pose"][:3]]
        position = [i[3] for i in pose_file["pose"][:3]]
        heading = transformation_quatrtnion2heading(quaternion.from_rotation_matrix([rotation_0, rotation_1, rotation_2]))

        if heading > np.pi * 2:
            heading -= np.pi * 2
        elif heading < 0:
            heading += np.pi * 2
        depth_map = np.load(os.path.join(depth_dir, file.replace("image_", "depth_").replace(".pt", ".npy")))
        point_clouds_2current = api.transformation_camera2robotcamera(np.expand_dims(depth_map / 10.0, axis=2))
        # color_map = cv2.imread(os.path.join(data_dir, file.replace(".pt", ".png")))
        color_map = cv2.imread(os.path.join(data_dir, file.replace(".pt", ".jpg")))

        pc_pos = []
        pc_feat = []
        
        for w in range(point_clouds_2current.shape[0]):
            for h in range(point_clouds_2current.shape[1]):
                if np.count_nonzero(feature_map[w, h]) == 0:
                    continue
                if color_map[w, h, 0] == 0 and color_map[w, h, 1] == 0 and color_map[w, h, 2] == 0:
                    continue

                pc2r = [point_clouds_2current[w, h, j] for j in range(point_clouds_2current.shape[-1])]

                pc2w = robot2world(position, pc2r[0] * 10, pc2r[1] * 10, heading)
                pc2w[1] = pc2r[2] * 10 + pc2w[1]

                if not (
                    (min_x - 0 < pc2w[0] < max_x + 0)
                    and (min_y - 0 < pc2w[1] < max_y + 0)
                    and (min_z - 0 < pc2w[2] < max_z + 0)
                ):
                    continue
                else:
                    pc_pos.append(pc2w)
                    pc_feat.append(feature_map[w, h])

        
        if pc_pos:
            print(f"pos shape: {len(pc_pos)}, feat shape: {len(pc_feat)}")
            pc_pos_list.append(np.array(pc_pos))
            pc_feat_list.append(np.array(pc_feat))

    print(" finished for loop")
    t1 = time.time()
    pc_pos = np.concatenate(pc_pos_list, axis=0)
    pc_feat = np.concatenate(pc_feat_list, axis=0)
    t2 = time.time()

    print(f"time to convert arrays: {t2 - t1}")
    if len(pc_pos) > sample_num:
        N = len(pc_pos)
        indices = np.arange(N)
        np.random.shuffle(indices)
        indices = indices[:sample_num]
        # indices = np.random.choice(N, sample_num, replace=False)
        final_points = pc_pos[indices]
        final_features = pc_feat[indices]
    else:
        final_points = pc_pos
        final_features = pc_feat
    # print(final_points.shape)

    NUM_SAMPLES=40

    for i in tqdm(range(NUM_SAMPLES), desc="sampling room", position=2, dynamic_ncols=True):
    # for i in range(NUM_SAMPLES):
        idxes = torch.sort(torch.randperm(pc_feat.shape[0])[:5000])[1]
        pc_feat_sample = final_features[idxes]
        pc_sample = final_points[idxes]

        torch.save(pc_feat_sample, os.path.join(data_dir, f"pcd_feat_{i}.pt"))
        torch.save(pc_sample, os.path.join(data_dir, f"pcd_pos_{i}.pt"))

    # torch.save(final_points, os.path.join(data_dir, "pcd_pos.pt"))
    # torch.save(final_features, os.path.join(data_dir, "pcd_feat.pt"))

    print("finished processing", room_name, "removing now ...")
    if (os.path.exists(os.path.join(feat_dir, room_name))):
        shutil.rmtree(os.path.join(feat_dir, room_name))
        print("removed the dir")

    # print("acquring lock")
    # with write_lock:
    #     torch.save(final_points, os.path.join(data_dir, "pcd_pos.pt"))
    #     torch.save(final_features, os.path.join(data_dir, "pcd_feat.pt"))
    # print("finished processing: ", room_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify dirs")
    parser.add_argument("--data_dir_path", default="./masked_rdp_data/", type=str)
    parser.add_argument("--depth_dir_path", default="./masked_rdp_data/", type=str)
    parser.add_argument("--feat_dir_path", default="./maskformer_masks/", type=str)
    parser.add_argument("--sample_num", default=300000, type=int)
    args = parser.parse_args()

    room_list = sorted(os.listdir(args.feat_dir_path))

    # PROCESS_THRESH = 500 # only processes scenes up and not including to this id

    #  # Filter rooms by threshold
    # filtered_rooms = []
    # for room_name in room_list:
    #     room_id = int(room_name[:5])
    #     if room_id < PROCESS_THRESH:
    #         continue
    #     else:
    #         filtered_rooms.append(room_name)

    # print(len(filtered_rooms))
    # write_lock = Lock()


    def main_wrapper(room_name):
        # Replace with your actual `main` function
        return main(room_name, args.data_dir_path, args.depth_dir_path, args.feat_dir_path, args.sample_num)
    
    with Pool(processes=8) as pool:
        list(tqdm(
            pool.imap_unordered(main_wrapper, room_list),
            total=len(room_list), position=0,
            dynamic_ncols=True,
            desc="Processing rooms"
        ))

    # for room_name in tqdm(room_list, desc="all rooms", position=0, dynamic_ncols=True):
    #     main(room_name, args.data_dir_path, args.depth_dir_path, args.feat_dir_path, args.sample_num)
